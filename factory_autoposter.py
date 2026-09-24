#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import plistlib
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator

from montage import engine


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RENDERED_DIR = Path.home() / "Desktop" / "factory" / "rendered"
DEFAULT_STATE_DIR = Path.home() / "Desktop" / "factory" / "autoposter"
CATEGORIES = ("hook", "main", "final")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".webm")
PUBLISH_INTERVAL = timedelta(hours=5)
AGENT_LABEL = "com.automontaje.factory-autoposter"
PUBLISH_NOW_REQUEST = "publish-now.request"
CONTROL_FILE = "control.json"
CONTROL_CHANGED_REQUEST = "control-changed.request"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def redact(text: str) -> str:
    return re.sub(
        r"(?i)((?:access[_ -]?token|authorization|bearer)\s*[=:]\s*)([^\s&,]+)",
        r"\1<REDACTED>",
        str(text),
    )


def posting_enabled(state_dir: Path) -> bool:
    try:
        payload = json.loads((state_dir / CONTROL_FILE).read_text(encoding="utf-8"))
        return bool(payload.get("posting_enabled", True))
    except (OSError, ValueError, json.JSONDecodeError):
        return True


def set_posting_enabled(state_dir: Path, enabled: bool) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    destination = state_dir / CONTROL_FILE
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"posting_enabled": bool(enabled)}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    if not enabled:
        (state_dir / PUBLISH_NOW_REQUEST).unlink(missing_ok=True)
    (state_dir / CONTROL_CHANGED_REQUEST).write_text(iso_now(), encoding="utf-8")
    update_waiting_progress(state_dir)
    return {"posting_enabled": bool(enabled)}


def connect_database(state_dir: Path) -> sqlite3.Connection:
    state_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_dir / "combinations.sqlite3")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS combinations (
            hook INTEGER NOT NULL,
            main INTEGER NOT NULL,
            final INTEGER NOT NULL,
            published INTEGER NOT NULL DEFAULT 0 CHECK (published IN (0, 1)),
            attempts INTEGER NOT NULL DEFAULT 0,
            last_attempt_at TEXT,
            published_at TEXT,
            media_id TEXT,
            permalink TEXT,
            archive_result TEXT,
            target_errors_json TEXT,
            queue_order INTEGER,
            PRIMARY KEY (hook, main, final)
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS active_job (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            hook INTEGER NOT NULL,
            main INTEGER NOT NULL,
            final INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            workdir TEXT NOT NULL,
            stitched_path TEXT NOT NULL,
            stage TEXT NOT NULL,
            retry_at TEXT,
            retry_delay_minutes INTEGER NOT NULL DEFAULT 1,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    active_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(active_job)").fetchall()
    }
    if "retry_at" not in active_columns:
        connection.execute("ALTER TABLE active_job ADD COLUMN retry_at TEXT")
    if "retry_delay_minutes" not in active_columns:
        connection.execute(
            "ALTER TABLE active_job ADD COLUMN retry_delay_minutes INTEGER NOT NULL DEFAULT 1"
        )
    combination_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(combinations)").fetchall()
    }
    if "queue_order" not in combination_columns:
        connection.execute("ALTER TABLE combinations ADD COLUMN queue_order INTEGER")
    connection.executemany(
        "INSERT OR IGNORE INTO combinations(hook, main, final) VALUES (?, ?, ?)",
        product(range(1, 6), repeat=3),
    )
    unordered = connection.execute(
        """
        SELECT hook, main, final FROM combinations
        WHERE published = 0 AND queue_order IS NULL
        """
    ).fetchall()
    if unordered:
        randomized = [tuple(int(row[key]) for key in CATEGORIES) for row in unordered]
        random.SystemRandom().shuffle(randomized)
        first_position = int(
            connection.execute("SELECT COALESCE(MAX(queue_order), 0) FROM combinations").fetchone()[0]
        ) + 1
        connection.executemany(
            "UPDATE combinations SET queue_order = ? WHERE hook = ? AND main = ? AND final = ?",
            [(first_position + index, *combination) for index, combination in enumerate(randomized)],
        )
    connection.commit()
    return connection


def export_csv(connection: sqlite3.Connection, state_dir: Path) -> Path:
    destination = state_dir / "combinations.csv"
    temporary = destination.with_suffix(".csv.tmp")
    rows = connection.execute(
        "SELECT * FROM combinations ORDER BY hook, main, final"
    ).fetchall()
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "hook", "main", "final", "очередь", "опубликовано", "попыток",
                "последняя попытка", "дата публикации", "media_id",
                "permalink", "архивация", "ошибки целей",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["hook"], row["main"], row["final"], row["queue_order"] or "",
                    "да" if row["published"] else "нет", row["attempts"],
                    row["last_attempt_at"] or "", row["published_at"] or "",
                    row["media_id"] or "", row["permalink"] or "",
                    row["archive_result"] or "", row["target_errors_json"] or "",
                ]
            )
    os.replace(temporary, destination)
    return destination


def validate_sources(rendered_dir: Path) -> None:
    for category in CATEGORIES:
        directory = rendered_dir / category
        if not directory.is_dir():
            raise RuntimeError(f"Не найдена директория: {directory}")
        for number in range(1, 6):
            text = directory / f"{number}.txt"
            if not text.is_file():
                raise RuntimeError(f"Не найдено описание: {text}")
            find_video(rendered_dir, category, number)


def find_video(rendered_dir: Path, category: str, number: int) -> Path:
    directory = rendered_dir / category
    matches = [
        path for path in directory.iterdir()
        if path.is_file()
        and path.stem == str(number)
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Для {category}/{number} ожидалось одно видео, найдено: {len(matches)}"
        )
    return matches[0]


def combination_caption(rendered_dir: Path, combination: tuple[int, int, int]) -> str:
    parts = []
    for category, number in zip(CATEGORIES, combination):
        parts.append((rendered_dir / category / f"{number}.txt").read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


def concatenate_videos(
    rendered_dir: Path,
    combination: tuple[int, int, int],
    destination: Path,
    progress: Callable[[float, str], None] | None = None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg не найден")
    sources = [
        find_video(rendered_dir, category, number)
        for category, number in zip(CATEGORIES, combination)
    ]
    durations = [float(engine.ffprobe(source)["format"]["duration"]) for source in sources]
    total_duration = max(0.1, sum(durations))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    video_filters = []
    audio_filters = []
    concat_inputs = []
    for index in range(3):
        video_filters.append(
            f"[{index}:v:0]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=30,"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        audio_filters.append(
            f"[{index}:a:0]aresample=48000:async=1:first_pts=0,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    filters = ";".join(
        video_filters
        + audio_filters
        + ["".join(concat_inputs) + "concat=n=3:v=1:a=1[v][a]"]
    )
    command = [ffmpeg, "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend(
        [
            "-filter_complex", filters, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.1",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-r", "30", "-fps_mode", "cfr", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats",
            str(destination),
        ]
    )
    log_path = destination.parent / "concat.log"
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        last_percent = -1
        for line in process.stdout:
            key, _, value = line.strip().partition("=")
            if key not in {"out_time_us", "out_time_ms"}:
                continue
            try:
                percent = min(99, max(0, int(float(value) / 1_000_000 / total_duration * 100)))
            except (ValueError, ZeroDivisionError):
                continue
            if progress and percent != last_percent:
                last_percent = percent
                progress(percent, f"Склейка видео: {percent}%")
        code = process.wait()
    if code != 0 or not destination.is_file():
        raise RuntimeError(f"ffmpeg concat failed with code {code}; журнал: {log_path}")
    if progress:
        progress(100, "Склейка завершена")


def setting(connection: sqlite3.Connection, key: str) -> str:
    row = connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else ""


def set_setting(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    connection.commit()


def next_combination(connection: sqlite3.Connection) -> tuple[int, int, int] | None:
    row = connection.execute(
        "SELECT hook, main, final FROM combinations WHERE published = 0 "
        "ORDER BY queue_order, hook, main, final LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return int(row["hook"]), int(row["main"]), int(row["final"])


def interval_elapsed(connection: sqlite3.Connection, now: datetime | None = None) -> bool:
    value = setting(connection, "last_publish_attempt_at")
    if not value:
        return True
    previous = datetime.fromisoformat(value)
    return (now or utc_now()) - previous >= PUBLISH_INTERVAL


def active_job(connection: sqlite3.Connection) -> sqlite3.Row | None:
    return connection.execute("SELECT * FROM active_job WHERE id = 1").fetchone()


def update_active_stage(connection: sqlite3.Connection, stage: str) -> None:
    connection.execute(
        "UPDATE active_job SET stage = ?, updated_at = ? WHERE id = 1",
        (stage, iso_now()),
    )
    connection.commit()


def schedule_retry(connection: sqlite3.Connection, minutes: int | None = None) -> tuple[int, str]:
    checkpoint = active_job(connection)
    configured = int(checkpoint["retry_delay_minutes"] or 1) if checkpoint else 1
    delay = max(1, min(24 * 60, int(minutes or configured)))
    retry_at = (utc_now() + timedelta(minutes=delay)).replace(microsecond=0).isoformat()
    connection.execute(
        """
        UPDATE active_job
        SET stage = 'failed', retry_at = ?, retry_delay_minutes = ?, updated_at = ?
        WHERE id = 1
        """,
        (retry_at, delay, iso_now()),
    )
    connection.commit()
    return delay, retry_at


def retry_due(checkpoint: sqlite3.Row, now: datetime | None = None) -> bool:
    retry_at = str(checkpoint["retry_at"] or "")
    if not retry_at:
        return True
    return (now or utc_now()) >= datetime.fromisoformat(retry_at)


def has_ambiguous_publish(errors: dict[str, str]) -> bool:
    return any("AMBIGUOUS_MEDIA_PUBLISH" in str(error) for error in errors.values())


def clear_active_job(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM active_job WHERE id = 1")
    connection.commit()


def write_progress(
    state_dir: Path,
    *,
    percent: float,
    title: str,
    detail: str,
    status: str = "running",
    retry_minutes: int | None = None,
    next_run_at: str | None = None,
) -> None:
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "progress.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "percent": max(0.0, min(100.0, float(percent))),
                    "title": title,
                    "detail": redact(detail),
                    "status": status,
                    "updated_at": iso_now(),
                    "owner_pid": os.getpid(),
                    "retry_minutes": retry_minutes,
                    "next_run_at": next_run_at,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        pass


def start_progress_overlay(state_dir: Path) -> None:
    try:
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--state-dir", str(state_dir),
                "progress-window",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        pass


def start_statusbar(state_dir: Path) -> None:
    source = PROJECT_DIR / "factory_statusbar.m"
    binary = state_dir / "factory-statusbar"
    try:
        if not binary.is_file() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
            subprocess.run(
                [
                    "xcrun", "clang", "-fobjc-arc", "-framework", "Cocoa",
                    str(source), "-o", str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        subprocess.Popen(
            [str(binary), str(state_dir), sys.executable, str(Path(__file__).resolve())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def show_progress_window(state_dir: Path) -> int:
    import tkinter as tk
    from tkinter import simpledialog, ttk

    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "progress-window.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        root = tk.Tk()
        root.title("Factory Autoposter")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg="#1f2329")
        width, height = 380, 158
        screen_width = root.winfo_screenwidth()

        def position(window_width: int, window_height: int) -> None:
            root.geometry(f"{window_width}x{window_height}+{screen_width - window_width - 24}+24")

        position(width, height)

        title_var = tk.StringVar(value="Factory Autoposter")
        detail_var = tk.StringVar(value="Подготовка...")
        countdown_var = tk.StringVar(value="")
        content = tk.Frame(root, bg="#1f2329")
        content.pack(fill="both", expand=True)
        header = tk.Frame(content, bg="#1f2329")
        header.pack(fill="x", padx=18, pady=(12, 2))
        title_label = tk.Label(
            header, textvariable=title_var, bg="#1f2329", fg="#ffffff",
            font=("Helvetica", 14, "bold"), anchor="w",
        )
        title_label.pack(side="left", fill="x", expand=True)
        hide_button = tk.Button(
            header, text="Скрыть", bg="#1f2329", fg="#9fa7b3",
            activebackground="#1f2329", activeforeground="#ffffff",
            relief="flat", borderwidth=0, highlightthickness=0,
            font=("Helvetica", 10), cursor="hand2",
        )
        hide_button.pack(side="right")
        detail_label = tk.Label(
            content, textvariable=detail_var, bg="#1f2329", fg="#c7ccd4",
            font=("Helvetica", 11), anchor="w",
        )
        detail_label.pack(fill="x", padx=18, pady=(0, 10))
        meter = tk.Frame(content, bg="#1f2329")
        meter.pack(fill="x", padx=18, pady=(0, 12))
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Autoposter.Horizontal.TProgressbar",
            troughcolor="#343a43", background="#5ea1ff", borderwidth=0,
        )
        bar = ttk.Progressbar(
            meter, orient="horizontal", mode="determinate", maximum=100,
            style="Autoposter.Horizontal.TProgressbar",
        )
        bar.pack(fill="x", ipady=3)
        countdown_label = tk.Label(
            meter, textvariable=countdown_var, bg="#343a43", fg="#ffffff",
            font=("Helvetica", 13, "bold"), padx=10, pady=7,
        )
        retry_button = tk.Button(
            content,
            text="Повторить через...",
            bg="#343a43",
            fg="#ffffff",
            activebackground="#46505d",
            activeforeground="#ffffff",
            relief="flat",
            font=("Helvetica", 11),
            cursor="hand2",
        )
        open_button = tk.Button(
            root, text="Открыть", bg="#1f2329", fg="#ffffff",
            activebackground="#343a43", activeforeground="#ffffff",
            relief="flat", borderwidth=0, highlightthickness=0,
            font=("Helvetica", 11, "bold"), cursor="hand2",
        )

        def collapse() -> None:
            content.pack_forget()
            open_button.pack(fill="both", expand=True)
            position(112, 38)

        def expand() -> None:
            open_button.pack_forget()
            content.pack(fill="both", expand=True)
            position(width, height)

        hide_button.configure(command=collapse)
        open_button.configure(command=expand)

        def choose_retry_delay() -> None:
            connection = connect_database(state_dir)
            try:
                checkpoint = active_job(connection)
                current = int(checkpoint["retry_delay_minutes"] or 1) if checkpoint else 1
                minutes = simpledialog.askinteger(
                    "Повтор публикации",
                    "Через сколько минут повторить?",
                    initialvalue=current,
                    minvalue=1,
                    maxvalue=24 * 60,
                    parent=root,
                )
                if minutes is None or checkpoint is None:
                    return
                _, retry_at = schedule_retry(connection, minutes)
                local_retry = datetime.fromisoformat(retry_at).astimezone().strftime("%H:%M")
                detail = f"Следующая попытка через {minutes} мин., в {local_retry}"
                detail_var.set(detail)
                write_progress(
                    state_dir, percent=100, title=title_var.get(), detail=detail,
                    status="error", retry_minutes=minutes, next_run_at=retry_at,
                )
            finally:
                connection.close()

        retry_button.configure(command=choose_retry_delay)

        def refresh() -> None:
            try:
                payload = json.loads((state_dir / "progress.json").read_text(encoding="utf-8"))
                title_var.set(str(payload.get("title") or "Factory Autoposter"))
                detail_var.set(str(payload.get("detail") or ""))
                bar["value"] = float(payload.get("percent") or 0)
                status_value = str(payload.get("status") or "running")
                if status_value == "waiting":
                    root.destroy()
                    return
                owner_pid = int(payload.get("owner_pid") or 0)
                if status_value == "running" and owner_pid:
                    try:
                        os.kill(owner_pid, 0)
                    except OSError:
                        detail_var.set("Задача прервана. Возобновление на ближайшем запуске...")
                if status_value in {"waiting", "error"}:
                    bar.pack_forget()
                    if not countdown_label.winfo_ismapped():
                        countdown_label.pack(fill="x")
                    next_run_at = str(payload.get("next_run_at") or "")
                    if next_run_at:
                        seconds = max(
                            0,
                            int((datetime.fromisoformat(next_run_at) - utc_now()).total_seconds()),
                        )
                        hours, remainder = divmod(seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        countdown_var.set(
                            f"До следующей публикации: {hours:02d}:{minutes:02d}:{seconds:02d}"
                        )
                    else:
                        countdown_var.set("Ожидание следующей публикации")
                else:
                    countdown_label.pack_forget()
                    if not bar.winfo_ismapped():
                        bar.pack(fill="x", ipady=3)
                if status_value == "error":
                    if not retry_button.winfo_ismapped():
                        retry_button.pack(padx=18, pady=(0, 12), anchor="e")
                else:
                    retry_button.pack_forget()
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            root.after(300, refresh)

        refresh()
        root.mainloop()
    return 0


def write_failure_report(
    state_dir: Path,
    combination: tuple[int, int, int],
    target_errors: dict[str, str],
    logs: list[str],
) -> Path:
    reports = state_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    report = reports / f"failure_{combination[0]}-{combination[1]}-{combination[2]}_{stamp}.txt"
    lines = [
        f"Комбинация: {combination[0]}-{combination[1]}-{combination[2]}",
        f"Время UTC: {iso_now()}",
        "",
        "Ошибки целей:",
    ]
    lines.extend(f"- {target}: {redact(error)}" for target, error in target_errors.items())
    if logs:
        lines.extend(["", "Журнал:", *[redact(line) for line in logs[-100:]]])
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    subprocess.run(["open", str(report)], check=False)
    return report


@contextmanager
def process_lock(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "autoposter.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Другая копия автопостера уже работает") from exc
        yield


def request_publish_now(state_dir: Path) -> dict[str, Any]:
    with process_lock(state_dir):
        if not posting_enabled(state_dir):
            raise RuntimeError("Автопостинг выключен")
        connection = connect_database(state_dir)
        try:
            if active_job(connection):
                raise RuntimeError("Публикация уже выполняется или ожидает проверки")
            combination = next_combination(connection)
            if combination is None:
                raise RuntimeError("Все комбинации уже опубликованы")
            request_path = state_dir / PUBLISH_NOW_REQUEST
            temporary = request_path.with_suffix(".request.tmp")
            temporary.write_text(iso_now(), encoding="utf-8")
            os.replace(temporary, request_path)
            combo_id = "-".join(map(str, combination))
            write_progress(
                state_dir,
                percent=0,
                title=f"Следующий Trial Reel {combo_id}",
                detail="Запрошена немедленная публикация...",
                status="queued",
                next_run_at=iso_now(),
            )
            return {"requested": True, "combination": list(combination)}
        finally:
            connection.close()


def archived_publication(job_id: str) -> dict[str, Any] | None:
    try:
        from montage.social_stats import stats_directory

        for metadata_path in stats_directory().glob("reels/*/metadata.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("job_id") or "") != job_id:
                continue
            media = payload.get("media") or {}
            media_id = str(media.get("id") or "").strip()
            if media_id:
                return {
                    "media_id": media_id,
                    "permalink": str(media.get("permalink") or ""),
                    "archive_dir": str(metadata_path.parent),
                    "kind": "Instagram пробное",
                }
    except Exception:
        return None
    return None


def record_attempt(connection: sqlite3.Connection, combination: tuple[int, int, int]) -> None:
    attempt_at = iso_now()
    set_setting(connection, "last_publish_attempt_at", attempt_at)
    connection.execute(
        "UPDATE combinations SET attempts = attempts + 1, last_attempt_at = ? "
        "WHERE hook = ? AND main = ? AND final = ?",
        (attempt_at, *combination),
    )
    connection.commit()


def publish_combination(
    connection: sqlite3.Connection,
    rendered_dir: Path,
    state_dir: Path,
    combination: tuple[int, int, int],
    *,
    dry_run: bool = False,
    publish_fn: Callable[..., dict[str, Any]] = engine.publish_job,
) -> dict[str, Any]:
    combo_id = f"{combination[0]}-{combination[1]}-{combination[2]}"
    caption = combination_caption(rendered_dir, combination)
    if dry_run:
        job_id = f"factory_{combo_id}_{time.time_ns()}"
        workdir = state_dir / "work" / job_id
        stitched = workdir / f"trial_{combo_id}.mp4"
        concatenate_videos(rendered_dir, combination, stitched)
        probe = engine.ffprobe(stitched)
        stitched.unlink(missing_ok=True)
        shutil.rmtree(workdir, ignore_errors=True)
        return {"dry_run": True, "combination": combo_id, "probe": probe, "caption": caption}

    checkpoint = active_job(connection)
    checkpoint_matches = bool(
        checkpoint
        and tuple(int(checkpoint[key]) for key in CATEGORIES) == combination
    )
    if checkpoint_matches:
        job_id = str(checkpoint["job_id"])
        workdir = Path(str(checkpoint["workdir"]))
        stitched = Path(str(checkpoint["stitched_path"]))
        previous_stage = str(checkpoint["stage"])
        if previous_stage == "failed":
            record_attempt(connection, combination)
            connection.execute(
                "UPDATE active_job SET stage = ?, retry_at = NULL, updated_at = ? WHERE id = 1",
                ("publishing" if stitched.is_file() else "concat", iso_now()),
            )
            connection.commit()
    else:
        if checkpoint:
            clear_active_job(connection)
        job_id = f"factory_{combo_id}_{time.time_ns()}"
        workdir = state_dir / "work" / job_id
        stitched = workdir / f"trial_{combo_id}.mp4"
        now = iso_now()
        connection.execute(
            """
            INSERT OR REPLACE INTO active_job(
                id, hook, main, final, job_id, workdir, stitched_path,
                stage, started_at, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, 'concat', ?, ?)
            """,
            (*combination, job_id, str(workdir), str(stitched), now, now),
        )
        connection.commit()
        record_attempt(connection, combination)

    title = f"Trial Reel {combo_id}"
    write_progress(
        state_dir, percent=1, title=title,
        detail="Возобновление задачи..." if checkpoint_matches else "Подготовка задачи...",
    )
    logs: list[str] = []
    target = {
        "id": "instagram_trial",
        "kind": "trial",
        "mirrored": False,
        "label": "Instagram пробное",
        "text": caption,
        "caption": caption,
    }
    try:
        stitched_ready = False
        if stitched.is_file():
            try:
                stitched_ready = float(engine.ffprobe(stitched)["format"]["duration"]) > 0
            except Exception:
                stitched.unlink(missing_ok=True)
        if not stitched_ready:
            update_active_stage(connection, "concat")
            concatenate_videos(
                rendered_dir,
                combination,
                stitched,
                lambda percent, detail: write_progress(
                    state_dir, percent=2 + percent * 0.18, title=title, detail=detail
                ),
            )
        else:
            write_progress(
                state_dir, percent=20, title=title,
                detail="Готовая склейка восстановлена",
            )
        update_active_stage(connection, "publishing")
        recovered = archived_publication(job_id)
        if recovered:
            logs.append(f"Восстановлен архив ранее завершённой публикации: {recovered['media_id']}")
            result = {"results": {"instagram_trial": recovered}, "errors": {}}
        else:
            staging_dir = workdir / "staging"
            for stale_video in staging_dir.glob("*.mp4") if staging_dir.exists() else []:
                stale_video.unlink(missing_ok=True)

            def publication_progress(percent: float, stage: str, detail: str) -> None:
                logs.append(f"{percent:.1f}% {stage}: {detail}")
                write_progress(
                    state_dir,
                    percent=20 + percent * 0.8,
                    title=title,
                    detail=detail,
                )

            result = publish_fn(
                job_id=job_id,
                workdir=workdir,
                source_video=stitched,
                original_video=stitched,
                transcript_text=caption,
                targets=[target],
                progress_cb=publication_progress,
                target_progress_cb=lambda target_id, percent, detail, status: logs.append(
                    f"{target_id} {percent:.1f}% {status}: {detail}"
                ),
                log_cb=lambda message: logs.append(str(message)),
            )
        target_errors = {
            str(key): redact(str(value)) for key, value in (result.get("errors") or {}).items()
        }
        published = (result.get("results") or {}).get("instagram_trial") or {}
        media_id = str(published.get("media_id") or "").strip()
        if target_errors or not media_id:
            if not target_errors:
                target_errors = {"instagram_trial": "media_publish не вернул media_id"}
            report = write_failure_report(state_dir, combination, target_errors, logs)
            connection.execute(
                "UPDATE combinations SET target_errors_json = ? WHERE hook = ? AND main = ? AND final = ?",
                (json.dumps(target_errors, ensure_ascii=False), *combination),
            )
            connection.commit()
            if has_ambiguous_publish(target_errors):
                update_active_stage(connection, "ambiguous")
                write_progress(
                    state_dir, percent=100, title=title,
                    detail="Meta могла опубликовать Reel. Автоповтор остановлен до проверки.",
                    status="error",
                )
                return {"published": False, "errors": target_errors, "report": str(report)}
            retry_minutes, retry_at = schedule_retry(connection)
            write_progress(
                state_dir, percent=100, title=title,
                detail=f"Ошибка. Повтор через {retry_minutes} мин. Отчёт: {report.name}",
                status="error", retry_minutes=retry_minutes, next_run_at=retry_at,
            )
            return {"published": False, "errors": target_errors, "report": str(report)}

        archive_result = str(published.get("archive_dir") or "")
        if published.get("archive_error"):
            archive_result = "ERROR: " + redact(str(published["archive_error"]))
        connection.execute(
            """
            UPDATE combinations SET published = 1, published_at = ?, media_id = ?,
                permalink = ?, archive_result = ?, target_errors_json = NULL
            WHERE hook = ? AND main = ? AND final = ?
            """,
            (
                iso_now(), media_id, str(published.get("permalink") or ""),
                archive_result, *combination,
            ),
        )
        connection.commit()
        clear_active_job(connection)
        shutil.rmtree(workdir, ignore_errors=True)
        write_progress(
            state_dir, percent=100, title=title,
            detail=f"Опубликовано: {media_id}", status="done",
        )
        return {"published": True, **published}
    except Exception as exc:
        errors = {"instagram_trial": redact(str(exc))}
        report = write_failure_report(state_dir, combination, errors, logs)
        connection.execute(
            "UPDATE combinations SET target_errors_json = ? WHERE hook = ? AND main = ? AND final = ?",
            (json.dumps(errors, ensure_ascii=False), *combination),
        )
        connection.commit()
        if has_ambiguous_publish(errors):
            update_active_stage(connection, "ambiguous")
            write_progress(
                state_dir, percent=100, title=title,
                detail="Meta могла опубликовать Reel. Автоповтор остановлен до проверки.",
                status="error",
            )
            return {"published": False, "errors": errors, "report": str(report)}
        retry_minutes, retry_at = schedule_retry(connection)
        write_progress(
            state_dir, percent=100, title=title,
            detail=f"Ошибка. Повтор через {retry_minutes} мин. Отчёт: {report.name}",
            status="error", retry_minutes=retry_minutes, next_run_at=retry_at,
        )
        return {"published": False, "errors": errors, "report": str(report)}
    finally:
        export_csv(connection, state_dir)


def run_once(rendered_dir: Path, state_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    validate_sources(rendered_dir)
    with process_lock(state_dir):
        if not dry_run and not posting_enabled(state_dir):
            return {"paused": True, "message": "Автопостинг выключен"}
        connection = connect_database(state_dir)
        try:
            export_csv(connection, state_dir)
            checkpoint = active_job(connection)
            if checkpoint:
                combination = tuple(int(checkpoint[key]) for key in CATEGORIES)
                published_row = connection.execute(
                    "SELECT published FROM combinations WHERE hook = ? AND main = ? AND final = ?",
                    combination,
                ).fetchone()
                if published_row and int(published_row["published"]):
                    clear_active_job(connection)
                    checkpoint = None
                    combination = next_combination(connection)
            else:
                combination = next_combination(connection)
            if combination is None:
                return {"complete": True, "message": "Все комбинации опубликованы"}
            publish_now = not dry_run and (state_dir / PUBLISH_NOW_REQUEST).is_file()
            interrupted = bool(checkpoint and str(checkpoint["stage"]) in {"concat", "publishing"})
            failed_retry = bool(checkpoint and str(checkpoint["stage"]) == "failed")
            ambiguous_publish = bool(checkpoint and str(checkpoint["stage"]) == "ambiguous")
            if not dry_run and ambiguous_publish:
                return {
                    "waiting": True,
                    "message": "Автоповтор остановлен: требуется проверка неоднозначного media_publish",
                }
            if not dry_run and failed_retry and not retry_due(checkpoint):
                return {
                    "waiting": True,
                    "message": "Ожидание минутного повтора после ошибки",
                    "retry_at": checkpoint["retry_at"],
                }
            if (
                not dry_run
                and not publish_now
                and not interrupted
                and not failed_retry
                and not interval_elapsed(connection)
            ):
                return {"waiting": True, "message": "Пять часов после предыдущей попытки еще не прошли"}
            if publish_now:
                (state_dir / PUBLISH_NOW_REQUEST).unlink(missing_ok=True)
            return publish_combination(
                connection, rendered_dir, state_dir, combination, dry_run=dry_run
            )
        finally:
            connection.close()


def update_waiting_progress(state_dir: Path) -> None:
    if not posting_enabled(state_dir):
        write_progress(
            state_dir,
            percent=0,
            title="Автопостер выключен",
            detail="Новые публикации не запускаются",
            status="paused",
        )
        return
    connection = connect_database(state_dir)
    try:
        checkpoint = active_job(connection)
        if checkpoint and str(checkpoint["stage"]) == "ambiguous":
            combination = tuple(int(checkpoint[key]) for key in CATEGORIES)
            write_progress(
                state_dir,
                percent=100,
                title=f"Проверить Trial Reel {'-'.join(map(str, combination))}",
                detail="Meta могла опубликовать Reel. Автоповтор остановлен.",
                status="error",
            )
            return
        if checkpoint and str(checkpoint["stage"]) == "failed":
            combination = tuple(int(checkpoint[key]) for key in CATEGORIES)
            retry_at = str(checkpoint["retry_at"] or iso_now())
            retry_minutes = int(checkpoint["retry_delay_minutes"] or 1)
            write_progress(
                state_dir,
                percent=100,
                title=f"Повтор Trial Reel {'-'.join(map(str, combination))}",
                detail=f"Ошибка публикации. Повтор через {retry_minutes} мин.",
                status="error",
                retry_minutes=retry_minutes,
                next_run_at=retry_at,
            )
            return

        combination = next_combination(connection)
        if combination is None:
            write_progress(
                state_dir, percent=100, title="Factory Autoposter",
                detail="Все комбинации опубликованы", status="waiting",
            )
            return
        last_attempt = setting(connection, "last_publish_attempt_at")
        next_run = (
            datetime.fromisoformat(last_attempt) + PUBLISH_INTERVAL
            if last_attempt else utc_now()
        )
        write_progress(
            state_dir,
            percent=0,
            title=f"Следующий Trial Reel {'-'.join(map(str, combination))}",
            detail="Автоматическая публикация без подтверждения",
            status="waiting",
            next_run_at=next_run.replace(microsecond=0).isoformat(),
        )
    finally:
        connection.close()


def run_daemon(rendered_dir: Path, state_dir: Path) -> int:
    update_waiting_progress(state_dir)
    start_statusbar(state_dir)
    while True:
        (state_dir / CONTROL_CHANGED_REQUEST).unlink(missing_ok=True)
        try:
            result = run_once(rendered_dir, state_dir, dry_run=False)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result.get("published"):
                time.sleep(8)
            if not result.get("errors"):
                update_waiting_progress(state_dir)
            delay = 3600 if result.get("complete") else 60
        except Exception as exc:
            print(f"Ошибка фонового цикла: {redact(str(exc))}", file=sys.stderr, flush=True)
            write_progress(
                state_dir, percent=100, title="Factory Autoposter",
                detail="Фоновый цикл прерван. Повтор через минуту.", status="error",
                next_run_at=(utc_now() + timedelta(minutes=1)).replace(microsecond=0).isoformat(),
            )
            delay = 60
        for _ in range(delay):
            time.sleep(1)
            if (
                (state_dir / PUBLISH_NOW_REQUEST).is_file()
                or (state_dir / CONTROL_CHANGED_REQUEST).is_file()
            ):
                break


def install_agent(rendered_dir: Path, state_dir: Path) -> Path:
    agents_dir = Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    plist_path = agents_dir / f"{AGENT_LABEL}.plist"
    payload = {
        "Label": AGENT_LABEL,
        "ProgramArguments": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--rendered-dir", str(rendered_dir),
            "--state-dir", str(state_dir),
            "daemon",
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "ProcessType": "Background",
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        },
        "StandardOutPath": str(state_dir / "agent.stdout.log"),
        "StandardErrorPath": str(state_dir / "agent.stderr.log"),
    }
    temporary = plist_path.with_suffix(".plist.tmp")
    temporary.write_bytes(plistlib.dumps(payload))
    os.replace(temporary, plist_path)
    return plist_path


def status(connection: sqlite3.Connection, state_dir: Path) -> dict[str, Any]:
    row = connection.execute(
        "SELECT COUNT(*) AS total, SUM(published) AS published FROM combinations"
    ).fetchone()
    checkpoint = active_job(connection)
    return {
        "total": int(row["total"]),
        "published": int(row["published"] or 0),
        "remaining": int(row["total"] - (row["published"] or 0)),
        "next": next_combination(connection),
        "last_publish_attempt_at": setting(connection, "last_publish_attempt_at") or None,
        "posting_enabled": posting_enabled(state_dir),
        "active_job": (
            {
                "combination": [checkpoint["hook"], checkpoint["main"], checkpoint["final"]],
                "job_id": checkpoint["job_id"],
                "stage": checkpoint["stage"],
                "retry_at": checkpoint["retry_at"],
                "retry_delay_minutes": checkpoint["retry_delay_minutes"],
                "updated_at": checkpoint["updated_at"],
            }
            if checkpoint else None
        ),
    }


def set_retry_delay(state_dir: Path, minutes: int) -> dict[str, Any]:
    connection = connect_database(state_dir)
    try:
        checkpoint = active_job(connection)
        if not checkpoint or str(checkpoint["stage"]) not in {"failed", "ambiguous"}:
            raise RuntimeError("Нет публикации, ожидающей повтора")
        delay, retry_at = schedule_retry(connection, minutes)
        combination = tuple(int(checkpoint[key]) for key in CATEGORIES)
        title = f"Повтор Trial Reel {'-'.join(map(str, combination))}"
        local_retry = datetime.fromisoformat(retry_at).astimezone().strftime("%H:%M")
        write_progress(
            state_dir,
            percent=100,
            title=title,
            detail=f"Следующая попытка через {delay} мин., в {local_retry}",
            status="error",
            retry_minutes=delay,
            next_run_at=retry_at,
        )
        return {"retry_delay_minutes": delay, "retry_at": retry_at}
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factory Instagram Trial Reel autoposter")
    parser.add_argument("--rendered-dir", type=Path, default=DEFAULT_RENDERED_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Создать таблицу 125 комбинаций")
    subparsers.add_parser("status", help="Показать состояние очереди")
    subparsers.add_parser("dry-run", help="Склеить следующую комбинацию без публикации")
    subparsers.add_parser("run-once", help="Выполнить один тик фонового постера")
    subparsers.add_parser("daemon", help="Постоянный фоновый цикл с проверкой раз в минуту")
    subparsers.add_parser("progress-window", help=argparse.SUPPRESS)
    retry_parser = subparsers.add_parser("set-retry", help=argparse.SUPPRESS)
    retry_parser.add_argument("--minutes", type=int, required=True)
    subparsers.add_parser("publish-now", help=argparse.SUPPRESS)
    posting_parser = subparsers.add_parser("set-posting", help=argparse.SUPPRESS)
    posting_parser.add_argument("--enabled", choices=("0", "1"), required=True)
    subparsers.add_parser("install-agent", help="Создать macOS LaunchAgent (без запуска)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered_dir = args.rendered_dir.expanduser().resolve()
    state_dir = args.state_dir.expanduser().resolve()
    if args.command == "progress-window":
        return show_progress_window(state_dir)
    if args.command == "set-retry":
        result = set_retry_delay(state_dir, args.minutes)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "publish-now":
        result = request_publish_now(state_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "set-posting":
        result = set_posting_enabled(state_dir, args.enabled == "1")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "daemon":
        validate_sources(rendered_dir)
        return run_daemon(rendered_dir, state_dir)
    if args.command == "install-agent":
        validate_sources(rendered_dir)
        path = install_agent(rendered_dir, state_dir)
        print(f"LaunchAgent создан, но не запущен: {path}")
        print(f"Запуск: launchctl bootstrap gui/$(id -u) {path}")
        return 0
    connection = connect_database(state_dir)
    try:
        csv_path = export_csv(connection, state_dir)
        if args.command == "init":
            validate_sources(rendered_dir)
            print(json.dumps({**status(connection, state_dir), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "status":
            print(json.dumps(status(connection, state_dir), ensure_ascii=False, indent=2))
            return 0
    finally:
        connection.close()
    result = run_once(rendered_dir, state_dir, dry_run=args.command == "dry-run")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(f"Ошибка: {redact(str(exc))}", file=sys.stderr)
        raise SystemExit(1)
