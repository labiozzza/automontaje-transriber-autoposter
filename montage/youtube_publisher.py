from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from PIL import Image, ImageOps


class YouTubePublisher:
    def __init__(self, instaposter_dir: Path, python: Path, log_path: Path) -> None:
        self.script = (instaposter_dir / "publish_youtube_short.py").resolve()
        self.python = python.resolve()
        self.log_path = log_path
        if not self.script.exists():
            raise RuntimeError(f"YouTube publisher script not found: {self.script}")
        if not self.python.exists():
            raise RuntimeError(f"instaposter Python not found: {self.python}")

    def _run(
        self,
        payload: dict[str, Any],
        progress: Callable[[float, str], None] | None = None,
        upload_ready: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        command = [str(self.python), "-X", "utf8", "-u", str(Path(__file__).resolve()), "--bridge", str(self.script)]
        timeout = 180 if payload.get("action") == "auth_check" else 3600
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        code = -1
        with self.log_path.open("a", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                process.stdin.flush()
                output_queue: queue.Queue[str | None] = queue.Queue()

                def read_output() -> None:
                    try:
                        for line in process.stdout:
                            output_queue.put(line)
                    finally:
                        output_queue.put(None)

                threading.Thread(target=read_output, daemon=True).start()
                deadline = time.monotonic() + timeout
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"YouTube publisher timed out after {timeout} seconds")
                    try:
                        raw = output_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    if raw is None:
                        break
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    event_type = event.get("type")
                    if event_type == "checkpoint":
                        if upload_ready is None:
                            raise RuntimeError("YouTube upload checkpoint callback is missing")
                        upload_ready()
                        process.stdin.write('{"action":"continue"}\n')
                        process.stdin.flush()
                        process.stdin.close()
                    elif event_type == "progress" and progress is not None:
                        percent = max(0, min(100, float(event.get("percent") or 0)))
                        progress(percent, f"YouTube upload: {int(percent)}%")
                    elif event_type == "result":
                        result = event
                    elif event_type == "error":
                        error = event
                code = process.wait(timeout=10)
            finally:
                if not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if error:
            raise RuntimeError(str(error.get("message") or error.get("kind") or "YouTube upload failed"))
        if code != 0 or result is None:
            raise RuntimeError(f"YouTube publisher failed with code {code}")
        return result

    def check_auth(self) -> None:
        result = self._run({"action": "auth_check"})
        if result.get("status") != "auth_ok":
            raise RuntimeError("YouTube authorization is not ready")

    def _duration(self, video_path: Path) -> float:
        ffprobe = shutil.which("ffprobe") or "ffprobe"
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError("Не удалось определить длительность YouTube-видео")
        return float(result.stdout.strip())

    def _frame_rate(self, video_path: Path) -> str:
        ffprobe = shutil.which("ffprobe") or "ffprobe"
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        try:
            source = Fraction(result.stdout.strip())
            value = float(source)
        except (ValueError, ZeroDivisionError):
            return "30000/1001"
        standards = [Fraction(24000, 1001), Fraction(24, 1), Fraction(25, 1), Fraction(30000, 1001), Fraction(30, 1), Fraction(50, 1), Fraction(60000, 1001), Fraction(60, 1)]
        nearest = min(standards, key=lambda rate: abs(float(rate) - value))
        return str(nearest) if abs(float(nearest) - value) <= 1.0 else str(source.limit_denominator(1001))

    def prepare_video(self, source: Path, output: Path, progress: Callable[[float, str], None]) -> Path:
        duration = max(0.001, self._duration(source))
        fps = self._frame_rate(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        command = [
            ffmpeg, "-y", "-fflags", "+genpts", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", f"setpts=PTS-STARTPTS,fps={fps}",
            "-af", "aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.1",
            "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-fps_mode", "cfr", "-avoid_negative_ts", "make_zero", "-shortest",
            "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output),
        ]
        with self.log_path.open("a", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=log, text=True,
                encoding="utf-8", errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                key, _, value = line.strip().partition("=")
                if key not in {"out_time_us", "out_time_ms"}:
                    continue
                try:
                    percent = min(99, max(0, float(value) / 1_000_000 / duration * 100))
                except ValueError:
                    continue
                progress(percent, f"Подготовка YouTube CFR: {percent:.0f}%")
            code = process.wait()
        if code != 0 or not output.exists():
            raise RuntimeError("Не удалось подготовить синхронизированную YouTube-версию")
        progress(100, f"YouTube CFR {fps}: готово")
        return output

    def prepare_thumbnail(self, source: Path, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            thumbnail = ImageOps.fit(image.convert("RGB"), (1280, 720), method=Image.Resampling.LANCZOS)
            quality = 92
            while True:
                thumbnail.save(output, "JPEG", quality=quality, optimize=True)
                if output.stat().st_size <= 2 * 1024 * 1024 or quality <= 65:
                    break
                quality -= 7
        return output

    def upload(
        self,
        video_path: Path,
        title: str,
        progress: Callable[[float, str], None],
        upload_ready: Callable[[], None],
        thumbnail_path: Path | None = None,
    ) -> dict[str, str]:
        result = self._run(
            {
                "action": "upload",
                "video_path": str(video_path.resolve()),
                "title": title,
                "description": "",
                "privacy": "public",
                "category_id": "22",
                "tags": [],
                "made_for_kids": False,
                "thumbnail_path": str(thumbnail_path.resolve()) if thumbnail_path else "",
            },
            progress,
            upload_ready,
        )
        video_id = str(result.get("video_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise RuntimeError("YouTube returned an invalid video ID")
        return {
            "video_id": video_id,
            "watch_url": f"https://www.youtube.com/watch?v={video_id}",
            "shorts_url": f"https://www.youtube.com/shorts/{video_id}",
        }


class _AuthRequired(Exception):
    pass


def _emit(stream: Any, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()


def _load_source_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("autovideo_youtube_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("YouTube source module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_save_token(path: str, content: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".youtube-token-", suffix=".tmp", dir=directory, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_credentials(module: ModuleType) -> Any:
    if not os.path.isfile(module.TOKEN_FILE):
        raise _AuthRequired()
    try:
        credentials = module.Credentials.from_authorized_user_file(module.TOKEN_FILE, module.SCOPES)
    except Exception:
        raise _AuthRequired() from None
    if credentials.valid:
        return credentials
    if not credentials.refresh_token:
        raise _AuthRequired()
    try:
        credentials.refresh(module.Request())
    except Exception:
        raise _AuthRequired() from None
    if not credentials.valid:
        raise _AuthRequired()
    _atomic_save_token(module.TOKEN_FILE, credentials.to_json())
    return credentials


def _bridge_upload(module: ModuleType, payload: dict[str, Any], protocol: Any, commands: Any) -> None:
    video_path = Path(str(payload.get("video_path") or "")).resolve()
    title = str(payload.get("title") or "").strip()
    description = str(payload.get("description") or "")
    privacy = str(payload.get("privacy") or "public")
    if not video_path.is_file():
        raise ValueError("video_not_found")
    if not title or len(title) > 119:
        raise ValueError("invalid_title")
    if privacy not in {"public", "private", "unlisted"}:
        raise ValueError("invalid_privacy")
    if len(description.encode("utf-8")) > 5000:
        raise ValueError("invalid_description")
    credentials = _load_credentials(module)
    youtube = module.build("youtube", "v3", credentials=credentials, cache_discovery=False)
    media = module.MediaFileUpload(str(video_path), chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": str(payload.get("category_id") or "22"),
                "tags": [str(value).strip() for value in payload.get("tags", []) if str(value).strip()],
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": bool(payload.get("made_for_kids", False)),
            },
        },
        media_body=media,
        notifySubscribers=False,
    )
    _emit(protocol, {"type": "checkpoint"})
    acknowledgement = json.loads(commands.readline())
    if acknowledgement != {"action": "continue"}:
        raise RuntimeError("upload_checkpoint_not_acknowledged")
    _emit(protocol, {"type": "progress", "percent": 0})
    response = None
    last_percent = -1
    while response is None:
        status, response = request.next_chunk(num_retries=3)
        if status is None:
            continue
        percent = max(0, min(99, int(status.progress() * 100)))
        if percent != last_percent:
            last_percent = percent
            _emit(protocol, {"type": "progress", "percent": percent})
    video_id = str((response or {}).get("id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise RuntimeError("invalid_video_id")
    thumbnail_path = Path(str(payload.get("thumbnail_path") or ""))
    if thumbnail_path.is_file():
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=module.MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg"),
        ).execute()
    _emit(protocol, {"type": "progress", "percent": 100})
    _emit(protocol, {"type": "result", "video_id": video_id})


def _bridge_main(script: Path) -> int:
    protocol = sys.stdout
    sink = open(os.devnull, "w", encoding="utf-8")
    try:
        payload = json.loads(sys.stdin.readline())
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            module = _load_source_module(script)
            if payload.get("action") == "auth_check":
                credentials = _load_credentials(module)
                del credentials
                _emit(protocol, {"type": "result", "status": "auth_ok"})
            elif payload.get("action") == "upload":
                _bridge_upload(module, payload, protocol, sys.stdin)
            else:
                raise ValueError("invalid_action")
        return 0
    except _AuthRequired:
        _emit(protocol, {"type": "error", "kind": "auth_required", "message": "YouTube authorization must be completed in the instaposter directory"})
        return 10
    except ValueError as exc:
        kind = str(exc)
        if kind not in {"video_not_found", "invalid_title", "invalid_privacy", "invalid_description", "invalid_action"}:
            kind = "invalid_request"
        _emit(protocol, {"type": "error", "kind": kind, "message": "YouTube upload request is invalid"})
        return 20
    except Exception:
        _emit(protocol, {"type": "error", "kind": "upload_failed", "message": "YouTube upload failed"})
        return 30
    finally:
        sink.close()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--bridge":
        raise SystemExit(2)
    raise SystemExit(_bridge_main(Path(sys.argv[2]).resolve()))
