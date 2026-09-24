from __future__ import annotations

import hmac
import importlib.util
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from PIL import Image


GITHUB_LIMIT_BYTES = 100 * 1024 * 1024
GITHUB_TARGET_BYTES = 88 * 1024 * 1024


class InstagramPublisher:
    def __init__(
        self,
        instaposter_dir: Path,
        ffmpeg: str,
        log_path: Path,
        on_log: Callable[[str], None],
    ) -> None:
        self.instaposter_dir = instaposter_dir.resolve()
        self.ffmpeg = ffmpeg
        self.log_path = log_path
        self.on_log = on_log
        self.main = self._load_module("autovideo_instaposter_main", self.instaposter_dir / "publish_reel_github.py")
        self.trial = self._load_module("autovideo_instaposter_trial", self.instaposter_dir / "publish_trial_reel_github.py")
        self._tokens = [str(getattr(self.main, "ACCESS_TOKEN", "")), str(getattr(self.trial, "ACCESS_TOKEN", ""))]
        if not all(token.strip() for token in self._tokens):
            raise RuntimeError("Instagram access token is not configured in instaposter")
        if not hmac.compare_digest(self._tokens[0], self._tokens[1]):
            raise RuntimeError("Instagram access tokens in instaposter scripts do not match")
        for name in (
            "IG_USER_ID",
            "GITHUB_REPO_DIR",
            "GITHUB_OWNER",
            "GITHUB_REPO",
            "GITHUB_BRANCH",
            "GITHUB_REELS_DIR",
            "GRAPH_HOST",
            "RAW_HOST",
            "API_VERSION",
        ):
            if getattr(self.main, name) != getattr(self.trial, name):
                raise RuntimeError(f"instaposter module configuration mismatch: {name}")
        self.main.die = self._safe_die
        self.trial.die = self._safe_die

    @staticmethod
    def _load_module(name: str, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"instaposter script not found: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load instaposter script: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def redact(self, text: str) -> str:
        clean = str(text or "")
        for token in self._tokens:
            if token:
                clean = clean.replace(token, "<REDACTED>")
        return re.sub(r"(?i)(access_token|authorization|token)=([^&\s]+)", r"\1=<REDACTED>", clean)

    def _safe_die(self, message: str) -> None:
        raise RuntimeError(self.redact(message))

    def _ffprobe_path(self) -> str:
        ffmpeg_path = Path(self.ffmpeg)
        if ffmpeg_path.is_file():
            sibling = ffmpeg_path.with_name("ffprobe.exe")
            if sibling.exists():
                return str(sibling)
        return shutil.which("ffprobe") or "ffprobe"

    def _duration(self, source: Path) -> float:
        result = subprocess.run(
            [
                self._ffprobe_path(),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[-500:]}")
        duration = float(result.stdout.strip())
        if duration <= 0:
            raise RuntimeError("cannot detect publication video duration")
        return duration

    def _transcode_for_github(
        self,
        source: Path,
        output: Path,
        progress: Callable[[float, str], None],
    ) -> None:
        duration = self._duration(source)
        audio_bitrate = 160_000
        target_video_bitrate = max(500_000, int(GITHUB_TARGET_BYTES * 8 * 0.96 / duration) - audio_bitrate)
        output.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            output.unlink(missing_ok=True)
            bitrate = max(500_000, int(target_video_bitrate * (0.88 ** attempt)))
            maxrate = int(bitrate * 1.08)
            command = [
                self.ffmpeg,
                "-y",
                "-fflags",
                "+genpts",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                "setpts=PTS-STARTPTS,fps=30",
                "-af",
                "aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-b:v",
                str(bitrate),
                "-maxrate",
                str(maxrate),
                "-bufsize",
                str(maxrate * 2),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-fps_mode",
                "cfr",
                "-shortest",
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                "-nostats",
                str(output),
            ]
            last_percent = -1
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8", errors="replace") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert process.stdout is not None
                for line in process.stdout:
                    key, _, value = line.strip().partition("=")
                    if key not in {"out_time_us", "out_time_ms"}:
                        continue
                    try:
                        elapsed = float(value) / 1_000_000.0
                    except ValueError:
                        continue
                    percent = min(98, max(1, int(elapsed / duration * 100)))
                    if percent != last_percent:
                        last_percent = percent
                        progress(percent, f"Подготовка файла: {percent}%")
                code = process.wait()
            if code != 0 or not output.exists():
                raise RuntimeError(f"publication ffmpeg failed with code {code}")
            if output.stat().st_size <= GITHUB_LIMIT_BYTES:
                progress(100, "Файл подготовлен для GitHub")
                return
            self.on_log(f"Файл все еще больше лимита GitHub, повторное сжатие (попытка {attempt + 2})")
        raise RuntimeError("cannot fit publication video under GitHub 100 MB limit")

    def _normalize_for_publication(
        self,
        source: Path,
        output: Path,
        progress: Callable[[float, str], None],
    ) -> None:
        duration = self._duration(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        command = [
            self.ffmpeg, "-y", "-fflags", "+genpts", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", "setpts=PTS-STARTPTS,fps=30",
            "-af", "aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.1",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-fps_mode", "cfr", "-shortest", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(output),
        ]
        last_percent = -1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
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
                    percent = min(99, max(0, int(float(value) / 1_000_000 / duration * 100)))
                except (ValueError, ZeroDivisionError):
                    continue
                if percent != last_percent:
                    last_percent = percent
                    progress(percent, f"Синхронизация видео и звука: {percent}%")
            code = process.wait()
        if code != 0 or not output.exists():
            raise RuntimeError("publication A/V normalization failed")
        progress(100, "Видео и звук синхронизированы")

    def prepare_variant(
        self,
        source: Path,
        staging_dir: Path,
        job_id: str,
        variant: str,
        progress: Callable[[float, str], None],
    ) -> Path:
        if not source.exists():
            raise RuntimeError(f"result video not found: {source.name}")
        staging_dir.mkdir(parents=True, exist_ok=True)
        generation = time.time_ns()
        staged = staging_dir / f"reel_{job_id}_{variant}_{generation}.mp4"
        staged.unlink(missing_ok=True)
        self.on_log(f"{source.name}: создаю H.264 CFR версию с синхронизированным звуком")
        self._normalize_for_publication(source, staged, progress)
        if staged.stat().st_size > GITHUB_LIMIT_BYTES:
            oversized = staged.with_name(f".{staged.stem}-oversized.mp4")
            os.replace(staged, oversized)
            try:
                self.on_log("Нормализованный файл больше 100 MB, дополнительно сжимаю")
                self._transcode_for_github(oversized, staged, progress)
            finally:
                oversized.unlink(missing_ok=True)
        self.main.validate_local_video(staged)
        return staged

    def resolve_hosts(self) -> tuple[str, str]:
        return (
            self.main.resolve_host_via_doh(self.main.GRAPH_HOST),
            self.main.resolve_host_via_doh(self.main.RAW_HOST),
        )

    def upload_to_github(self, staged: Path) -> tuple[Path, str]:
        return self.main.publish_video_to_github(staged)

    def upload_cover_to_github(self, cover: Path) -> tuple[Path, str]:
        prepared = cover.with_name(f".{cover.stem}-instagram.jpg")
        with Image.open(cover) as image:
            image.convert("RGB").save(prepared, "JPEG", quality=94, optimize=True)
        return self.main.publish_image_to_github(prepared)

    def verify_public_video(self, raw_url: str, raw_ip: str, progress: Callable[[float, str], None]) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 7):
            progress(min(95, attempt * 16), f"Проверка HTTPS-ссылки, попытка {attempt}/6")
            try:
                self.main.check_public_video(raw_url, raw_ip)
                progress(100, "HTTPS-ссылка доступна")
                return
            except Exception as exc:
                last_error = exc
                if attempt < 6:
                    time.sleep(3)
        raise RuntimeError(f"GitHub Raw URL is not ready: {self.redact(str(last_error))}")

    def create_container(self, kind: str, graph_ip: str, video_url: str, caption: str, cover_url: str = "") -> str:
        if kind == "trial":
            return self.trial.create_reel_container(graph_ip, video_url, caption, "MANUAL", cover_url)
        return self.main.create_reel_container(graph_ip, video_url, caption, True, cover_url)

    def wait_container(
        self,
        kind: str,
        graph_ip: str,
        container_id: str,
        progress: Callable[[float, str], None],
    ) -> None:
        module = self.trial if kind == "trial" else self.main
        started = time.time()
        timeout = float(module.POLL_TIMEOUT_SECONDS)
        attempt = 0
        while time.time() - started < timeout:
            attempt += 1
            data = module.meta_request(
                graph_ip,
                "GET",
                container_id,
                {"fields": "status_code,status", "access_token": module.ACCESS_TOKEN},
            )
            code = str(data.get("status_code") or "IN_PROGRESS")
            status_text = str(data.get("status") or "")
            elapsed = time.time() - started
            percent = min(95, max(2, int(elapsed / timeout * 100)))
            progress(percent, f"Instagram обрабатывает видео: {code} ({attempt}) {status_text}".strip())
            if code in {"FINISHED", "PUBLISHED"}:
                progress(100, "Instagram container готов")
                return
            if code in {"ERROR", "EXPIRED"}:
                raise RuntimeError(f"Instagram container failed: {code} {status_text}".strip())
            time.sleep(float(module.POLL_INTERVAL_SECONDS))
        raise TimeoutError("Instagram container processing timed out")

    def publish_container(self, kind: str, graph_ip: str, container_id: str) -> str:
        module = self.trial if kind == "trial" else self.main
        return module.publish_container(graph_ip, container_id)

    def find_recent_publication(
        self,
        kind: str,
        graph_ip: str,
        caption: str,
        published_after: datetime,
        timeout: float = 120,
    ) -> dict[str, Any] | None:
        module = self.trial if kind == "trial" else self.main
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                payload = module.meta_request(
                    graph_ip,
                    "GET",
                    f"{module.IG_USER_ID}/media",
                    {
                        "fields": "id,caption,media_type,media_product_type,permalink,timestamp",
                        "limit": "25",
                        "access_token": module.ACCESS_TOKEN,
                    },
                )
                for media in payload.get("data") or []:
                    if str(media.get("caption") or "").strip() != caption.strip():
                        continue
                    timestamp = str(media.get("timestamp") or "")
                    try:
                        published_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)
                    if published_at >= published_after:
                        return dict(media)
            except Exception as exc:
                self.on_log(f"Проверка результата media_publish не удалась: {self.redact(str(exc))}")
            time.sleep(10)
        return None

    def media_info(self, kind: str, graph_ip: str, media_id: str) -> dict[str, Any]:
        module = self.trial if kind == "trial" else self.main
        return module.get_media_info(graph_ip, media_id)
