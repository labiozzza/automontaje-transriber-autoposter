from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .secrets_env import load_secrets_env, get_software_defaults

MONAGE_DIR = Path(__file__).resolve().parent
INSTAPOSTER_DIR = MONAGE_DIR / "instaposter"
MODELS_DIR = MONAGE_DIR / "models"
FONTS_DIR = MONAGE_DIR / "fonts"
ANIMATIONS_DIR = MONAGE_DIR / "animations"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"
DEFAULT_FONT = FONTS_DIR / "Comfortaa.ttf"
SYSTEM_FONT = Path("/System/Library/Fonts/Helvetica.ttc")
MEDIA_REPO_DIR = Path(os.environ.get("GITHUB_REPO_DIR", str(MONAGE_DIR.parent / "instagram-media")))

PYTHON = sys.executable

_internet_cache: dict[str, Any] = {"t": 0.0, "ok": False}


def log(type_: str, message: str) -> None:
    print(f"[montage] {type_}: {message}", file=sys.stderr)


def _url_open(url: str, timeout: float = 6.0) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def internet_available(force: bool = False) -> bool:
    global _internet_cache
    now = time.time()
    if not force and now - _internet_cache["t"] < 30:
        return _internet_cache["ok"]
    ok = _url_open("https://opencode.ai/") or _url_open("https://www.google.com/")
    _internet_cache = {"t": now, "ok": ok}
    return ok


def opencode_bin() -> str:
    return os.environ.get("OPENCODE_BIN", "") or shutil.which("opencode") or "opencode"


def face_model_ready() -> bool:
    return FACE_MODEL.exists()


def opencode_ready() -> bool:
    return shutil.which("opencode") is not None or Path(opencode_bin()).exists()


def media_repo_ready() -> bool:
    return (MEDIA_REPO_DIR / ".git").is_dir()


def animation_ids() -> list[dict[str, Any]]:
    items = []
    if ANIMATIONS_DIR.is_dir():
        for manifest in sorted(ANIMATIONS_DIR.glob("*/manifest.json")):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append({"id": manifest.parent.name, "name": data.get("name", manifest.parent.name)})
    return items


def font_options() -> list[dict[str, str]]:
    if not DEFAULT_FONT.exists():
        return []
    items = [{"id": DEFAULT_FONT.name, "name": "Comfortaa (встроен)", "path": str(DEFAULT_FONT)}]
    if SYSTEM_FONT.exists():
        items.append({"id": "system", "name": "Системный (Helvetica)", "path": str(SYSTEM_FONT)})
    return items


def montage_status() -> dict[str, Any]:
    load_secrets_env()
    sw = get_software_defaults()
    online = internet_available()
    return {
        "online": online,
        "opencode": opencode_ready(),
        "bigpickle_enabled": online and opencode_ready(),
        "face_model": face_model_ready(),
        "font": DEFAULT_FONT.exists(),
        "fonts": font_options(),
        "media_repo": media_repo_ready(),
        "animations": animation_ids(),
        "publishers": {
            "instagram_reels": sw["instagram_ready"],
            "threads": sw["threads_ready"],
            "youtube": sw["youtube_ready"],
            "telegram": sw["telegram_ready"],
        },
    }


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    line_cb: Callable[[str], None] | None = None,
    timeout: int = 3600,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if line_cb:
                try:
                    line_cb(line)
                except Exception:
                    pass
        return proc.wait()


def build_words_srt(segments: list[dict[str, Any]]) -> str:
    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        if ms == 1000:
            s += 1
            ms = 0
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        start = float(seg.get("start") or 0)
        end = float(seg.get("end") or start)
        if end <= start:
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        lines.extend([str(i), f"{fmt(start)} --> {fmt(end)}", text, ""])
    return "\n".join(lines).strip() + "\n"


def format_timecode(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        ms -= 1000
        s += 1
    return f"{m:02d}:{s:02d}.{ms:03d}"


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=width,height,r_frame_rate,codec_type", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[-400:]}")
    return json.loads(result.stdout)


def ensure_tracking_has_anchors(tracking_path: Path) -> bool:
    """apply_face_zoom_v5 requires >=1 face-anchor. If the video had no detected faces,
    synthesize center-anchored targets so zoom/render still works (camera stays centered)."""
    try:
        data = json.loads(tracking_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    targets = data.get("camera_targets") or []
    if any(item.get("face_found") for item in targets):
        return True
    source = data.get("source", {}) or {}
    fps = float(source.get("fps") or 30)
    duration = float(source.get("duration") or 0) or (len(targets) * 3.0 / 10.0)
    step = max(1, int(round(fps / 10.0)))
    n = max(1, int(round(duration * fps / step)))
    synthetic = []
    for i in range(n):
        t = i * step / fps
        synthetic.append({
            "frame": i * step,
            "time": round(t, 6),
            "face_found": True,
            "confidence": 0.0,
            "left_eye": {"x": 0.5, "y": 0.36},
            "right_eye": {"x": 0.55, "y": 0.36},
            "eye_center": {"x": 0.525, "y": 0.36},
            "target": {"x": 0.525, "y": 0.42, "zoom": 1.0},
        })
    data["camera_targets"] = synthetic
    data["_synthetic_anchors"] = True
    tracking_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return True


def render_montage(
    *,
    job_id: str,
    workdir: Path,
    source_video: Path,
    srt_text: str,
    options: dict[str, Any],
    progress_cb: Callable[[float, str, str], None],
    log_cb: Callable[[str], None],
) -> dict[str, Any]:
    load_secrets_env()
    workdir.mkdir(parents=True, exist_ok=True)
    transcript = Path(workdir) / f"{job_id}_words.srt"
    transcript.write_text(srt_text, encoding="utf-8")

    # ---- Stage: normalise source rotation (iPhone .mov autorotate) ----
    # Apple stores orientation in the Display Matrix (side_data / displaymatrix),
    # so probe BOTH tags.rotate AND the side_data rotation before deciding.
    source_video = Path(source_video)
    _rover: dict[str, Any] = {}
    try:
        _probe = json.loads(subprocess.check_output(
            [shutil.which("ffprobe") or "ffprobe", "-v", "error",
             "-show_streams", "-of", "json", str(source_video)],
            text=True, encoding="utf-8", errors="replace", timeout=90,
        ))
        for _st in _probe.get("streams") or []:
            if _st.get("codec_type") != "video":
                continue
            _rot: int | None = None
            if isinstance(_st.get("tags"), dict):
                try:
                    rotate_tag = _st["tags"].get("rotate")
                    _rot = int(rotate_tag) if rotate_tag not in (None, "") else None
                except (TypeError, ValueError):
                    _rot = None
            if _rot is None:
                for _sd in _st.get("side_data_list") or []:
                    if _sd.get("side_data_type") in ("Display Matrix", "displaymatrix"):
                        _rot = _sd.get("rotation")
                        try:
                            _rot = int(_rot)
                        except (TypeError, ValueError):
                            _rot = None
                        if _rot is not None:
                            break
            if _rot is None:
                # Display Matrix without explicit rotation -> parse displaymatrix
                # determinant (rows 0,1 vs 4,5) if present.
                for _sd in _st.get("side_data_list") or []:
                    _dm = _sd.get("displaymatrix") if isinstance(_sd.get("displaymatrix"), str) else ""
                    if _dm:
                        try:
                            nums = [int(p) for p in _dm.replace("x", " ").split() if p.lstrip("-").isdigit()]
                            if len(nums) >= 4:
                                _det = nums[0] * nums[3] - nums[1] * nums[2]
                                _rot = 0 if _det == 0 else (90 if _det > 0 else 270)
                        except Exception:
                            _rot = None
                        if _rot is not None:
                            break
            _rover = {"rotate": int(_rot or 0) % 360, "w": int(_st.get("width") or 0), "h": int(_st.get("height") or 0)}
            break
    except Exception:
        _rover = {}
    if _rover.get("rotate") in (90, 270):
        progress_cb(1, "rotate", "Нормализация ориентации (автоповорот)...")
        _norm = workdir / "normalized_source.mp4"
        _src_rot = int(_rover.get("rotate") or 0)
        _vw, _vh = int(_rover.get("w") or 0), int(_rover.get("h") or 0)
        if _src_rot in (90, 270) and _vw and _vh:
            _slot = f"{_vh}x{_vw}"
            _cmd = [
                shutil.which("ffmpeg") or "ffmpeg", "-y", "-noautorotate", "-display_rotation:v:0", "0", "-i", str(source_video),
                "-map", "0:v:0", "-map", "0:a:0?", "-vf", f"transpose={1 if _src_rot == 270 else 2},scale={_slot}:flags=lanczos",
                "-c:v", "libx264", "-preset", "fast", "-crf", "0", "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(_norm),
            ]
            _code = subprocess.run(_cmd, capture_output=True, timeout=1800).returncode
            if _code == 0 and _norm.exists():
                source_video = _norm
                progress_cb(4, "rotate", "Ориентация нормализована")
    else:
        if source_video.suffix.lower() == ".mov" and _rover.get("rotate") not in (90, 270):
            # Portrait iPhone .mov without explicit rotation tag: keep as-is (no rotation -> already upright)
            pass

    do_mirror = bool(options.get("mirror_horizontal", False))
    if do_mirror:
        progress_cb(4, "mirror", "Отражение исходного видео по горизонтали...")
        mirrored_source = workdir / "mirrored_source.mp4"
        mirror_cmd = [
            shutil.which("ffmpeg") or "ffmpeg", "-y", "-i", str(source_video),
            "-map", "0:v:0", "-map", "0:a:0?", "-vf", "hflip",
            "-c:v", "libx264", "-preset", "fast", "-crf", "0", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-metadata:s:v:0", "rotate=0", "-movflags", "+faststart",
            str(mirrored_source),
        ]
        code = run_cmd(mirror_cmd, cwd=workdir, log_path=workdir / "mirror.log", timeout=1800)
        if code != 0 or not mirrored_source.exists():
            raise RuntimeError("Не удалось отразить исходное видео")
        source_video = mirrored_source
        progress_cb(5, "mirror", "Исходное видео отражено")

    subtitle_mode = str(options.get("subtitle_mode") or "words")
    subtitle_position = str(options.get("subtitle_position") or "custom")
    animation_cfg = options.get("animation") or {}
    zoom_override = options.get("zoom_timeline_override")
    do_face_track = bool(options.get("face_tracking", False))
    do_zoom = bool(options.get("autozoom", False))
    do_subtitles = bool(options.get("subtitles", True))
    use_bigpickle = bool(options.get("bigpickle", True)) and internet_available()
    edge_mode = str(options.get("edge_mode") or "scale")

    log_cb(f"BigPickle: {use_bigpickle} (интернет {'есть' if internet_available() else 'нет'})")

    tracking_path = workdir / "face_tracking.json"
    zoom_out = workdir / "zoom.mp4"

    # ---- Stage: generate zoom/subtitle configs ----
    progress_cb(5, "configs", "Генерация конфигов зума и подсветки (BigPickle)...")
    env = dict(os.environ)
    env["USE_BIGPICKLE"] = "1" if use_bigpickle else "0"
    code = run_cmd(
        [PYTHON, "-u", str(MONAGE_DIR / "generate_configs_from_transcript_srt.py"), str(transcript)],
        cwd=workdir, log_path=workdir / "configs.log", timeout=900,
        line_cb=lambda line: progress_cb(8, "configs", line.strip()[:120]),
    )
    if code != 0:
        raise RuntimeError("Не удалось сгенерировать конфиги зума/субтитров")
    output_for = {
        "zoom_timeline_config.json": str(zoom_out),
        "subtitle_timeline_config.json": str(workdir / "sub_sentences.mp4"),
        "subtitle_timeline_config_words.json": str(workdir / "sub_words.mp4"),
        "subtitle_timeline_config_phrases.json": str(workdir / "sub_phrases.mp4"),
    }
    for config_name, out_mp4 in output_for.items():
        config_path = workdir / config_name
        if not config_path.exists():
            continue
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        render = cfg.get("render") or {}
        render["video"] = str(source_video.resolve())
        render["output"] = out_mp4
        if config_name.startswith("subtitle"):
            override_y = options.get("subtitle_y")
            if isinstance(override_y, (int, float)) and 0 <= override_y <= 1:
                position_y = float(override_y)
            else:
                position_y = {"bottom": 0.78, "center": 0.5, "top": 0.18}.get(subtitle_position, 0.78)
            override_x = options.get("subtitle_x")
            if isinstance(override_x, (int, float)) and 0 <= override_x <= 1:
                position_x = float(override_x)
            else:
                position_x = 0.12
            render["position"] = {"x": position_x, "y": position_y, "anchor": "left"}
            if isinstance(render.get("style"), dict):
                scale = options.get("subtitle_scale")
                base_font = float(render["style"].get("font_size") or 64)
                if isinstance(scale, (int, float)) and scale > 0:
                    render["style"]["font_size"] = int(round(base_font * max(0.5, min(2.5, float(scale)))))
            if options.get("subtitle_box") is not None:
                render.setdefault("box", {})["enabled"] = bool(options["subtitle_box"])
        cfg["render"] = render
        config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    zoom_config_path = workdir / "zoom_timeline_config.json"
    if zoom_override not in (None, [], "") and zoom_config_path.exists():
        cfg = json.loads(zoom_config_path.read_text(encoding="utf-8"))
        events = []
        for entry in zoom_override if isinstance(zoom_override, list) else []:
            if not isinstance(entry, dict):
                continue
            try:
                time_val = float(entry.get("time") or 0)
                percent = int(entry.get("percent") or 0)
            except (TypeError, ValueError):
                continue
            events.append({"timecode": format_timecode(time_val), "percent": percent})
        if events:
            cfg["render"]["zoom"]["timeline"] = events
            zoom_config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    progress_cb(15, "configs", "Конфиги готовы")

    if do_face_track and do_zoom:
        progress_cb(16, "faces", "Анализ лиц и глаз (MediaPipe)...")
        if not face_model_ready():
            raise RuntimeError("Модель face_landmarker.task не найдена — она скачивается один раз при наличии интернета")
        cmd = [
            PYTHON, "-u", str(MONAGE_DIR / "face_tracking.py"),
            "--input", str(source_video),
            "--output", str(tracking_path),
            "--model", str(FACE_MODEL),
            "--target-fps", "10",
            "--sample-step", "3",
            "--roi-mode", "upper",
            "--roi-upscale", "2.5",
        ]
        last_line: list[str] = ["Анализ лиц..."]
        code = run_cmd(cmd, cwd=workdir, log_path=workdir / "tracking.log",
                       line_cb=lambda line: (last_line.clear(), last_line.append("Анализ лиц: " + line.strip()[:100])))
        if code != 0:
            raise RuntimeError("Анализ лиц не удался")
        ensure_tracking_has_anchors(tracking_path)
        progress_cb(35, "faces", "Анализ лиц завершён")
    else:
        if tracking_path.exists():
            tracking_path.unlink()

    if do_zoom:
        progress_cb(36, "zoom", "Применение автозума...")
        tracking_arg = str(tracking_path) if tracking_path.exists() else ""
        cmd = [
            PYTHON, "-u", str(MONAGE_DIR / "apply_face_zoom_v5.py"),
            "--config", str(workdir / "zoom_timeline_config.json"),
            "--video", str(source_video),
            "--output", str(zoom_out),
            "--zoom-source", "timeline",
            "--edge-mode", edge_mode,
        ] + ([ "--tracking", tracking_arg] if tracking_arg else [])
        last_pct: list[int] = [36]
        def zoom_line(line: str) -> None:
            if "processed" in line:
                match = [p for p in line.split() if "/" in p]
                if match:
                    current, total = match[-1].split("/")
                    try:
                        pct = 36 + int(current) / max(1, int(total)) * 29
                        progress_cb(pct, "zoom", line.strip()[:140])
                    except Exception:
                        pass
        code = run_cmd(cmd, cwd=workdir, log_path=workdir / "zoom.log", line_cb=zoom_line)
        if code != 0:
            raise RuntimeError("Автозум не удался")
        progress_cb(66, "zoom", "Автозум готов")
        base_video = zoom_out
    else:
        base_video = source_video

    # ---- Stage: subtitles ----
    outputs: dict[str, Path] = {}
    if do_subtitles:
        selected_font = str(options.get("subtitle_font") or "")
        allowed_fonts = {item["path"] for item in font_options()}
        font = Path(selected_font) if selected_font in allowed_fonts else (DEFAULT_FONT if DEFAULT_FONT.exists() else None)
        configs = {
            "sentences": ("subtitle_timeline_config.json", "sub_sentences.mp4"),
            "words": ("subtitle_timeline_config_words.json", "sub_words.mp4"),
            "phrases": ("subtitle_timeline_config_phrases.json", "sub_phrases.mp4"),
        }
        config_name, out_name = configs.get(subtitle_mode, configs["sentences"])
        out_mp4 = workdir / out_name
        progress_cb(68, "subtitles", "Наложение цветных субтитров...")
        cmd = [
            PYTHON, "-u", str(MONAGE_DIR / "apply_auto_subtitles_v2.py"),
            "--config", str(workdir / config_name),
            "--video", str(base_video),
            "--output", str(out_mp4),
        ]
        if font:
            cmd += ["--font", str(font)]
        subtitle_end = 89 if bool(animation_cfg.get("enabled")) else 99
        def sub_line(line: str) -> None:
            if "progress:" in line or "frame=" in line:
                try:
                    prefix = line.split("(")[-1].split("%")[0].strip()
                    source_pct = max(0.0, min(100.0, float(prefix)))
                    pct = 68 + source_pct * (subtitle_end - 68) / 100
                    progress_cb(pct, "subtitles", line.strip()[:140])
                except Exception:
                    progress_cb(75, "subtitles", line.strip()[:140])
        code = run_cmd(cmd, cwd=workdir, log_path=workdir / "subtitles.log", line_cb=sub_line)
        if code != 0:
            raise RuntimeError("Наложение субтитров не удалось")
        outputs[subtitle_mode] = out_mp4
        progress_cb(subtitle_end, "subtitles", "Субтитры готовы")
    else:
        outputs["video"] = base_video

    final_mp4 = outputs.get(subtitle_mode) or outputs.get("video")
    if final_mp4 is None or not Path(final_mp4).exists():
        raise RuntimeError("Не создан итоговый файл")
    final_mp4 = Path(final_mp4)

    # ---- Stage: timeline animation (optional) ----
    used_animation: dict[str, Any] | None = None
    if bool(animation_cfg.get("enabled")) and str(animation_cfg.get("id") or ""):
        animation_id = str(animation_cfg.get("id"))
        if not (ANIMATIONS_DIR / animation_id / "manifest.json").exists():
            raise RuntimeError(f"Анимация {animation_id} не найдена в {ANIMATIONS_DIR}")
        anim_out = workdir / "with_animation.mp4"
        anim_start = max(0.0, float(animation_cfg.get("start") or 0.0))
        progress_cb(90, "animation", f"Анимация таймлайна ({animation_id})...")
        cmd = [
            PYTHON, "-u", str(MONAGE_DIR / "apply_timeline_animation.py"),
            "--video", str(final_mp4), "--output", str(anim_out),
            "--animations-dir", str(ANIMATIONS_DIR),
            "--animation", animation_id,
            "--start", str(anim_start),
            "--offset-x", str(int(animation_cfg.get("offset_x") or 0)),
            "--item-size", str(int(animation_cfg.get("item_size") or 72)),
            "--bar", "0",
        ]

        def anim_line(line: str) -> None:
            if "processed" in line:
                try:
                    current, total = [p for p in line.split() if "/" in p][0].split("/")
                    pct = 90 + min(9, int(current) / max(1, int(total)) * 9)
                    progress_cb(pct, "animation", line.strip()[:140])
                except Exception:
                    pass
        code = run_cmd(cmd, cwd=workdir, log_path=workdir / "animation.log", line_cb=anim_line)
        if code != 0:
            raise RuntimeError("Анимация таймлайна не удалась")
        final_mp4 = anim_out
        used_animation = {
            "id": animation_id,
            "start": anim_start,
            "offset_x": int(animation_cfg.get("offset_x") or 0),
            "item_size": int(animation_cfg.get("item_size") or 72),
        }
        progress_cb(99, "animation", "Анимация готова")

    if not final_mp4.exists():
        raise RuntimeError("Не создан итоговый файл")

    delivery_mp4 = workdir / "final.mp4"
    delivery_cmd = [
        shutil.which("ffmpeg") or "ffmpeg", "-y", "-i", str(final_mp4),
        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx265", "-tag:v", "hvc1",
        "-preset", "medium", "-crf", "14",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(delivery_mp4),
    ]
    preview_mp4 = workdir / "preview.mp4"
    preview_cmd = [
        shutil.which("ffmpeg") or "ffmpeg", "-y", "-i", str(final_mp4),
        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264",
        "-profile:v", "high", "-level:v", "4.1", "-preset", "medium", "-crf", "12",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(preview_mp4),
    ]
    progress_cb(99, "finalize", "Одновременное кодирование финального файла и preview...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        delivery_future = executor.submit(subprocess.run, delivery_cmd, capture_output=True, timeout=1800)
        preview_future = executor.submit(subprocess.run, preview_cmd, capture_output=True, timeout=1800)
        delivery_result = delivery_future.result()
        preview_result = preview_future.result()
    if delivery_result.returncode != 0:
        delivery_mp4 = final_mp4
    if preview_result.returncode != 0:
        preview_mp4 = delivery_mp4

    timeline_data: dict[str, Any] = {
        "subtitle_position": subtitle_position,
        "subtitle_x": options.get("subtitle_x"),
        "subtitle_y": options.get("subtitle_y"),
        "subtitle_scale": options.get("subtitle_scale"),
        "subtitle_mode": subtitle_mode,
    }
    try:
        zoom_cfg = json.loads((workdir / "zoom_timeline_config.json").read_text(encoding="utf-8"))
        events = (zoom_cfg.get("render") or {}).get("zoom", {}).get("timeline") or []
        timeline_data["zoom"] = [
            {"timecode": e.get("timecode") if isinstance(e, dict) else None,
             "time": (float(e.get("time")) if isinstance(e, dict) and e.get("time") is not None else None),
             "percent": int(e.get("percent") or 0) if isinstance(e, dict) else 0}
            for e in events if isinstance(e, dict)
        ]
    except Exception:
        pass
    if used_animation:
        timeline_data["animation"] = used_animation

    meta: dict[str, Any] = {
        "job_id": job_id,
        "result_video": str(Path(delivery_mp4).resolve()),
        "preview_video": str(Path(preview_mp4).resolve()),
        "subtitle_mode": subtitle_mode,
        "face_tracking": do_face_track,
        "autozoom": do_zoom,
        "mirror_horizontal": do_mirror,
        "subtitles": do_subtitles,
        "bigpickle": use_bigpickle,
        "animations": list(animation_ids()),
        "timeline": timeline_data,
    }
    try:
        probe = ffprobe(Path(final_mp4))
        meta["duration"] = float(probe.get("format", {}).get("duration") or 0)
    except Exception:
        meta["duration"] = 0
    progress_cb(100, "done", "Рендер завершён")
    return meta


# ---------------- Publishers ----------------

def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transcode_to_fit(source: Path, output: Path, target_bytes: int, duration: float,
                      progress_cb: Callable[[float, str], None]) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    audio_bitrate = 160_000
    vbr = max(400_000, int(target_bytes * 8 * 0.96 / max(0.1, duration)) - audio_bitrate)
    for attempt in range(2):
        output.unlink(missing_ok=True)
        bitrate = max(400_000, int(vbr * (0.88 ** attempt)))
        cmd = [ffmpeg, "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
               "-c:v", "libx264", "-preset", "medium", "-b:v", str(bitrate),
               "-maxrate", str(int(bitrate * 1.08)), "-bufsize", str(int(bitrate * 2.16)),
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
               "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        last = -1
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key not in {"out_time_us", "out_time_ms"}:
                continue
            try:
                percent = min(99, max(1, int(float(value) / 1_000_000 / max(0.1, duration) * 100)))
            except (ValueError, ZeroDivisionError):
                continue
            if percent != last:
                last = percent
                progress_cb(percent, f"Сжатие для публикации: {percent}%")
        code = proc.wait()
        if code != 0 or not output.exists():
            raise RuntimeError("ffmpeg transcode for publishing failed")
        if output.stat().st_size <= target_bytes:
            progress_cb(100, "Файл готов для публикации")
            return
    raise RuntimeError("cannot fit file under target size")


def _publish_targets(
    *,
    job_id: str,
    workdir: Path,
    source_video: Path,
    mirrored_source_video: Path | None,
    original_video: Path,
    transcript_text: str,
    targets: list[dict[str, Any]],
    progress_cb: Callable[[float, str, str], None],
    log_cb: Callable[[str], None],
) -> dict[str, Any]:
    load_secrets_env()
    sw = get_software_defaults()
    results: dict[str, Any] = {}

    reel = _load_module("publish_reel", INSTAPOSTER_DIR / "publish_reel_github.py")
    trial_mod = _load_module("publish_trial", INSTAPOSTER_DIR / "publish_trial_reel_github.py")
    threads_mod = _load_module("publish_threads", INSTAPOSTER_DIR / "publish_threads_carousel_github.py")

    for index, target in enumerate(targets):
        kind = str(target.get("kind") or "")
        label = str(target.get("label") or kind)
        base = index * 100 / max(1, len(targets))
        span = 100 / max(1, len(targets))
        result_key = str(target.get("id") or kind)
        target_source = mirrored_source_video if bool(target.get("mirrored")) else source_video
        if target_source is None or not Path(target_source).exists():
            raise RuntimeError(f"Не подготовлена видео-версия для цели: {label}")

        def sub_progress(p: float, detail: str) -> None:
            progress_cb(min(99, base + span * p / 100), kind, detail)

        log_cb(f"publish target: {label}")
        progress_cb(base + 1, kind, f"Публикация: {label}")

        if kind in {"instagram", "trial"}:
            if not sw["instagram_ready"]:
                if not sw.get("github_ready"):
                    raise RuntimeError(
                        "Instagram: для загрузки видео в GitHub Raw не задан GITHUB_TOKEN"
                    )
                raise RuntimeError("Instagram: не настроен INSTAGRAM_ACCESS_TOKEN / IG_USER_ID")
            from .instagram_publisher import InstagramPublisher
            instagram = InstagramPublisher(
                INSTAPOSTER_DIR, shutil.which("ffmpeg") or "ffmpeg",
                workdir / "publish.log", log_cb,
            )
            variant = "trial" if kind == "trial" else "normal"
            staged = instagram.prepare_variant(
                target_source, workdir / "staging", job_id, variant,
                lambda p, d: sub_progress(p * 0.2, d),
            )
            graph_ip, raw_ip = instagram.resolve_hosts()
            sub_progress(25, "Загрузка в GitHub Raw...")
            _, raw_url = instagram.upload_to_github(staged)
            time.sleep(3)
            instagram.verify_public_video(raw_url, raw_ip, lambda p, d: sub_progress(25 + p * 0.2, d))
            cover_url = ""
            cover_path = Path(str(target.get("cover_path") or ""))
            if cover_path.is_file():
                sub_progress(44, "Загрузка обложки в GitHub Raw...")
                _, cover_url = instagram.upload_cover_to_github(cover_path)
                time.sleep(3)
            caption = str(target.get("caption") or target.get("text") or "")
            container = instagram.create_container(variant, graph_ip, raw_url, caption, cover_url)
            instagram.wait_container(variant, graph_ip, container, lambda p, d: sub_progress(45 + p * 0.5, d))
            publish_started_at = datetime.now(timezone.utc)
            try:
                media_id = instagram.publish_container(variant, graph_ip, container)
                info = instagram.media_info(variant, graph_ip, media_id)
            except Exception as publish_exc:
                log_cb(
                    "media_publish вернул ошибку; проверяю, не была ли публикация "
                    "фактически создана Meta"
                )
                info = instagram.find_recent_publication(
                    variant,
                    graph_ip,
                    caption,
                    publish_started_at - timedelta(minutes=2),
                )
                if not info or not str(info.get("id") or "").strip():
                    raise RuntimeError(
                        "AMBIGUOUS_MEDIA_PUBLISH: Meta могла опубликовать Reel, "
                        "но media_id не подтверждён; автоматический повтор запрещён. "
                        f"Исходная ошибка: {publish_exc}"
                    ) from publish_exc
                media_id = str(info["id"])
                log_cb(f"media_publish восстановлен по списку публикаций: {media_id}")
            permalink = str(info.get("permalink") or "")
            published_result = {"media_id": media_id, "permalink": permalink, "kind": label}
            try:
                from .social_stats import archive_instagram_reel
                archive_dir = archive_instagram_reel(
                    media=info,
                    account_id=str(reel.IG_USER_ID),
                    transcript=transcript_text,
                    source_video=original_video,
                    published_video=staged,
                    post_text=caption,
                    job_id=job_id,
                    variant=result_key,
                )
                published_result["archive_dir"] = str(archive_dir)
                log_cb(f"socialmediastats: {media_id} сохранён в {archive_dir}")
            except Exception as exc:
                published_result["archive_error"] = str(exc)
                log_cb(f"socialmediastats archive failed: {media_id}: {exc}")
            results[result_key] = published_result
            progress_cb(base + span, kind, f"Instagram готов: {permalink or media_id}")

        elif kind == "youtube":
            if not sw["youtube_ready"]:
                raise RuntimeError("YouTube: OAuth token.json не настроен")
            from .youtube_publisher import YouTubePublisher
            youtube = YouTubePublisher(INSTAPOSTER_DIR, Path(sys.executable), workdir / "youtube.log")
            youtube.check_auth()
            youtube_source = workdir / "youtube_staging" / f"{result_key}.mp4"
            youtube.prepare_video(
                target_source,
                youtube_source,
                lambda p, detail: sub_progress(p * 0.25, detail),
            )
            title = str(target.get("title") or target.get("text") or "Мой ролик")
            thumbnail_path = None
            cover_path = Path(str(target.get("cover_path") or ""))
            if cover_path.is_file():
                thumbnail_path = youtube.prepare_thumbnail(
                    cover_path, workdir / "youtube_staging" / f"{result_key}_thumbnail.jpg"
                )
            upload_ready_called: list[bool] = [False]
            def upload_ready() -> None:
                upload_ready_called[0] = True
            def yt_progress(p: float, detail: str) -> None:
                sub_progress(25 + p * 0.75, detail)
            info = youtube.upload(youtube_source, title, yt_progress, upload_ready, thumbnail_path)
            results[result_key] = {"video_id": info["video_id"], "shorts_url": info["shorts_url"], "watch_url": info["watch_url"], "kind": label}
            progress_cb(base + span, kind, f"YouTube готов: {info['shorts_url']}")

        elif kind == "threads":
            if not sw["threads_ready"]:
                if not sw.get("github_ready"):
                    raise RuntimeError(
                        "Threads: для загрузки видео в GitHub Raw не задан GITHUB_TOKEN"
                    )
                raise RuntimeError("Threads: не настроен THREADS_ACCESS_TOKEN / THREADS_USER_ID")
            from .instagram_publisher import InstagramPublisher
            instagram = InstagramPublisher(
                INSTAPOSTER_DIR, shutil.which("ffmpeg") or "ffmpeg",
                workdir / "publish.log", log_cb,
            )
            sub_progress(10, "Подготовка видео для GitHub...")
            staged = instagram.prepare_variant(target_source, workdir / "staging", job_id, "threads", lambda p, d: sub_progress(p, d))
            graph_ip = threads_mod.resolve_host_via_doh(threads_mod.THREADS_HOST)
            raw_ip = threads_mod.resolve_host_via_doh(threads_mod.RAW_HOST)
            sub_progress(35, "Загрузка в GitHub Raw...")
            _, raw_url = reel.publish_video_to_github(staged)
            time.sleep(3)
            sub_progress(45, "Проверка HTTPS-ссылки...")
            reel.check_public_video(raw_url, raw_ip)
            text = str(target.get("text") or target.get("caption") or "")
            container = threads_mod.threads_request(graph_ip, "POST", "me/threads", {
                "media_type": "VIDEO",
                "video_url": raw_url,
                "text": text,
                "access_token": threads_mod.ACCESS_TOKEN,
            })
            container_id = str(container["id"])
            sub_progress(55, "Ожидание обработки Threads...")
            threads_mod.wait_container(graph_ip, container_id, timeout_seconds=180)
            published = threads_mod.threads_request(graph_ip, "POST", "me/threads_publish", {
                "creation_id": container_id,
                "access_token": threads_mod.ACCESS_TOKEN,
            })
            thread_id = str(published["id"])
            info = threads_mod.get_thread_info(graph_ip, thread_id)
            permalink = str(info.get("permalink") or "")
            results[result_key] = {"thread_id": thread_id, "permalink": permalink, "kind": label}
            progress_cb(base + span, kind, f"Threads готов: {permalink or thread_id}")

        elif kind == "telegram":
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not bot_token or not chat_id:
                raise RuntimeError("Telegram: не настроен TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
            duration = 0.0
            try:
                duration = float(ffprobe(target_source).get("format", {}).get("duration") or 0) or 30.0
            except Exception:
                duration = 30.0
            tg_file = workdir / f"{job_id}_tg.mp4"
            sub_progress(15, "Подготовка файла до 45 MB для Telegram...")
            _transcode_to_fit(target_source, tg_file, 45 * 1024 * 1024, duration, sub_progress)
            caption = str(target.get("caption") or target.get("text") or "")
            curl = shutil.which("curl") or "curl"
            cmd = [curl, "-sS", "-X", "POST",
                   f"https://api.telegram.org/bot{bot_token}/sendVideo",
                   "-F", f"chat_id={chat_id}",
                   "-F", "supports_streaming=true",
                   "-F", f"caption={caption}",
                   "-F", "video=@%s;type=video/mp4" % str(tg_file)]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
            if proc.returncode != 0:
                raise RuntimeError(f"Telegram send failed: {proc.stderr[-400:]}")
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                raise RuntimeError(f"Telegram bad response: {proc.stdout[:400]}")
            if not data.get("ok"):
                raise RuntimeError(f"Telegram error: {json.dumps(data, ensure_ascii=False)[:400]}")
            result = data.get("result", {}) or {}
            message_id = str(result.get("message_id") or "")
            results[result_key] = {"message_id": message_id, "kind": label, "ok": True}
            progress_cb(base + span, kind, f"Telegram готов: message_id={message_id}")

        else:
            raise RuntimeError(f"unsupported publish target: {kind}")

    return {"results": results}


def publish_job(
    *,
    job_id: str,
    workdir: Path,
    source_video: Path,
    mirrored_source_video: Path | None = None,
    original_video: Path | None = None,
    transcript_text: str = "",
    targets: list[dict[str, Any]],
    progress_cb: Callable[[float, str, str], None],
    log_cb: Callable[[str], None],
    target_progress_cb: Callable[[str, float, str, str], None] | None = None,
) -> dict[str, Any]:
    instagram_priority = {("trial", False): 0, ("trial", True): 1, ("instagram", False): 2, ("instagram", True): 3}
    instagram_targets = sorted(
        [target for target in targets if str(target.get("kind") or "") in {"trial", "instagram"}],
        key=lambda target: instagram_priority.get(
            (str(target.get("kind") or ""), bool(target.get("mirrored"))), 99
        ),
    )
    youtube_targets = sorted(
        [target for target in targets if str(target.get("kind") or "") == "youtube"],
        key=lambda target: bool(target.get("mirrored")),
    )
    other_targets = [
        target for target in targets
        if str(target.get("kind") or "") not in {"trial", "instagram", "youtube"}
    ]
    lanes = [("instagram", instagram_targets + other_targets), ("youtube", youtube_targets)]
    lanes = [(name, lane_targets) for name, lane_targets in lanes if lane_targets]
    state = {name: 0.0 for name, _ in lanes}
    weights = {name: len(lane_targets) for name, lane_targets in lanes}
    total_weight = max(1, sum(weights.values()))
    state_lock = threading.Lock()

    def update_lane(name: str, percent: float, stage: str, detail: str) -> None:
        with state_lock:
            state[name] = max(state[name], max(0.0, min(100.0, percent)))
            overall = sum(state[key] * weights[key] for key in state) / total_weight
        progress_cb(overall, stage, detail)

    def run_lane(name: str, lane_targets: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
        lane_results: dict[str, Any] = {}
        lane_errors: dict[str, str] = {}
        count = max(1, len(lane_targets))
        for index, target in enumerate(lane_targets):
            target_id = str(target.get("id") or target.get("kind") or f"target_{index}")

            def target_progress(percent: float, stage: str, detail: str) -> None:
                if target_progress_cb:
                    target_progress_cb(target_id, percent, detail, "publishing")
                update_lane(name, (index + percent / 100) * 100 / count, stage, detail)

            try:
                result = _publish_targets(
                    job_id=job_id,
                    workdir=workdir,
                    source_video=source_video,
                    mirrored_source_video=mirrored_source_video,
                    original_video=original_video or source_video,
                    transcript_text=transcript_text,
                    targets=[target],
                    progress_cb=target_progress,
                    log_cb=log_cb,
                )
                lane_results.update(result["results"])
                if target_progress_cb:
                    target_progress_cb(target_id, 100, "Опубликовано", "done")
            except Exception as exc:
                lane_errors[target_id] = str(exc)
                if target_progress_cb:
                    target_progress_cb(target_id, 100, str(exc), "error")
                log_cb(f"publish target failed: {target_id}: {exc}")
            update_lane(name, (index + 1) * 100 / count, target_id, f"Цель завершена: {target_id}")
        return lane_results, lane_errors

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(2, len(lanes))) as executor:
        futures = [executor.submit(run_lane, name, lane_targets) for name, lane_targets in lanes]
        for future in futures:
            lane_results, lane_errors = future.result()
            results.update(lane_results)
            errors.update(lane_errors)
    return {"results": results, "errors": errors}
