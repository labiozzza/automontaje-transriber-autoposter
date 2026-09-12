#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply attention-retention zoom-in / zoom-out to a vertical video using face_tracking.json.

Modes:
- auto/json: use target.zoom from face_tracking.json when available.
- pulse: old gradual zoom-in / zoom-out curve.
- step: fast random stepped zoom every N seconds.
- mixed: random mix of fast stepped moves and gradual moves.
- timeline: deterministic zoom changes by timecodes from JSON config.

The script keeps the eyes near a fixed screen point, or can optionally add eye-follow lag
so the camera follows the eyes with a small delay.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EDGE_MODES = {
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "black": cv2.BORDER_CONSTANT,
}


ZOOM_STEP_VALUES_DEFAULT = "0,20,40,60,80"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fail(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "да", "истина", "вкл"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "нет", "ложь", "выкл"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got: {value!r}")


def smoothstep(t: np.ndarray | float) -> np.ndarray | float:
    return t * t * (3.0 - 2.0 * t)


def load_json_file(path: Path, label: str = "JSON") -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        fail(f"{label} not found: {path}")
    except json.JSONDecodeError as e:
        fail(f"invalid {label}: {path}: {e}")

    if not isinstance(data, dict):
        fail(f"{label} root must be an object")
    return data


def load_json(path: Path) -> dict[str, Any]:
    data = load_json_file(path, "tracking JSON")

    if "camera_targets" not in data:
        fail("JSON must contain camera_targets")
    if not isinstance(data["camera_targets"], list) or not data["camera_targets"]:
        fail("camera_targets is empty")
    return data


def resolve_video_path(args_video: str | None, tracking_path: Path, data: dict[str, Any]) -> Path:
    if args_video:
        return Path(args_video)

    source_path = str(data.get("source", {}).get("path") or "")
    if not source_path:
        fail("video path was not passed and JSON source.path is empty")

    candidates = [
        Path(source_path),
        tracking_path.parent / source_path,
        Path.cwd() / source_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    fail(
        "source video not found. Pass it explicitly: "
        f"--video path/to/source.mp4. JSON source.path={source_path!r}"
    )


def extract_anchor(
    target: dict[str, Any],
    mode: str,
    eye_y_offset: float,
) -> tuple[float, float] | None:
    if not target.get("face_found", False):
        return None

    if mode == "smoothed_eye":
        # Tracker target usually stores eye_center + eye_y_offset.
        # Subtract the offset to get the smoothed eye position.
        t = target.get("target") or {}
        if "x" in t and "y" in t:
            return float(t["x"]), float(t["y"]) - eye_y_offset

    if mode in {"eye_center", "smoothed_eye"}:
        e = target.get("eye_center") or {}
        if "x" in e and "y" in e:
            return float(e["x"]), float(e["y"])

    if mode == "target":
        t = target.get("target") or {}
        if "x" in t and "y" in t:
            return float(t["x"]), float(t["y"])

    return None


def parse_step_values(raw: str, max_percent: float) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            fail(f"invalid --step-values item: {part!r}")
        value = clamp(value, 0.0, 100.0)
        if value <= max_percent + 1e-9:
            values.append(value)

    if not values:
        values = [0.0]
    if 0.0 not in values:
        values.insert(0, 0.0)

    return sorted(set(values))


def level_percent_to_zoom(
    level_percent: np.ndarray,
    zoom_min: float,
    zoom_max: float,
    step_percent_mode: str,
    direct_strength: float = 1.0,
) -> np.ndarray:
    if step_percent_mode == "range":
        # OLD behavior:
        # 0% = zoom_min, 100% = zoom_max.
        # Example: zoom_min=1.0, zoom_max=1.12, level=60 => zoom=1.072.
        return zoom_min + (zoom_max - zoom_min) * (level_percent / 100.0)

    if step_percent_mode == "direct":
        # Strong direct behavior:
        # direct_strength=1.0: 60% => zoom 1.60
        # direct_strength=2.0: 60% => zoom 2.20
        # direct_strength=2.5: 60% => zoom 2.50
        return zoom_min * (1.0 + direct_strength * level_percent / 100.0)

    fail(f"unsupported --step-percent-mode: {step_percent_mode}")




def parse_timecode(value: Any) -> float:
    """Return seconds. Supports 12.5, "12.5", "MM:SS(.ms)", "HH:MM:SS(.ms)"."""
    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds):
            fail(f"invalid timecode: {value!r}")
        return max(0.0, seconds)

    if not isinstance(value, str):
        fail(f"invalid timecode type: {value!r}")

    raw = value.strip().replace(",", ".")
    if not raw:
        fail("empty timecode")

    if ":" not in raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            fail(f"invalid timecode: {value!r}")

    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        fail(f"invalid timecode: {value!r}")

    try:
        nums = [float(part) for part in parts]
    except ValueError:
        fail(f"invalid timecode: {value!r}")

    if len(nums) == 2:
        minutes, seconds = nums
        return max(0.0, minutes * 60.0 + seconds)

    hours, minutes, seconds = nums
    return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)


def zoom_value_from_event(
    event: dict[str, Any],
    zoom_min: float,
    zoom_max: float,
    step_percent_mode: str,
    direct_strength: float,
) -> float:
    if "zoom" in event:
        value = float(event["zoom"])
        if value < 1.0:
            fail(f"event zoom must be >= 1.0: {event}")
        return value

    percent_keys = ["percent", "zoom_percent", "step_percent", "level_percent"]
    for key in percent_keys:
        if key in event:
            percent = np.asarray([float(event[key])], dtype=np.float64)
            return float(
                level_percent_to_zoom(
                    percent,
                    zoom_min,
                    zoom_max,
                    step_percent_mode,
                    direct_strength=direct_strength,
                )[0]
            )

    fail(f"timeline event must contain zoom or percent: {event}")


def get_event_time(event: dict[str, Any]) -> float:
    for key in ("time", "timecode", "at", "start"):
        if key in event:
            return parse_timecode(event[key])
    fail(f"timeline event must contain time/timecode/at/start: {event}")


def expand_timeline_events(raw_events: list[Any], default_zoom: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            fail(f"timeline event must be an object: {raw!r}")

        # Segment syntax: {"start":"00:04", "end":"00:08", "percent":60}
        # It expands into: zoom-in at start, return to default_zoom at end.
        if "start" in raw and "end" in raw:
            start_event = dict(raw)
            start_event.pop("end", None)
            start_event["time"] = start_event.pop("start")
            events.append(start_event)
            end_event = {"time": raw["end"], "zoom": default_zoom}
            events.append(end_event)
            continue

        events.append(dict(raw))

    return events


def build_timeline_zoom(
    total_frames: int,
    fps: float,
    zoom_min: float,
    zoom_max: float,
    raw_events: list[Any],
    default_zoom: float,
    transition: float,
    step_percent_mode: str,
    direct_strength: float,
) -> np.ndarray:
    if not raw_events:
        fail("--zoom-source timeline requires render.zoom.timeline/events in JSON config")
    if transition < 0:
        fail("timeline transition must be >= 0")

    default_zoom = max(1.0, float(default_zoom))
    events = expand_timeline_events(raw_events, default_zoom=default_zoom)

    parsed: list[tuple[float, float]] = []
    for event in events:
        t = get_event_time(event)
        z = zoom_value_from_event(event, zoom_min, zoom_max, step_percent_mode, direct_strength)
        parsed.append((t, z))

    parsed.sort(key=lambda item: item[0])

    out = np.full(total_frames, default_zoom, dtype=np.float64)
    current_zoom = default_zoom

    for event_time, target_zoom in parsed:
        start_frame = int(round(event_time * fps))
        if start_frame >= total_frames:
            continue
        start_frame = max(0, start_frame)
        transition_frames = int(round(transition * fps))

        if transition_frames <= 0:
            out[start_frame:] = target_zoom
            current_zoom = target_zoom
            continue

        end_frame = min(total_frames, start_frame + transition_frames)
        count = max(1, end_frame - start_frame)
        t = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float64)
        w = smoothstep(t)
        out[start_frame:end_frame] = current_zoom + (target_zoom - current_zoom) * w
        if end_frame < total_frames:
            out[end_frame:] = target_zoom
        current_zoom = target_zoom

    return out

def build_pulse_zoom(
    total_frames: int,
    fps: float,
    zoom_min: float,
    zoom_max: float,
    period: float,
) -> np.ndarray:
    if period <= 0:
        fail("--period must be greater than 0")

    out_frames = np.arange(total_frames, dtype=np.float64)
    times = out_frames / fps
    phase = (times % period) / period
    pulse = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
    return zoom_min + (zoom_max - zoom_min) * pulse


def build_random_step_zoom(
    total_frames: int,
    fps: float,
    zoom_min: float,
    zoom_max: float,
    period: float,
    transition: float,
    step_values: list[float],
    step_percent_mode: str,
    direct_strength: float,
    seed: int | None,
    mixed: bool,
    mixed_step_probability: float,
) -> np.ndarray:
    if period <= 0:
        fail("--period must be greater than 0")
    if transition < 0:
        fail("--step-transition must be >= 0")

    transition = min(transition, period)
    rng = random.Random(seed)

    duration = total_frames / fps
    segment_count = int(math.ceil(duration / period)) + 1

    targets = [rng.choice(step_values) for _ in range(segment_count)]
    targets[0] = rng.choice(step_values)

    segment_is_fast = [True] * segment_count
    if mixed:
        p = clamp(mixed_step_probability, 0.0, 1.0)
        segment_is_fast = [rng.random() < p for _ in range(segment_count)]

    levels = np.zeros(total_frames, dtype=np.float64)

    for frame_index in range(total_frames):
        t = frame_index / fps
        segment_index = min(int(t // period), segment_count - 1)
        local_t = t - segment_index * period

        previous_level = 0.0 if segment_index == 0 else targets[segment_index - 1]
        target_level = targets[segment_index]

        if mixed and not segment_is_fast[segment_index]:
            # Gradual move during the whole segment.
            w = smoothstep(clamp(local_t / period, 0.0, 1.0))
            level = previous_level + (target_level - previous_level) * w
        else:
            # Fast move at the start of the segment, then hold.
            if transition <= 0:
                level = target_level
            elif local_t < transition:
                w = smoothstep(clamp(local_t / transition, 0.0, 1.0))
                level = previous_level + (target_level - previous_level) * w
            else:
                level = target_level

        levels[frame_index] = level

    return level_percent_to_zoom(
        levels,
        zoom_min,
        zoom_max,
        step_percent_mode,
        direct_strength=direct_strength,
    )


def apply_eye_follow_lag(xs: np.ndarray, ys: np.ndarray, fps: float, lag_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    if lag_seconds <= 0:
        return xs, ys

    # Exponential smoothing. Bigger lag_seconds = more delayed camera.
    alpha = 1.0 - math.exp(-1.0 / max(1.0, fps * lag_seconds))
    out_x = np.empty_like(xs)
    out_y = np.empty_like(ys)
    out_x[0] = xs[0]
    out_y[0] = ys[0]

    for i in range(1, len(xs)):
        out_x[i] = out_x[i - 1] + alpha * (xs[i] - out_x[i - 1])
        out_y[i] = out_y[i - 1] + alpha * (ys[i] - out_y[i - 1])

    return out_x, out_y


def build_interpolated_series(
    data: dict[str, Any],
    total_frames: int,
    fps: float,
    anchor_mode: str,
    zoom_source: str,
    zoom_min: float,
    zoom_max: float,
    pulse_period: float,
    step_values: list[float],
    step_transition: float,
    step_seed: int | None,
    mixed_step_probability: float,
    step_percent_mode: str,
    direct_strength: float,
    timeline_events: list[Any] | None,
    timeline_default_zoom: float,
    timeline_transition: float,
    eye_follow_lag: bool,
    eye_lag_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    settings = data.get("settings", {}) or {}
    eye_y_offset = float(settings.get("eye_y_offset", 0.06))

    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    has_json_zoom = False

    for item in data["camera_targets"]:
        if not isinstance(item, dict):
            continue

        frame = item.get("frame")
        if frame is None:
            time_value = item.get("time")
            if time_value is None:
                continue
            frame = int(round(float(time_value) * fps))
        frame = int(frame)

        anchor = extract_anchor(item, anchor_mode, eye_y_offset)
        if anchor is None:
            continue

        x, y = anchor
        if not (math.isfinite(x) and math.isfinite(y)):
            continue

        frames.append(frame)
        xs.append(clamp(x, 0.0, 1.0))
        ys.append(clamp(y, 0.0, 1.0))

        t = item.get("target") or {}
        if "zoom" in t and t["zoom"] is not None:
            has_json_zoom = True
            zs.append(float(t["zoom"]))
        else:
            zs.append(float("nan"))

    if not frames:
        fail("no valid face points found in camera_targets")

    order = np.argsort(np.asarray(frames, dtype=np.int64))
    src_frames = np.asarray(frames, dtype=np.float64)[order]
    src_x = np.asarray(xs, dtype=np.float64)[order]
    src_y = np.asarray(ys, dtype=np.float64)[order]
    src_z = np.asarray(zs, dtype=np.float64)[order]

    # Remove duplicated frame numbers. Keep the last item for that frame.
    unique_frames = np.unique(src_frames)
    if len(unique_frames) != len(src_frames):
        last_indices = []
        for frame in unique_frames:
            last_indices.append(np.where(src_frames == frame)[0][-1])
        idx = np.asarray(last_indices, dtype=np.int64)
        src_frames = src_frames[idx]
        src_x = src_x[idx]
        src_y = src_y[idx]
        src_z = src_z[idx]

    out_frames = np.arange(total_frames, dtype=np.float64)
    out_x = np.interp(out_frames, src_frames, src_x)
    out_y = np.interp(out_frames, src_frames, src_y)

    if eye_follow_lag:
        out_x, out_y = apply_eye_follow_lag(out_x, out_y, fps=fps, lag_seconds=eye_lag_seconds)

    use_json_zoom = zoom_source == "json" or (zoom_source == "auto" and has_json_zoom)
    if use_json_zoom and np.isfinite(src_z).any():
        valid = np.isfinite(src_z)
        out_z = np.interp(out_frames, src_frames[valid], src_z[valid])
    elif zoom_source in {"auto", "pulse"}:
        out_z = build_pulse_zoom(
            total_frames=total_frames,
            fps=fps,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            period=pulse_period,
        )
    elif zoom_source == "step":
        out_z = build_random_step_zoom(
            total_frames=total_frames,
            fps=fps,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            period=pulse_period,
            transition=step_transition,
            step_values=step_values,
            step_percent_mode=step_percent_mode,
            direct_strength=direct_strength,
            seed=step_seed,
            mixed=False,
            mixed_step_probability=mixed_step_probability,
        )
    elif zoom_source == "mixed":
        out_z = build_random_step_zoom(
            total_frames=total_frames,
            fps=fps,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            period=pulse_period,
            transition=step_transition,
            step_values=step_values,
            step_percent_mode=step_percent_mode,
            direct_strength=direct_strength,
            seed=step_seed,
            mixed=True,
            mixed_step_probability=mixed_step_probability,
        )
    elif zoom_source == "timeline":
        out_z = build_timeline_zoom(
            total_frames=total_frames,
            fps=fps,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            raw_events=timeline_events or [],
            default_zoom=timeline_default_zoom,
            transition=timeline_transition,
            step_percent_mode=step_percent_mode,
            direct_strength=direct_strength,
        )
    else:
        fail(f"unsupported --zoom-source: {zoom_source}")

    if zoom_source in {"step", "mixed", "timeline"} and step_percent_mode == "direct":
        # In direct mode --step-max-percent is the effective maximum zoom strength.
        # Do not clamp it back to --zoom-max, otherwise 60% would again become tiny.
        out_z = np.maximum(out_z, zoom_min)
    else:
        out_z = np.clip(out_z, zoom_min, zoom_max)
    return out_x, out_y, out_z


def transform_frame(
    frame: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    zoom: float,
    eye_screen_x: float,
    eye_screen_y: float,
    edge_mode: str,
) -> np.ndarray:
    height, width = frame.shape[:2]
    zoom = max(1.0, float(zoom))
    scale = 1.0 / zoom

    # Source coordinate that should land at the fixed eye position on output.
    x1 = anchor_x * width - eye_screen_x * width * scale
    y1 = anchor_y * height - eye_screen_y * height * scale

    if edge_mode == "clamp":
        crop_w = width * scale
        crop_h = height * scale
        x1 = clamp(x1, 0.0, width - crop_w)
        y1 = clamp(y1, 0.0, height - crop_h)
        border_mode = cv2.BORDER_REPLICATE
    else:
        border_mode = EDGE_MODES[edge_mode]

    # M maps destination pixels to source pixels.
    matrix = np.array([[scale, 0.0, x1], [0.0, scale, y1]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=border_mode,
        borderValue=(0, 0, 0),
    )


def start_ffmpeg(output: Path, video_path: Path, width: int, height: int, fps: float, crf: int, preset: str) -> subprocess.Popen:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found in PATH")

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE)




def render_config_root(config_data: dict[str, Any]) -> dict[str, Any]:
    if not config_data:
        return {}
    if isinstance(config_data.get("render"), dict):
        return config_data["render"]
    if isinstance(config_data.get("effects"), dict):
        return config_data["effects"]
    return config_data


def cfg_get(root: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def cfg_bool(root: dict[str, Any], path: str, default: bool) -> bool:
    value = cfg_get(root, path, default)
    return str_to_bool(value)


def cfg_float(root: dict[str, Any], path: str, default: float) -> float:
    value = cfg_get(root, path, default)
    return float(value)


def cfg_int(root: dict[str, Any], path: str, default: int) -> int:
    value = cfg_get(root, path, default)
    return int(value)


def cfg_str(root: dict[str, Any], path: str, default: str | None) -> str | None:
    value = cfg_get(root, path, default)
    if value is None:
        return None
    return str(value)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create face zoom-in / zoom-out while keeping eyes near a fixed screen point."
    )
    parser.add_argument("--tracking", default="face_tracking.json", help="Path to face_tracking.json")
    parser.add_argument("--config", default=None, help="Optional JSON with render/effects settings and zoom timeline")
    parser.add_argument("--video", default=None, help="Path to source video. If omitted, JSON source.path is used")
    parser.add_argument("--output", default="output/face_zoom.mp4", help="Output mp4 path")

    parser.add_argument(
        "--anchor-mode",
        choices=["smoothed_eye", "eye_center", "target"],
        default="smoothed_eye",
        help="Point to keep stable. smoothed_eye = target minus eye_y_offset; best default.",
    )
    parser.add_argument("--eye-screen-x", type=float, default=0.50, help="Final eye X position, normalized 0..1")
    parser.add_argument("--eye-screen-y", type=float, default=0.42, help="Final eye Y position, normalized 0..1")

    parser.add_argument(
        "--zoom-source",
        choices=["auto", "json", "pulse", "step", "mixed", "timeline"],
        default="auto",
        help=(
            "auto/json uses target.zoom from JSON when available; "
            "pulse = old gradual zoom; step = fast random steps; "
            "mixed = fast + gradual segments; timeline = zoom by JSON timecodes"
        ),
    )
    parser.add_argument("--period", type=float, default=4.0, help="Zoom period/segment duration in seconds")
    parser.add_argument("--timeline-transition", type=float, default=0.18, help="Transition duration for --zoom-source timeline")
    parser.add_argument("--zoom-min", type=float, default=None, help="Minimum zoom. Default: JSON settings.zoom_base or 1.0")
    parser.add_argument("--zoom-max", type=float, default=None, help="Maximum zoom. Default: JSON settings.zoom_in or 1.12")

    parser.add_argument(
        "--step-values",
        default=ZOOM_STEP_VALUES_DEFAULT,
        help="Comma-separated stepped zoom levels in percent of configured zoom range. Default: 0,20,40,60,80",
    )
    parser.add_argument(
        "--step-max-percent",
        type=float,
        default=80.0,
        help="Maximum allowed random step percent. Example: 60 => only 0/20/40/60 are used",
    )
    parser.add_argument(
        "--step-percent-mode",
        choices=["direct", "range"],
        default="direct",
        help=(
            "direct: 60 means zoom 1.60 from the base frame; "
            "range: old behavior, 60 means 60%% between --zoom-min and --zoom-max"
        ),
    )
    parser.add_argument(
        "--direct-strength",
        type=float,
        default=1.0,
        help=(
            "Strength multiplier for percent_mode=direct. "
            "1.0: 60%% => 1.60 zoom; 2.0: 60%% => 2.20 zoom; 2.5: 60%% => 2.50 zoom"
        ),
    )

    parser.add_argument(
        "--step-transition",
        type=float,
        default=0.25,
        help="Fast transition duration in seconds for --zoom-source step/mixed",
    )
    parser.add_argument(
        "--step-seed",
        type=int,
        default=None,
        help="Random seed. Set it to reproduce the same random zoom sequence",
    )
    parser.add_argument(
        "--mixed-step-probability",
        type=float,
        default=0.55,
        help="For --zoom-source mixed: probability that a segment is fast stepped instead of gradual",
    )

    parser.add_argument(
        "--eye-follow-lag",
        type=str_to_bool,
        default=False,
        help="true/false. true = camera follows eyes with delay; false = eyes stay locked as much as possible",
    )
    parser.add_argument(
        "--eye-lag-seconds",
        type=float,
        default=0.35,
        help="Delay strength for --eye-follow-lag true. Good range: 0.25..0.60",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["reflect", "replicate", "black", "clamp"],
        default="reflect",
        help="reflect keeps eyes fixed even near edges; clamp avoids artificial edges but eyes may drift",
    )
    parser.add_argument("--crf", type=int, default=18, help="x264 quality: lower is better, 18 is high quality")
    parser.add_argument("--preset", default="veryfast", help="x264 preset: ultrafast, veryfast, medium, slow...")
    parser.add_argument("--limit-frames", type=int, default=0, help="Debug only: process first N frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tracking_path = Path(args.tracking)
    data = load_json(tracking_path)
    settings = data.get("settings", {}) or {}

    config_data: dict[str, Any] = {}
    if args.config:
        config_data = load_json_file(Path(args.config), "render config JSON")
    else:
        # Allows putting a separate "render" key directly inside face_tracking.json.
        config_data = {"render": data.get("render", {})} if isinstance(data.get("render"), dict) else {}

    render_cfg = render_config_root(config_data)
    zoom_cfg = cfg_get(render_cfg, "zoom", {}) or {}
    if not isinstance(zoom_cfg, dict):
        fail("render.zoom must be an object")

    video_arg = cfg_str(render_cfg, "video", args.video)
    output_arg = cfg_str(render_cfg, "output", args.output) or args.output
    video_path = resolve_video_path(video_arg, tracking_path, data)
    output_path = Path(output_arg)

    zoom_min_default = float(settings.get("zoom_base", 1.0))
    zoom_max_default = float(settings.get("zoom_in", 1.12))
    zoom_min = float(cfg_get(zoom_cfg, "min", args.zoom_min if args.zoom_min is not None else zoom_min_default))
    zoom_max = float(cfg_get(zoom_cfg, "max", args.zoom_max if args.zoom_max is not None else zoom_max_default))
    if zoom_min < 1.0:
        zoom_min = 1.0
    if zoom_max < zoom_min:
        fail("zoom.max / --zoom-max must be >= zoom.min / --zoom-min")

    step_max_percent = clamp(float(cfg_get(zoom_cfg, "step_max_percent", args.step_max_percent)), 0.0, 100.0)
    step_values_raw = str(cfg_get(zoom_cfg, "step_values", args.step_values))
    step_values = parse_step_values(step_values_raw, step_max_percent)

    zoom_source = str(cfg_get(zoom_cfg, "source", args.zoom_source))
    period = float(cfg_get(zoom_cfg, "period", args.period))
    step_percent_mode = str(cfg_get(zoom_cfg, "percent_mode", args.step_percent_mode))
    direct_strength = float(cfg_get(zoom_cfg, "direct_strength", args.direct_strength))
    if direct_strength <= 0:
        fail("render.zoom.direct_strength / --direct-strength must be > 0")
    step_transition = float(cfg_get(zoom_cfg, "step_transition", args.step_transition))
    timeline_transition = float(cfg_get(zoom_cfg, "transition", args.timeline_transition))
    timeline_events = cfg_get(zoom_cfg, "timeline", cfg_get(zoom_cfg, "events", None))
    timeline_default_zoom = float(cfg_get(zoom_cfg, "default_zoom", zoom_min))
    if timeline_events is not None and not isinstance(timeline_events, list):
        fail("render.zoom.timeline/events must be a list")

    if zoom_source not in {"auto", "json", "pulse", "step", "mixed", "timeline"}:
        fail(f"unsupported render.zoom.source / --zoom-source: {zoom_source}")
    if step_percent_mode not in {"direct", "range"}:
        fail(f"unsupported render.zoom.percent_mode / --step-percent-mode: {step_percent_mode}")

    anchor_mode = cfg_str(render_cfg, "anchor_mode", args.anchor_mode) or args.anchor_mode
    if anchor_mode not in {"smoothed_eye", "eye_center", "target"}:
        fail(f"unsupported render.anchor_mode / --anchor-mode: {anchor_mode}")

    eye_screen_x = cfg_float(render_cfg, "eye_screen.x", cfg_float(render_cfg, "eye_screen_x", args.eye_screen_x))
    eye_screen_y = cfg_float(render_cfg, "eye_screen.y", cfg_float(render_cfg, "eye_screen_y", args.eye_screen_y))

    raw_eye_follow_lag = cfg_get(render_cfg, "eye_follow_lag", None)
    if isinstance(raw_eye_follow_lag, dict):
        eye_follow_lag = cfg_bool(render_cfg, "eye_follow_lag.enabled", bool(args.eye_follow_lag))
        eye_lag_seconds = cfg_float(render_cfg, "eye_follow_lag.seconds", args.eye_lag_seconds)
    else:
        eye_follow_lag = cfg_bool(render_cfg, "eye_follow_lag", bool(args.eye_follow_lag))
        eye_lag_seconds = float(args.eye_lag_seconds)
    edge_mode = cfg_str(render_cfg, "edge_mode", args.edge_mode) or args.edge_mode
    if edge_mode not in {"reflect", "replicate", "black", "clamp"}:
        fail(f"unsupported render.edge_mode / --edge-mode: {edge_mode}")

    crf = cfg_int(render_cfg, "encoding.crf", args.crf)
    preset = cfg_str(render_cfg, "encoding.preset", args.preset) or args.preset

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or data.get("source", {}).get("fps") or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or data.get("source", {}).get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or data.get("source", {}).get("height") or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or data.get("source", {}).get("frames") or 0)

    if width <= 0 or height <= 0 or fps <= 0:
        fail("cannot detect video width/height/fps")
    if args.limit_frames > 0:
        total_frames = min(total_frames, args.limit_frames) if total_frames else args.limit_frames
    if total_frames <= 0:
        fail("cannot detect frame count")

    xs, ys, zs = build_interpolated_series(
        data=data,
        total_frames=total_frames,
        fps=fps,
        anchor_mode=anchor_mode,
        zoom_source=zoom_source,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        pulse_period=period,
        step_values=step_values,
        step_transition=step_transition,
        step_seed=int(cfg_get(zoom_cfg, "seed", args.step_seed)) if cfg_get(zoom_cfg, "seed", args.step_seed) is not None else None,
        mixed_step_probability=float(cfg_get(zoom_cfg, "mixed_step_probability", args.mixed_step_probability)),
        step_percent_mode=step_percent_mode,
        direct_strength=direct_strength,
        timeline_events=timeline_events,
        timeline_default_zoom=timeline_default_zoom,
        timeline_transition=timeline_transition,
        eye_follow_lag=eye_follow_lag,
        eye_lag_seconds=eye_lag_seconds,
    )

    print(
        "settings: "
        f"zoom_source={zoom_source}, period={period}, zoom_min={zoom_min}, zoom_max={zoom_max}, "
        f"steps={step_values}, step_percent_mode={step_percent_mode}, "
        f"direct_strength={direct_strength}, "
        f"max_effective_zoom={zoom_min * (1 + direct_strength * max(step_values) / 100):.3f}, "
        f"eye_follow_lag={eye_follow_lag}, edge_mode={edge_mode}"
    )
    print(
        f"zoom_debug: actual_min={float(np.min(zs)):.3f}, "
        f"actual_max={float(np.max(zs)):.3f}, "
        f"first={float(zs[0]):.3f}, "
        f"frame_2s={float(zs[int(min(len(zs) - 1, fps * 2))]):.3f}, "
        f"frame_3s={float(zs[int(min(len(zs) - 1, fps * 3))]):.3f}"
    )

    ffmpeg = start_ffmpeg(output_path, video_path, width, height, fps, crf, preset)
    assert ffmpeg.stdin is not None

    frame_index = 0
    try:
        while frame_index < total_frames:
            ok, frame = cap.read()
            if not ok:
                break

            out = transform_frame(
                frame=frame,
                anchor_x=float(xs[frame_index]),
                anchor_y=float(ys[frame_index]),
                zoom=float(zs[frame_index]),
                eye_screen_x=clamp(float(eye_screen_x), 0.0, 1.0),
                eye_screen_y=clamp(float(eye_screen_y), 0.0, 1.0),
                edge_mode=edge_mode,
            )
            ffmpeg.stdin.write(out.tobytes())

            frame_index += 1
            if frame_index % int(max(1, fps * 5)) == 0:
                percent = 100.0 * frame_index / total_frames
                print(f"processed {frame_index}/{total_frames} frames ({percent:.1f}%)")

    except BrokenPipeError:
        fail("ffmpeg pipe closed unexpectedly")
    finally:
        cap.release()
        try:
            ffmpeg.stdin.close()
        except Exception:
            pass
        return_code = ffmpeg.wait()

    if return_code != 0:
        fail(f"ffmpeg failed with code {return_code}")

    print(f"OK: saved {output_path}")


if __name__ == "__main__":
    main()
