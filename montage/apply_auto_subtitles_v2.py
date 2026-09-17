#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Burn styled subtitles from subtitle_timeline_config.json into a video.

Works as a separate post-processing step after apply_face_zoom_v5.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        fail(f"JSON not found: {path}")
    except json.JSONDecodeError as e:
        fail(f"Invalid JSON: {path}: {e}")
    if not isinstance(data, dict):
        fail("JSON root must be an object")
    return data


def cfg_get(root: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def parse_hex_color(value: str | list[int] | tuple[int, ...], default_alpha: int = 255) -> tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)):
        vals = [int(x) for x in value]
        if len(vals) == 3:
            return vals[0], vals[1], vals[2], default_alpha
        if len(vals) == 4:
            return vals[0], vals[1], vals[2], vals[3]
        fail(f"Invalid color list: {value!r}")

    raw = str(value).strip()
    if not raw:
        return (255, 255, 255, default_alpha)
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 6:
        r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
        return r, g, b, default_alpha
    if len(raw) == 8:
        r, g, b, a = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16), int(raw[6:8], 16)
        return r, g, b, a
    fail(f"Invalid color: {value!r}. Use #RRGGBB or #RRGGBBAA")


def resolve_path(path_value: str | None, config_path: Path, label: str, must_exist: bool = True) -> Path | None:
    if path_value is None or str(path_value).strip() == "":
        return None

    raw = Path(str(path_value))
    candidates = [raw]
    if not raw.is_absolute():
        candidates += [config_path.parent / raw, Path.cwd() / raw]

    for candidate in candidates:
        if candidate.exists() or not must_exist:
            return candidate

    if must_exist:
        fail(f"{label} not found: {path_value}")
    return raw


def auto_find_font(config_path: Path) -> Path:
    search_dirs = [Path.cwd(), config_path.parent]
    names: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        names.extend(sorted(directory.glob("*.ttf")))
        names.extend(sorted(directory.glob("*.otf")))
    if not names:
        fail("Font was not passed and no .ttf/.otf font found in project root. Pass --font .\\font.ttf")
    return names[0]


def load_font(font_path: Path, font_size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(font_path), font_size)
    except Exception as e:
        fail(f"Cannot load font: {font_path}: {e}")


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_width: int = 0) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_width: int = 0) -> int:
    box = text_bbox(draw, text, font, stroke_width=stroke_width)
    return int(box[2] - box[0])


def normalize_tokens(item: dict[str, Any]) -> list[dict[str, str]]:
    tokens = item.get("tokens") or item.get("words")
    if isinstance(tokens, list) and tokens:
        out: list[dict[str, str]] = []
        for token in tokens:
            if isinstance(token, dict):
                text = str(token.get("text", token.get("word", ""))).strip()
                color = str(token.get("color", "default")).strip().lower() or "default"
            else:
                text = str(token).strip()
                color = "default"
            if text:
                out.append({"text": text, "color": color})
        return out

    text = str(item.get("text", "")).strip()
    return [{"text": x, "color": "default"} for x in text.split() if x.strip()]


def wrap_tokens(
    draw: ImageDraw.ImageDraw,
    tokens: list[dict[str, str]],
    font: ImageFont.FreeTypeFont,
    max_width_px: int,
    max_lines: int,
    stroke_width: int,
) -> list[list[dict[str, str]]]:
    lines: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []

    def line_width(line: list[dict[str, str]]) -> int:
        if not line:
            return 0
        text = " ".join(t["text"] for t in line)
        return text_width(draw, text, font=font, stroke_width=stroke_width)

    for token in tokens:
        candidate = current + [token]
        if current and line_width(candidate) > max_width_px:
            lines.append(current)
            current = [token]
        else:
            current = candidate

    if current:
        lines.append(current)

    if len(lines) <= max_lines:
        return lines

    # Keep the last max_lines and add ellipsis to show the cue was compressed.
    lines = lines[:max_lines]
    if lines and lines[-1]:
        lines[-1][-1] = {**lines[-1][-1], "text": re.sub(r"\.*$", "…", lines[-1][-1]["text"])}
    return lines


def start_ffmpeg(
    output: Path,
    video_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> subprocess.Popen:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found in PATH")

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def color_for_token(color_name: str, style: dict[str, Any]) -> tuple[int, int, int, int]:
    color_name = color_name.lower().strip()
    if color_name in {"green", "yellow", "red"}:
        return parse_hex_color(cfg_get(style, f"highlight.{color_name}", "#FFFFFF"))
    if color_name.startswith("#"):
        return parse_hex_color(color_name)
    return parse_hex_color(style.get("text_color", "#FFFFFF"))


def render_subtitle_on_frame(
    frame_bgr: np.ndarray,
    item: dict[str, Any],
    font: ImageFont.FreeTypeFont,
    render_cfg: dict[str, Any],
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    style = cfg_get(render_cfg, "style", {}) or {}
    box_cfg = cfg_get(render_cfg, "box", {}) or {}
    position = cfg_get(render_cfg, "position", {}) or {}
    layout = cfg_get(render_cfg, "layout", {}) or {}

    uppercase = bool(layout.get("uppercase", False))
    max_lines = int(layout.get("max_lines", 2))
    stroke_width = int(style.get("stroke_width", 5))
    stroke_color = parse_hex_color(style.get("stroke_color", "#000000"))
    line_spacing = float(style.get("line_spacing", 1.12))

    tokens = normalize_tokens(item)
    if uppercase:
        tokens = [{**token, "text": token["text"].upper()} for token in tokens]

    padding_x = int(box_cfg.get("padding_x", 24))
    padding_y = int(box_cfg.get("padding_y", 16))
    anchor = str(position.get("anchor", "center")).strip().lower()
    x_value = float(position.get("x", 0.5))
    y_value = float(position.get("y", 0.78))
    anchor_x = int(width * x_value) if x_value <= 1.0 else int(x_value)
    center_y = int(height * y_value) if y_value <= 1.0 else int(y_value)
    anchor_x = max(0, min(width - 1, anchor_x))

    max_width_value = float(box_cfg.get("max_width", 0.86))
    configured_max_width = int(width * max_width_value) if max_width_value <= 1.0 else int(max_width_value)
    if anchor == "left":
        available_block_w = max(1, width - anchor_x)
        max_width_px = min(configured_max_width, max(1, available_block_w - padding_x * 2))
    else:
        available_block_w = width
        max_width_px = configured_max_width

    base_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA))
    measure = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(measure)

    lines = wrap_tokens(draw, tokens, font=font, max_width_px=max_width_px, max_lines=max_lines, stroke_width=stroke_width)
    if not lines:
        return frame_bgr

    # Metrics.
    sample_bbox = text_bbox(draw, "АБВgyp", font, stroke_width=stroke_width)
    line_height = max(1, int((sample_bbox[3] - sample_bbox[1]) * line_spacing))
    line_widths = [text_width(draw, " ".join(t["text"] for t in line), font, stroke_width=stroke_width) for line in lines]
    block_text_w = max(line_widths) if line_widths else 0
    block_text_h = line_height * len(lines)

    block_w = min(available_block_w, block_text_w + padding_x * 2)
    block_h = block_text_h + padding_y * 2

    if anchor == "left":
        x0 = anchor_x
    else:
        x0 = max(0, min(width - block_w, anchor_x - block_w // 2))
    y0 = max(0, min(height - block_h, center_y - block_h // 2))
    x1 = x0 + block_w
    y1 = y0 + block_h

    block = Image.new("RGBA", (max(1, block_w), max(1, block_h)), (0, 0, 0, 0))
    block_draw = ImageDraw.Draw(block)

    if bool(box_cfg.get("enabled", True)):
        radius = int(box_cfg.get("radius", 26))
        bg = parse_hex_color(box_cfg.get("background", "#00000099"), default_alpha=153)
        block_draw.rounded_rectangle([0, 0, block_w, block_h], radius=radius, fill=bg)

    y = padding_y
    for line, line_w in zip(lines, line_widths):
        x = max(padding_x, (block_w - line_w) // 2)
        for idx, token in enumerate(line):
            text = token["text"]
            if idx < len(line) - 1:
                draw_text = text + " "
            else:
                draw_text = text
            fill = color_for_token(token.get("color", "default"), style)
            block_draw.text(
                (x, y),
                draw_text,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_color,
            )
            x += text_width(block_draw, draw_text, font=font, stroke_width=stroke_width)
        y += line_height

    def composite_clipped(dst: Image.Image, src: Image.Image, dst_x: int, dst_y: int) -> None:
        src_x = 0
        src_y = 0
        if dst_x < 0:
            src_x = -dst_x
            dst_x = 0
        if dst_y < 0:
            src_y = -dst_y
            dst_y = 0
        crop_w = min(src.width - src_x, dst.width - dst_x)
        crop_h = min(src.height - src_y, dst.height - dst_y)
        if crop_w <= 0 or crop_h <= 0:
            return
        part = src.crop((src_x, src_y, src_x + crop_w, src_y + crop_h))
        dst.alpha_composite(part, (dst_x, dst_y))

    rotation = float(position.get("rotation", 0.0) or 0.0)
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    if abs(rotation) > 0.001:
        rendered_block = block.rotate(-rotation, expand=True, resample=Image.BICUBIC)
        rotated_x = x0 if anchor == "left" else int(anchor_x - rendered_block.width / 2)
        composite_clipped(
            overlay,
            rendered_block,
            rotated_x,
            int(center_y - rendered_block.height / 2),
        )
    else:
        composite_clipped(overlay, block, x0, y0)

    composed = Image.alpha_composite(base_img, overlay)
    return cv2.cvtColor(np.array(composed.convert("RGB")), cv2.COLOR_RGB2BGR)


def prepare_subtitles(raw: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_item in raw:
        if not isinstance(raw_item, dict):
            continue
        try:
            start = float(raw_item.get("start"))
            end = float(raw_item.get("end"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        items.append({**raw_item, "start": start, "end": end})
    items.sort(key=lambda x: (x["start"], x["end"]))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Burn color subtitles into video")
    parser.add_argument("--config", default="subtitle_timeline_config.json", help="Subtitle JSON config")
    parser.add_argument("--video", default=None, help="Override render.video")
    parser.add_argument("--output", default=None, help="Override render.output")
    parser.add_argument("--font", default=None, help="Override render.font")
    parser.add_argument("--limit-frames", type=int, default=0, help="Debug only: render first N frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    data = load_json(config_path)
    render_cfg = data.get("render", {}) if isinstance(data.get("render"), dict) else {}

    video_arg = args.video or str(render_cfg.get("video", ""))
    output_arg = args.output or str(render_cfg.get("output", "output/timeline_zoom_subtitles.mp4"))
    font_arg = args.font or render_cfg.get("font")

    video_path = resolve_path(video_arg, config_path, "Video", must_exist=True)
    output_path = resolve_path(output_arg, config_path, "Output", must_exist=False)
    assert video_path is not None and output_path is not None

    font_path = resolve_path(str(font_arg), config_path, "Font", must_exist=True) if font_arg else auto_find_font(config_path)
    assert font_path is not None

    style = cfg_get(render_cfg, "style", {}) or {}
    font_size = int(style.get("font_size", 78))
    font = load_font(font_path, font_size=font_size)

    subtitles = prepare_subtitles(data.get("subtitles", []))
    if not subtitles:
        fail("No valid subtitles found in JSON")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0 or fps <= 0:
        fail("Cannot detect video width/height/fps")
    if args.limit_frames > 0:
        total_frames = min(total_frames, args.limit_frames) if total_frames else args.limit_frames

    crf = int(cfg_get(render_cfg, "encoding.crf", 18))
    preset = str(cfg_get(render_cfg, "encoding.preset", "veryfast"))
    process = start_ffmpeg(output_path, video_path, width, height, fps, crf, preset)
    if process.stdin is None:
        fail("ffmpeg stdin is unavailable")

    subtitle_index = 0
    rendered = 0
    frame_index = 0
    last_decoded_frame: np.ndarray | None = None

    try:
        while True:
            if total_frames and frame_index >= total_frames:
                break
            ok, frame = cap.read()
            if not ok:
                missing = total_frames - frame_index if total_frames else 0
                if last_decoded_frame is None or missing <= 0 or missing > max(1, int(round(fps))):
                    break
                frame = last_decoded_frame.copy()
            else:
                last_decoded_frame = frame.copy()

            t = frame_index / fps
            while subtitle_index < len(subtitles) and subtitles[subtitle_index]["end"] <= t:
                subtitle_index += 1

            if subtitle_index < len(subtitles):
                item = subtitles[subtitle_index]
                if item["start"] <= t < item["end"]:
                    frame = render_subtitle_on_frame(frame, item, font, render_cfg)
                    rendered += 1

            process.stdin.write(frame.tobytes())
            frame_index += 1

            if frame_index % max(1, int(fps * 5)) == 0:
                pct = (frame_index / total_frames * 100.0) if total_frames else 0.0
                print(f"progress: frame={frame_index}/{total_frames or '?'} ({pct:.1f}%)")

    except BrokenPipeError:
        fail("ffmpeg pipe closed unexpectedly")
    finally:
        cap.release()
        try:
            process.stdin.close()
        except Exception:
            pass
        return_code = process.wait()

    if return_code != 0:
        fail(f"ffmpeg failed with code {return_code}")

    print(f"OK: wrote {output_path}")
    print(f"frames={frame_index}, subtitle_frames={rendered}, font={font_path}")


if __name__ == "__main__":
    main()
