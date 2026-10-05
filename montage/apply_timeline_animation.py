from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ffprobe_meta(video: Path) -> dict:
    cmd = [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries",
           "format=duration:stream=width,height,r_frame_rate,codec_type", "-of", "json", str(video)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[-400:]}")
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    fps = 30.0
    width = 0
    height = 0
    for stream in data.get("streams") or []:
        if stream.get("codec_type") != "video":
            continue
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        try:
            num, den = map(float, str(stream.get("r_frame_rate") or "30/1").split("/"))
            fps = num / den if den else 30.0
        except Exception:
            fps = 30.0
        break
    frames = int(duration * fps)
    return {"duration": duration, "fps": fps, "width": width, "height": height, "frames": frames}


def load_animation_frames(anim_dir: Path) -> tuple[list[Image.Image], float]:
    try:
        manifest = json.loads((anim_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    frames_raw = manifest.get("frames") if isinstance(manifest, dict) else []
    frames: list[Image.Image] = []
    for frame in frames_raw or []:
        file_name = Path(str(frame.get("file") if isinstance(frame, dict) else frame)).name
        path = anim_dir / file_name
        if path.exists():
            try:
                frames.append(Image.open(path).convert("RGBA"))
            except Exception:
                pass
    fps = float(manifest.get("fps") or 12) if isinstance(manifest, dict) else 12.0
    return frames, fps


def resize_keep(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = size / max(w, h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def overlay_rgba(base: Image.Image, overlay: Image.Image, x: int, y: int) -> None:
    base.alpha_composite(overlay, (x, y))


def _path_length(points: list[list[float]]) -> float:
    return sum(
        math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
        for index in range(1, len(points))
    )


def _visible_points(points: list[list[float]], length: float) -> list[list[float]]:
    if not points or length <= 0:
        return []
    visible = [points[0]]
    remaining = length
    for index in range(1, len(points)):
        start, end = points[index - 1], points[index]
        segment = math.hypot(end[0] - start[0], end[1] - start[1])
        if segment <= remaining:
            visible.append(end)
            remaining -= segment
            continue
        if segment > 0:
            ratio = max(0.0, remaining / segment)
            visible.append([
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            ])
        break
    return visible


def _ease_hand_motion(progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return progress * progress * (3.0 - 2.0 * progress)


_CHART_AXIS_COLOR = (255, 255, 255, 235)
_CHART_LABEL_COLOR = (255, 255, 255, 255)


def _chart_font(scale: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(__file__).resolve().parent / "fonts" / "Comfortaa.ttf",
        Path("montage/fonts/Comfortaa.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), max(10, int(round(44 * scale))))
            except OSError:
                break
    return ImageFont.load_default()


def _arrowhead(painter: ImageDraw.ImageDraw, tip: tuple[int, int], direction_deg: float,
               base_len: int, half_width: int, fill: tuple[int, int, int, int]) -> None:
    rad = math.radians(direction_deg)
    base = (tip[0] - base_len * math.cos(rad), tip[1] - base_len * math.sin(rad))
    perp = math.radians(direction_deg + 90.0)
    left = (base[0] + half_width * math.cos(perp), base[1] + half_width * math.sin(perp))
    right = (base[0] - half_width * math.cos(perp), base[1] - half_width * math.sin(perp))
    painter.polygon([tip, left, right], fill=fill)


def _draw_chart_widgets(painter: ImageDraw.ImageDraw, canvas: Image.Image,
                        scale: float, viz: dict) -> None:
    x_begin, x_end, y_level = 120, 880, 920
    y_top = 80
    line_width = max(1, int(round(3 * scale)))
    arrow = max(8, int(round(26 * scale)))
    px = lambda dx, dy: (int(round(dx * scale)), int(round(dy * scale)))  # noqa: E731

    painter.line([px(x_begin, y_level), px(x_end, y_level)], fill=_CHART_AXIS_COLOR, width=line_width)
    _arrowhead(painter, px(x_end, y_level), 0.0, arrow, max(1, int(round(arrow * 0.6))), _CHART_AXIS_COLOR)
    painter.line([px(x_begin, y_top), px(x_begin, y_level)], fill=_CHART_AXIS_COLOR, width=line_width)
    _arrowhead(painter, px(x_begin, y_top), 90.0, arrow, max(1, int(round(arrow * 0.6))), _CHART_AXIS_COLOR)

    axes = viz.get("axes") if isinstance(viz.get("axes"), dict) else {}
    x_label = str(axes.get("x") or "").strip()
    y_label = str(axes.get("y") or "").strip()
    font = _chart_font(scale)

    if x_label:
        painter.text(px(500, 982), x_label, font=font, fill=_CHART_LABEL_COLOR, anchor="mm")
    if y_label:
        try:
            tmp = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(tmp)
            bbox = temp_draw.textbbox((0, 0), y_label, font=font)
            tmp = tmp.crop((0, 0, max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])))
            temp_draw = ImageDraw.Draw(tmp)
            temp_draw.text((bbox[0], bbox[1]), y_label, font=font, fill=_CHART_LABEL_COLOR, anchor="la")
            rotated = tmp.rotate(-90, expand=True)
            dest = (int(round(52 * scale - rotated.width / 2)), int(round(500 * scale - rotated.height / 2)))
            canvas.alpha_composite(rotated, dest)
        except OSError:
            painter.text(px(48, 500), y_label, font=font, fill=_CHART_LABEL_COLOR, anchor="mm")


def render_drawing(drawing: dict, size: int, elapsed: float, draw_speed: float = 350.0) -> Image.Image:
    source_width = max(1, int(drawing.get("width") or 1000))
    source_height = max(1, int(drawing.get("height") or 1000))
    scale = size / max(source_width, source_height)
    output = Image.new(
        "RGBA",
        (max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))),
        (0, 0, 0, 0),
    )
    painter = ImageDraw.Draw(output)
    cursor = 0.0
    elapsed = max(0.0, elapsed)
    draw_speed = max(1.0, draw_speed)
    stroke_pause = min(0.1, max(0.0, float(drawing.get("stroke_pause", 0.03))))
    for path in drawing.get("paths") or []:
        points = path.get("points") or []
        full_length = _path_length(points)
        if full_length <= 0:
            continue
        stroke_duration = max(0.04, full_length / draw_speed)
        progress = _ease_hand_motion((elapsed - cursor) / stroke_duration)
        cursor += stroke_duration + stroke_pause
        if progress <= 0:
            continue
        shown = _visible_points(points, full_length * progress)
        complete = progress >= 1.0
        if complete and path.get("closed") and shown:
            shown = [*shown, shown[0]]
        if len(shown) < 2:
            continue
        color = str(path.get("stroke") or "#FFFFFF").lstrip("#")
        if len(color) not in {6, 8}:
            color = "FFFFFF"
        channels = tuple(int(color[index:index + 2], 16) for index in range(0, len(color), 2))
        alpha = channels[3] if len(channels) == 4 else 255
        alpha = int(alpha * min(1.0, max(0.0, float(path.get("opacity") or 1))))
        scaled = [(round(point[0] * scale), round(point[1] * scale)) for point in shown]
        painter.line(
            scaled,
            fill=(*channels[:3], alpha),
            width=max(1, int(round(float(path.get("stroke_width") or 8) * scale))),
            joint="curve",
        )
    viz = drawing.get("viz") if isinstance(drawing.get("viz"), dict) else {}
    if viz.get("type") == "chart":
        _draw_chart_widgets(painter, output, scale, viz)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeline animation overlays")
    parser.add_argument("--config", help="JSON config with video, output and overlays")
    parser.add_argument("--video")
    parser.add_argument("--output")
    parser.add_argument("--animations-dir", default=None, help="Root dir with animation presets")
    parser.add_argument("--animation", default="", help="Animation preset id (folder name)")
    parser.add_argument("--start", type=float, default=0.0, help="Overlay appears after N seconds")
    parser.add_argument("--offset-x", type=int, default=0)
    parser.add_argument("--item-size", type=int, default=72)
    parser.add_argument("--bar", type=int, default=0, help="Deprecated; the timeline bar is not rendered")
    args = parser.parse_args()

    config: dict = {}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    video_value = config.get("video") or args.video
    output_value = config.get("output") or args.output
    if not video_value or not output_value:
        raise RuntimeError("video and output are required")
    input_video = Path(str(video_value))
    output_video = Path(str(output_value))
    output_video.parent.mkdir(parents=True, exist_ok=True)

    meta = ffprobe_meta(input_video)
    fps = float(meta["fps"] or 30.0)
    width = int(meta["width"])
    height = int(meta["height"])
    total_frames = int(meta["frames"])
    duration = float(meta["duration"] or 0.0)
    if total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Bad video metadata for timeline animation")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    animations_dir_value = config.get("animations_dir") or args.animations_dir
    animations_dir = Path(str(animations_dir_value)) if animations_dir_value else None
    overlays = config.get("overlays") if isinstance(config.get("overlays"), list) else []
    if not overlays and args.animation:
        overlays = [{
            "id": "legacy-overlay",
            "animation_id": args.animation,
            "start": max(0.0, float(args.start)),
            "end": duration,
            "size": max(20, int(args.item_size)),
            "offset_x": int(args.offset_x),
            "motion": "progress",
            "z_index": 0,
        }]

    frame_cache: dict[str, tuple[list[Image.Image], float]] = {}
    resized_cache: dict[tuple[str, int], list[Image.Image]] = {}
    prepared: list[dict] = []
    for index, raw in enumerate(overlays[:20]):
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "animation")
        if kind == "drawing":
            drawing = raw.get("drawing")
            if not isinstance(drawing, dict) or not isinstance(drawing.get("paths"), list):
                continue
            start = max(0.0, float(raw.get("start") or 0.0))
            end = min(duration, float(raw.get("end") if raw.get("end") is not None else duration))
            if end <= start:
                continue
            prepared.append({
                "uid": str(raw.get("uid") or f"overlay-{index + 1}"),
                "kind": "drawing",
                "drawing": drawing,
                "draw_speed": max(10.0, min(50000.0, float(raw.get("draw_speed") or 350))),
                "size": max(20, int(raw.get("size") or 360)),
                "start": start,
                "end": end,
                "x": min(1.0, max(0.0, float(raw.get("x") if raw.get("x") is not None else 0.5))),
                "y": min(1.0, max(0.0, float(raw.get("y") if raw.get("y") is not None else 0.5))),
                "z_index": int(raw.get("z_index") if raw.get("z_index") is not None else index),
                "motion": "static",
                "offset_x": 0,
            })
            continue
        if kind == "image":
            image_path = Path(str(raw.get("image_path") or ""))
            if not image_path.is_file():
                continue
            source_key = f"image:{image_path.resolve()}"
            if source_key not in frame_cache:
                try:
                    frame_cache[source_key] = ([Image.open(image_path).convert("RGBA")], 1.0)
                except Exception:
                    frame_cache[source_key] = ([], 1.0)
        else:
            animation_id = str(raw.get("animation_id") or raw.get("id") or "")
            if not animation_id or Path(animation_id).name != animation_id or animations_dir is None:
                continue
            source_key = f"animation:{animation_id}"
            if source_key not in frame_cache:
                frame_cache[source_key] = load_animation_frames(animations_dir / animation_id)
        frames, overlay_fps = frame_cache[source_key]
        if not frames:
            continue
        start = max(0.0, float(raw.get("start") or 0.0))
        end = min(duration, float(raw.get("end") if raw.get("end") is not None else duration))
        if end <= start:
            continue
        size = max(20, int(raw.get("size") or 180))
        resized_key = (source_key, size)
        if resized_key not in resized_cache:
            resized_cache[resized_key] = [resize_keep(frame, size) for frame in frames]
        prepared.append({
            "uid": str(raw.get("uid") or f"overlay-{index + 1}"),
            "kind": kind,
            "frames": resized_cache[resized_key],
            "fps": max(0.1, float(overlay_fps)),
            "start": start,
            "end": end,
            "x": min(1.0, max(0.0, float(raw.get("x") if raw.get("x") is not None else 0.5))),
            "y": min(1.0, max(0.0, float(raw.get("y") if raw.get("y") is not None else 0.5))),
            "z_index": int(raw.get("z_index") if raw.get("z_index") is not None else index),
            "motion": str(raw.get("motion") or "static"),
            "offset_x": int(raw.get("offset_x") or 0),
        })
    prepared.sort(key=lambda item: item["z_index"])

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    process = subprocess.Popen(
        [
            ffmpeg, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
            "-i", str(input_video),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "0",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            "-movflags", "+faststart",
            "-shortest", str(output_video),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None

    bottom_padding = 6

    try:
        frame_index = 0
        while frame_index < total_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_index / fps
            base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))

            for overlay in prepared:
                if not (overlay["start"] <= t < overlay["end"]):
                    continue
                if overlay["kind"] == "drawing":
                    img = render_drawing(
                        overlay["drawing"],
                        overlay["size"],
                        t - overlay["start"],
                        overlay["draw_speed"],
                    )
                else:
                    frame_time = t if overlay["motion"] == "progress" else t - overlay["start"]
                    idx = int(frame_time * overlay["fps"]) % len(overlay["frames"])
                    img = overlay["frames"][idx]
                if overlay["motion"] == "progress":
                    rel = min(1.0, (t - overlay["start"]) / max(0.001, overlay["end"] - overlay["start"]))
                    center_x = width * (0.10 + 0.80 * rel) + overlay["offset_x"]
                    center_y = height - img.height / 2 - bottom_padding
                else:
                    center_x = width * overlay["x"]
                    center_y = height * overlay["y"]
                x = int(center_x - img.width / 2)
                y = int(center_y - img.height / 2)
                overlay_rgba(base, img, x, y)

            out = cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)
            process.stdin.write(out.tobytes())
            frame_index += 1
            if frame_index % max(1, int(fps * 5)) == 0:
                print(f"processed {frame_index}/{total_frames} frames ({frame_index / total_frames * 100:.1f}%)", flush=True)
    finally:
        cap.release()
        try:
            process.stdin.close()
        except Exception:
            pass
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"timeline animation ffmpeg failed with code {code}")
    print("OK", flush=True)


if __name__ == "__main__":
    main()
