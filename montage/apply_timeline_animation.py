from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeline with running animation overlay")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--animations-dir", default=None, help="Root dir with animation presets")
    parser.add_argument("--animation", default="", help="Animation preset id (folder name)")
    parser.add_argument("--start", type=float, default=0.0, help="Overlay appears after N seconds")
    parser.add_argument("--offset-x", type=int, default=0)
    parser.add_argument("--item-size", type=int, default=72)
    parser.add_argument("--bar", type=int, default=0, help="Deprecated; the timeline bar is not rendered")
    args = parser.parse_args()

    input_video = Path(args.video)
    output_video = Path(args.output)
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

    anim_frames: list[Image.Image] = []
    anim_fps = 12.0
    if args.animation:
        anim_dir = Path(args.animations_dir) / args.animation if args.animations_dir else Path(args.animation)
        if (anim_dir / "manifest.json").exists():
            anim_frames, anim_fps = load_animation_frames(anim_dir)

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

    timeline_x = int(width * 0.10)
    timeline_w = int(width * 0.80)
    timeline_y = height - 28
    bottom_padding = 6
    start_time = max(0.0, float(args.start))
    walk_span = max(0.001, duration - start_time)

    try:
        frame_index = 0
        while frame_index < total_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_index / fps
            if t < start_time:
                rel = 0.0
            else:
                rel = min(1.0, (t - start_time) / walk_span)
            base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))

            base_x = timeline_x + int(timeline_w * rel)
            if anim_frames and t >= start_time:
                idx = int(t * anim_fps) % len(anim_frames)
                img = resize_keep(anim_frames[idx], args.item_size)
                x = int(base_x + args.offset_x - img.width / 2)
                y = int(height - img.height - bottom_padding)
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
