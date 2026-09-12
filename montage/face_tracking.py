from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# MediaPipe Face Mesh / Face Landmarker indices.
# Names are from the subject's perspective.
LEFT_EYE_INDICES = [33, 133, 159, 145, 153, 154, 155, 246, 161, 160, 158, 157, 173]
RIGHT_EYE_INDICES = [362, 263, 386, 374, 380, 381, 382, 466, 388, 387, 385, 384, 398]

CENTER_FALLBACK = {"x": 0.5, "y": 0.42, "zoom": 1.0}
LOST_FACE_HOLD_SECONDS = 1.0


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def avg_landmark(landmarks: list[Any], indices: list[int]) -> dict[str, float]:
    xs = [float(landmarks[i].x) for i in indices]
    ys = [float(landmarks[i].y) for i in indices]
    return {"x": clamp01(sum(xs) / len(xs)), "y": clamp01(sum(ys) / len(ys))}


def face_area(landmarks: list[Any]) -> float:
    xs = [float(p.x) for p in landmarks]
    ys = [float(p.y) for p in landmarks]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return max(0.0, width) * max(0.0, height)


def choose_largest_face(all_faces: list[list[Any]]) -> Optional[list[Any]]:
    if not all_faces:
        return None
    return max(all_faces, key=face_area)


def zoom_for_time(time_sec: float, zoom_base: float, zoom_in: float, zoom_period: float) -> float:
    if zoom_period <= 0:
        return zoom_base
    period_index = int(time_sec / zoom_period)
    return zoom_base if period_index % 2 == 0 else zoom_in


def smooth_target(
    previous: Optional[dict[str, float]],
    current: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    if previous is None:
        return {
            "x": clamp01(current["x"]),
            "y": clamp01(current["y"]),
            "zoom": float(current["zoom"]),
        }

    alpha = max(0.0, min(1.0, alpha))
    beta = 1.0 - alpha
    return {
        "x": clamp01(previous["x"] * alpha + current["x"] * beta),
        "y": clamp01(previous["y"] * alpha + current["y"] * beta),
        "zoom": float(previous["zoom"] * alpha + current["zoom"] * beta),
    }


def make_mp_image_from_bgr(frame_bgr: np.ndarray) -> mp.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

def get_roi(frame: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, float]]:
    """
    Возвращает crop и параметры пересчёта координат обратно в полный кадр.
    Координаты bbox нормализованные.
    """

    h, w = frame.shape[:2]

    if mode == "none":
        return frame, {
            "x0": 0.0,
            "y0": 0.0,
            "w": 1.0,
            "h": 1.0,
        }

    if mode == "upper":
        # Под твой тип кадра: человек по центру, лицо в верхней половине.
        x0 = 0.18
        y0 = 0.16
        x1 = 0.82
        y1 = 0.62

        px0 = int(w * x0)
        py0 = int(h * y0)
        px1 = int(w * x1)
        py1 = int(h * y1)

        crop = frame[py0:py1, px0:px1]

        return crop, {
            "x0": x0,
            "y0": y0,
            "w": x1 - x0,
            "h": y1 - y0,
        }

    return frame, {
        "x0": 0.0,
        "y0": 0.0,
        "w": 1.0,
        "h": 1.0,
    }


def upscale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return frame

    h, w = frame.shape[:2]

    return cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,
    )


def remap_point_from_roi(point: Optional[dict[str, float]], roi: dict[str, float]) -> Optional[dict[str, float]]:
    if point is None:
        return None

    return {
        "x": clamp01(roi["x0"] + point["x"] * roi["w"]),
        "y": clamp01(roi["y0"] + point["y"] * roi["h"]),
    }


def remap_landmarks_from_roi(landmarks: list[Any], roi: dict[str, float]) -> list[Any]:
    class Point:
        def __init__(self, x: float, y: float, z: float = 0.0):
            self.x = x
            self.y = y
            self.z = z

    remapped = []

    for p in landmarks:
        remapped.append(
            Point(
                x=clamp01(roi["x0"] + float(p.x) * roi["w"]),
                y=clamp01(roi["y0"] + float(p.y) * roi["h"]),
                z=float(getattr(p, "z", 0.0)),
            )
        )

    return remapped


def draw_roi(frame: np.ndarray, roi: dict[str, float]) -> None:
    h, w = frame.shape[:2]

    x0 = int(roi["x0"] * w)
    y0 = int(roi["y0"] * h)
    x1 = int((roi["x0"] + roi["w"]) * w)
    y1 = int((roi["y0"] + roi["h"]) * h)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
def draw_point(frame: np.ndarray, point: Optional[dict[str, float]], color: tuple[int, int, int], label: str) -> None:
    if point is None:
        return
    h, w = frame.shape[:2]
    x = int(clamp01(point["x"]) * w)
    y = int(clamp01(point["y"]) * h)
    cv2.circle(frame, (x, y), 7, color, -1)
    cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_debug_frame(
    frame: np.ndarray,
    item: dict[str, Any],
    left_eye: Optional[dict[str, float]],
    right_eye: Optional[dict[str, float]],
    eye_center: Optional[dict[str, float]],
) -> np.ndarray:
    out = frame.copy()
    draw_point(out, left_eye, (0, 255, 0), "L")
    draw_point(out, right_eye, (255, 0, 0), "R")
    draw_point(out, eye_center, (0, 255, 255), "EYE")
    draw_point(out, item["target"], (0, 0, 255), "TARGET")

    text = f"face_found={item['face_found']} zoom={item['target']['zoom']:.3f} t={item['time']:.2f}s"
    cv2.rectangle(out, (12, 12), (760, 55), (0, 0, 0), -1)
    cv2.putText(out, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def process_video(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    if not input_path.exists():
        fail(f"input video not found: {input_path}")
    if not model_path.exists():
        fail(f"face landmarker model not found: {model_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        fail("cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0 or fps <= 0:
        cap.release()
        fail("cannot read video metadata")

    duration = frame_count / fps if frame_count > 0 else 0.0
    fps_step = max(1, int(round(fps / args.target_fps))) if args.target_fps > 0 else 1
    effective_step = max(1, int(args.sample_step), fps_step)

    debug_writer = None
    if args.debug_video:
        debug_path = Path(args.debug_video)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_fps = max(1.0, fps / effective_step)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(str(debug_path), fourcc, debug_fps, (width, height))
        if not debug_writer.isOpened():
            cap.release()
            fail(f"cannot create debug video: {debug_path}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=max(1, int(args.max_faces)),
        min_face_detection_confidence=float(args.min_confidence),
        min_face_presence_confidence=float(args.min_confidence),
        min_tracking_confidence=float(args.min_confidence),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    camera_targets: list[dict[str, Any]] = []
    analyzed = 0
    found = 0
    previous_smoothed: Optional[dict[str, float]] = None
    last_valid_raw: Optional[dict[str, float]] = None
    last_valid_time: Optional[float] = None

    with FaceLandmarker.create_from_options(options) as landmarker:
        pbar_total = frame_count if frame_count > 0 else None
        with tqdm(total=pbar_total, desc="Analyzing frames", unit="frame") as pbar:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % effective_step != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                time_sec = frame_idx / fps
                timestamp_ms = int(round(time_sec * 1000))
                raw_zoom = zoom_for_time(time_sec, args.zoom_base, args.zoom_in, args.zoom_period)

                roi_frame, roi_info = get_roi(frame, args.roi_mode)
                roi_frame = upscale_frame(roi_frame, args.roi_upscale)

                mp_image = make_mp_image_from_bgr(roi_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                selected_face = choose_largest_face(result.face_landmarks)

                if selected_face is not None:
                    selected_face = remap_landmarks_from_roi(selected_face, roi_info)

                left_eye = None
                right_eye = None
                eye_center = None
                face_found = selected_face is not None

                if face_found:
                    left_eye = avg_landmark(selected_face, LEFT_EYE_INDICES)
                    right_eye = avg_landmark(selected_face, RIGHT_EYE_INDICES)
                    eye_center = {
                        "x": clamp01((left_eye["x"] + right_eye["x"]) / 2.0),
                        "y": clamp01((left_eye["y"] + right_eye["y"]) / 2.0),
                    }
                    raw_target = {
                        "x": eye_center["x"],
                        "y": clamp01(eye_center["y"] + args.eye_y_offset),
                        "zoom": raw_zoom,
                    }
                    last_valid_raw = raw_target
                    last_valid_time = time_sec
                    found += 1
                else:
                    if last_valid_raw is not None and last_valid_time is not None and (time_sec - last_valid_time) <= LOST_FACE_HOLD_SECONDS:
                        raw_target = dict(last_valid_raw)
                        raw_target["zoom"] = raw_zoom
                    else:
                        raw_target = dict(CENTER_FALLBACK)

                smoothed_target = smooth_target(previous_smoothed, raw_target, args.smooth_alpha)
                previous_smoothed = smoothed_target

                item = {
                    "frame": frame_idx,
                    "time": round(time_sec, 6),
                    "face_found": bool(face_found),
                    "confidence": 1.0 if face_found else 0.0,
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "eye_center": eye_center,
                    "target": {
                        "x": round(smoothed_target["x"], 6),
                        "y": round(smoothed_target["y"], 6),
                        "zoom": round(smoothed_target["zoom"], 6),
                    },
                }
                camera_targets.append(item)
                analyzed += 1

                if debug_writer is not None:
                    debug_frame = draw_debug_frame(
                        frame=frame,
                        item=item,
                        left_eye=left_eye,
                        right_eye=right_eye,
                        eye_center=eye_center,
                    )

                    draw_roi(debug_frame, roi_info)
                    debug_writer.write(debug_frame)

                frame_idx += 1
                pbar.update(1)

    cap.release()
    if debug_writer is not None:
        debug_writer.release()

    if analyzed == 0:
        fail("no frames processed")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source": {
            "path": str(input_path).replace("\\", "/"),
            "width": width,
            "height": height,
            "fps": round(fps, 6),
            "duration": round(duration, 6),
            "frames": frame_count,
        },
        "settings": {
            "sample_step": int(args.sample_step),
            "target_fps": float(args.target_fps),
            "effective_step": int(effective_step),
            "min_confidence": float(args.min_confidence),
            "zoom_base": float(args.zoom_base),
            "zoom_in": float(args.zoom_in),
            "zoom_period": float(args.zoom_period),
            "smooth_window": int(args.smooth_window),
            "smooth_alpha": float(args.smooth_alpha),
            "eye_y_offset": float(args.eye_y_offset),
            "max_faces": int(args.max_faces),
            "model": str(model_path).replace("\\", "/"),
        },
        "camera_targets": camera_targets,
    }

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    detection_rate = found / analyzed * 100.0
    print(f"OK: face tracking saved to {output_path}")
    print(f"Video: {width}x{height}, fps={fps:.3f}, duration={duration:.2f}s")
    print(f"Frames analyzed: {analyzed}")
    print(f"Face detection rate: {detection_rate:.1f}%")
    if args.debug_video:
        print(f"Debug video saved to {args.debug_video}")

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face and eye tracking via MediaPipe Face Landmarker")
    parser.add_argument("--input", required=True, help="путь к исходному видео")
    parser.add_argument("--output", required=True, help="путь сохранения face_tracking.json")
    parser.add_argument("--model", default="models/face_landmarker.task", help="путь к модели Face Landmarker")
    parser.add_argument("--sample-step", type=int, default=3, help="анализировать каждый N-й кадр")
    parser.add_argument("--target-fps", type=float, default=10.0, help="целевая частота точек трекинга")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="минимальная уверенность детекции/трекинга")
    parser.add_argument("--zoom-base", type=float, default=1.0, help="обычный зум")
    parser.add_argument("--zoom-in", type=float, default=1.12, help="усиленный зум")
    parser.add_argument("--zoom-period", type=float, default=3.0, help="период смены зума в секундах")
    parser.add_argument("--smooth-window", type=int, default=7, help="оставлено для совместимости; используется smooth-alpha")
    parser.add_argument("--smooth-alpha", type=float, default=0.85, help="коэффициент exponential smoothing")
    parser.add_argument("--eye-y-offset", type=float, default=0.06, help="смещение камеры вниз от центра глаз")
    parser.add_argument("--max-faces", type=int, default=5, help="сколько лиц искать; выбирается самое крупное")
    parser.add_argument("--debug-video", default=None, help="опциональный путь debug-видео")
    parser.add_argument(
        "--roi-mode",
        default="upper",
        choices=["none", "upper"],
        help="режим предварительного crop перед Face Landmarker",
    )

    parser.add_argument(
        "--roi-upscale",
        type=float,
        default=2.5,
        help="во сколько раз увеличить crop перед Face Landmarker",
    )
    return parser.parse_args()


if __name__ == "__main__":
    process_video(parse_args())
