from pathlib import Path
from PIL import Image
import numpy as np
import argparse


def make_dark_mask(arr: np.ndarray, threshold: int) -> np.ndarray:
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    a = arr[:, :, 3]

    luma = 0.299 * r + 0.587 * g + 0.114 * b

    return (a > 0) & (luma < threshold)


def find_runs(col_has: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None

    for x, has in enumerate(col_has):
        if has and start is None:
            start = x
        elif not has and start is not None:
            runs.append((start, x - 1))
            start = None

    if start is not None:
        runs.append((start, len(col_has) - 1))

    return runs


def merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []

    merged = [runs[0]]

    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1

        if gap <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def detect_sprite_intervals(mask: np.ndarray, frames: int, max_gap: int, min_width: int):
    col_has = mask.any(axis=0)

    runs = find_runs(col_has)
    runs = merge_close_runs(runs, max_gap=max_gap)
    runs = [(x1, x2) for x1, x2 in runs if (x2 - x1 + 1) >= min_width]

    if len(runs) != frames:
        raise RuntimeError(
            f"Найдено объектов: {len(runs)}, ожидалось: {frames}.\n"
            f"Интервалы: {runs}\n"
            f"Попробуй изменить --detect-threshold или --max-gap."
        )

    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--out", default="sprites_fixed")
    parser.add_argument("--prefix", default="run")

    parser.add_argument("--detect-threshold", type=int, default=140)
    parser.add_argument("--keep-threshold", type=int, default=210)
    parser.add_argument("--max-gap", type=int, default=20)
    parser.add_argument("--min-width", type=int, default=30)
    parser.add_argument("--padding", type=int, default=24)
    parser.add_argument("--bbox-expand", type=int, default=4)

    parser.add_argument("--frame-width", type=int, default=0)
    parser.add_argument("--frame-height", type=int, default=0)
    parser.add_argument("--anchor", choices=["center", "bottom"], default="center")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]

    detect_mask = make_dark_mask(arr, args.detect_threshold)
    keep_mask = make_dark_mask(arr, args.keep_threshold)

    # Делаем настоящий прозрачный фон:
    # всё, что не является тёмной линией, становится прозрачным.
    clean_arr = np.zeros_like(arr)
    clean_arr[keep_mask] = [0, 0, 0, 255]
    clean_img = Image.fromarray(clean_arr, "RGBA")

    intervals = detect_sprite_intervals(
        detect_mask,
        frames=args.frames,
        max_gap=args.max_gap,
        min_width=args.min_width,
    )

    bboxes = []

    for x1, x2 in intervals:
        submask = detect_mask[:, x1:x2 + 1]
        ys, xs = np.where(submask)

        if len(xs) == 0 or len(ys) == 0:
            continue

        bx1 = x1 + xs.min()
        bx2 = x1 + xs.max() + 1
        by1 = ys.min()
        by2 = ys.max() + 1

        bx1 = max(0, bx1 - args.bbox_expand)
        by1 = max(0, by1 - args.bbox_expand)
        bx2 = min(w, bx2 + args.bbox_expand)
        by2 = min(h, by2 + args.bbox_expand)

        bboxes.append((int(bx1), int(by1), int(bx2), int(by2)))

    if len(bboxes) != args.frames:
        raise RuntimeError(f"Получилось bbox: {len(bboxes)}, ожидалось: {args.frames}")

    max_sprite_w = max(x2 - x1 for x1, y1, x2, y2 in bboxes)
    max_sprite_h = max(y2 - y1 for x1, y1, x2, y2 in bboxes)

    frame_w = args.frame_width or (max_sprite_w + args.padding * 2)
    frame_h = args.frame_height or (max_sprite_h + args.padding * 2)

    saved = []

    for i, bbox in enumerate(bboxes, start=1):
        x1, y1, x2, y2 = bbox
        sprite = clean_img.crop((x1, y1, x2, y2))

        frame = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))

        paste_x = (frame_w - sprite.width) // 2

        if args.anchor == "bottom":
            paste_y = frame_h - sprite.height - args.padding
        else:
            paste_y = (frame_h - sprite.height) // 2

        frame.alpha_composite(sprite, (paste_x, paste_y))

        output_path = out_dir / f"{args.prefix}_{i:03}.png"
        frame.save(output_path)
        saved.append(frame)

        print(f"Saved: {output_path} | sprite bbox={sprite.width}x{sprite.height} | frame={frame_w}x{frame_h}")

    sheet = Image.new("RGBA", (frame_w * args.frames, frame_h), (0, 0, 0, 0))

    for i, frame in enumerate(saved):
        sheet.alpha_composite(frame, (i * frame_w, 0))

    sheet_path = out_dir / f"{args.prefix}_sheet_fixed.png"
    sheet.save(sheet_path)

    print()
    print(f"Done: {args.frames} frames")
    print(f"Frame size: {frame_w}x{frame_h}")
    print(f"Fixed sheet: {sheet_path}")


if __name__ == "__main__":
    main()