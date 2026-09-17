#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import whisper


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FACTORY = Path.home() / "Desktop" / "factory"
CATEGORIES = ("hook", "main", "final")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
FONT_PATH = PROJECT_DIR / "montage" / "fonts" / "Comfortaa.ttf"
RENDERER = PROJECT_DIR / "montage" / "apply_auto_subtitles_v2.py"


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def collect_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.get("segments") or []:
        segment_words = segment.get("words") or []
        if segment_words:
            for item in segment_words:
                text = str(item.get("word") or "").strip()
                start = float(item.get("start") or 0)
                end = float(item.get("end") or start)
                if text and end > start:
                    words.append({"start": start, "end": end, "text": text})
            continue

        parts = str(segment.get("text") or "").split()
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        if not parts or end <= start:
            continue
        duration = (end - start) / len(parts)
        for index, text in enumerate(parts):
            words.append({
                "start": start + duration * index,
                "end": start + duration * (index + 1),
                "text": text,
            })
    return words


def video_width(source: Path) -> int:
    result = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width", "-of", "default=nw=1:nk=1", str(source),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Не удалось определить ширину {source}")
    return int(result.stdout.strip())


def subtitle_config(source: Path, output: Path, words: list[dict[str, Any]], font_size: int) -> dict[str, Any]:
    return {
        "render": {
            "video": str(source),
            "output": str(output),
            "font": str(FONT_PATH),
            "position": {"x": 100, "y": 0.70, "anchor": "left"},
            "box": {
                "enabled": False,
                "max_width": 0.90,
                "padding_x": 4,
                "padding_y": 4,
                "radius": 0,
                "background": "#00000000",
            },
            "style": {
                "font_size": font_size,
                "line_spacing": 1.0,
                "text_color": "#FFFFFF",
                "stroke_color": "#000000",
                "stroke_width": 4,
                "highlight": {},
            },
            "layout": {"max_lines": 1, "uppercase": False},
            "encoding": {"crf": 14, "preset": "medium"},
        },
        "subtitles": [
            {
                "start": word["start"],
                "end": word["end"],
                "text": word["text"],
                "tokens": [{"text": word["text"], "color": "default"}],
            }
            for word in words
        ],
    }


def process_video(
    model: Any,
    source: Path,
    destination_dir: Path,
    language: str,
    force: bool,
    preserve_text: bool,
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_video = destination_dir / source.name
    output_text = destination_dir / f"{source.stem}.txt"
    preserved_text = output_text.read_bytes() if preserve_text and output_text.is_file() else None
    if not force and output_video.is_file() and output_text.is_file():
        print(f"[skip] {source}: результат уже существует", flush=True)
        return

    print(f"[transcribe] {source}", flush=True)
    result = model.transcribe(
        str(source), language=language, word_timestamps=True,
        fp16=False, verbose=False, condition_on_previous_text=True,
    )
    words = collect_words(result)
    if not words:
        raise RuntimeError(f"Whisper не нашёл слов в {source}")
    transcript = str(result.get("text") or "").strip()
    width = video_width(source)
    font_size = max(1, round(40 * width / 1080))

    with tempfile.TemporaryDirectory(prefix="factory-render-") as temporary_dir:
        temporary_dir = Path(temporary_dir)
        temporary_video = destination_dir / f".{source.stem}.rendering{source.suffix}"
        config_path = temporary_dir / "subtitles.json"
        config_path.write_text(
            json.dumps(
                subtitle_config(source.resolve(), temporary_video.resolve(), words, font_size),
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        temporary_video.unlink(missing_ok=True)
        print(f"[render] {source.name}: {len(words)} слов, шрифт {font_size}px", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-u", str(RENDERER), "--config", str(config_path)],
                cwd=str(PROJECT_DIR), check=True,
            )
            if not temporary_video.is_file():
                raise RuntimeError(f"Renderer не создал {temporary_video}")
            os.replace(temporary_video, output_video)
            if not preserve_text:
                atomic_text(output_text, transcript)
            elif preserved_text is not None and output_text.read_bytes() != preserved_text:
                raise RuntimeError(f"TXT был неожиданно изменён: {output_text}")
        finally:
            temporary_video.unlink(missing_ok=True)
    print(f"[done] {output_video} + {output_text.name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch transcription and one-word subtitle rendering for Desktop/factory")
    parser.add_argument("--factory", type=Path, default=DEFAULT_FACTORY)
    parser.add_argument("--model", default="small", choices=("tiny", "base", "small", "medium"))
    parser.add_argument("--language", default="ru")
    parser.add_argument("--limit", type=int, default=1, help="Files per category; default is safe test mode")
    parser.add_argument("--all", action="store_true", help="Process every video in all three categories")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preserve-text", action="store_true", help="Do not create or overwrite TXT files")
    args = parser.parse_args()

    factory = args.factory.expanduser().resolve()
    rendered = factory / "rendered"
    for category in CATEGORIES:
        source_dir = factory / category
        if not source_dir.is_dir():
            raise RuntimeError(f"Нет директории: {source_dir}")
        (rendered / category).mkdir(parents=True, exist_ok=True)
    if not FONT_PATH.is_file():
        raise RuntimeError(f"Не найден Comfortaa: {FONT_PATH}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe не найдены")

    selected: list[tuple[str, Path]] = []
    for category in CATEGORIES:
        files = sorted(
            (path for path in (factory / category).iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
            key=natural_key,
        )
        if not args.all:
            files = files[:max(0, args.limit)]
        selected.extend((category, path) for path in files)
    if not selected:
        print("Нет видео для обработки")
        return 0

    model_path = Path.home() / ".cache" / "whisper" / f"{args.model}.pt"
    print(f"[model] Whisper {args.model}: {model_path if model_path.exists() else 'будет скачан'}", flush=True)
    model = whisper.load_model(str(model_path) if model_path.exists() else args.model, device="cpu")
    failures: list[str] = []
    try:
        for index, (category, source) in enumerate(selected, 1):
            print(f"\n[{index}/{len(selected)}] {category}/{source.name}", flush=True)
            try:
                process_video(model, source, rendered / category, args.language, args.force, args.preserve_text)
            except Exception as exc:
                failures.append(f"{category}/{source.name}: {exc}")
                print(f"[error] {failures[-1]}", file=sys.stderr, flush=True)
    finally:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if failures:
        print("\nОшибки:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"\nГотово: {len(selected)} видео. Результаты: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
