from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


COVER_SIZE = (1080, 1920)
FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Comfortaa.ttf"


def _duration(video: Path) -> float:
    result = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError("Не удалось определить длительность видео для обложек")
    return max(0.1, float(result.stdout.strip()))


def extract_frame(video: Path, timestamp: float, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg", "-y", "-ss", f"{max(0, timestamp):.3f}",
            "-i", str(video), "-frames:v", "1", "-q:v", "2", str(output),
        ],
        capture_output=True, timeout=180,
    )
    if result.returncode != 0 or not output.exists():
        raise RuntimeError("Не удалось извлечь стоп-кадр для обложки")
    return output


def _normalize_cover(path: Path) -> None:
    with Image.open(path) as source:
        image = ImageOps.fit(source.convert("RGB"), COVER_SIZE, method=Image.Resampling.LANCZOS)
        temporary = path.with_name(f".{path.stem}-normalized.png")
        image.save(temporary, "PNG", optimize=True)
    temporary.replace(path)


def _wrap_title(draw: ImageDraw.ImageDraw, title: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    lines: list[str] = []
    current = ""
    for word in title.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font, stroke_width=5)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _title_font(size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONT_PATH), size)
    font.set_variation_by_name("Bold")
    return font


def _finish_generated_cover(path: Path, title: str) -> None:
    with Image.open(path) as source:
        image = ImageOps.fit(source.convert("RGB"), COVER_SIZE, method=Image.Resampling.LANCZOS)
        image = ImageEnhance.Color(image).enhance(0.68)
        image = ImageEnhance.Brightness(image).enhance(0.88)
        image = ImageEnhance.Contrast(image).enhance(0.95)
        draw = ImageDraw.Draw(image, "RGBA")

        max_width = 880
        chosen_font = _title_font(64)
        wrapped = title.strip()
        for size in range(156, 63, -4):
            font = _title_font(size)
            candidate = _wrap_title(draw, title.strip(), font, max_width)
            bbox = draw.multiline_textbbox((0, 0), candidate, font=font, spacing=18, align="center", stroke_width=5)
            if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= 620:
                chosen_font = font
                wrapped = candidate
                break

        bbox = draw.multiline_textbbox((0, 0), wrapped, font=chosen_font, spacing=18, align="center", stroke_width=5)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (COVER_SIZE[0] - text_width) / 2 - bbox[0]
        y = (COVER_SIZE[1] - text_height) / 2 - bbox[1]
        padding_x, padding_y = 48, 36
        draw.rounded_rectangle(
            (
                x + bbox[0] - padding_x,
                y + bbox[1] - padding_y,
                x + bbox[2] + padding_x,
                y + bbox[3] + padding_y,
            ),
            radius=30,
            fill=(0, 0, 0, 112),
        )
        draw.multiline_text(
            (x, y), wrapped, font=chosen_font, fill=(245, 243, 238, 255),
            spacing=18, align="center", stroke_width=5, stroke_fill=(0, 0, 0, 205),
        )
        temporary = path.with_name(f".{path.stem}-finished.png")
        image.save(temporary, "PNG", optimize=True)
    temporary.replace(path)


def _auto_titles(transcript: str, count: int) -> list[str]:
    lines = []
    for raw in transcript.splitlines():
        line = re.sub(r"<[^>]+>", " ", raw).strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        line = re.sub(r"^[A-ZА-ЯЁ][A-ZА-ЯЁ0-9 _-]{1,24}:\s*", "", line)
        lines.append(line)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    chunks = [chunk.strip(" .,!?:;—-\"'«»") for chunk in re.split(r"[.!?;]+", text) if chunk.strip()]
    phrases = []
    for chunk in chunks:
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", chunk)
        lowered = [word.lower() for word in words]
        for prefix in (("сегодня", "мы"), ("сейчас", "мы"), ("в", "этом", "видео"), ("в", "этом", "ролике"), ("давайте",)):
            if lowered[:len(prefix)] == list(prefix):
                words = words[len(prefix):]
                break
        if len(words) >= 3:
            phrases.append(" ".join(words[:5]))
    if not phrases:
        phrases = ["главное в этом ролике"]

    hooks = (
        "Главное: {phrase}",
        "Вот где ошибка: {phrase}",
        "Это меняет всё: {phrase}",
        "Об этом обычно молчат: {phrase}",
    )
    titles = []
    for index in range(count):
        phrase = phrases[min(len(phrases) - 1, index * len(phrases) // count)].lower()
        title = hooks[index % len(hooks)].format(phrase=phrase)
        titles.append(title[:100].rstrip(" .,!?:;—-"))
    return titles


def _bigpickle_titles(transcript: str, count: int) -> list[str]:
    from .generate_configs_from_transcript_srt import _extract_text_from_ndjson, _run_opencode, extract_json_object

    prompt = f"""Придумай ровно {count} разных кликбейтных заголовка для обложек короткого вертикального видео по транскрипции ниже.

Требования:
- русский язык;
- каждый заголовок содержит 3-7 слов и не длиннее 100 символов;
- передай смысл конкретно этого ролика, не используй универсальные пустые фразы;
- варианты должны отражать разные сильные мысли из разных частей ролика;
- без CAPS LOCK, эмодзи, кавычек, точки в конце и лишних восклицательных знаков;
- не вызывай инструменты, не создавай изображения и файлы;
- ответь только JSON-объектом: {{"titles":["...", "...", "...", "..."]}}.

ТРАНСКРИПЦИЯ:
---
{transcript}
---"""
    raw = _run_opencode(prompt)
    payload = extract_json_object(_extract_text_from_ndjson(raw)) or extract_json_object(raw)
    candidates = payload.get("titles") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        raise RuntimeError("BigPickle не вернул список заголовков")
    titles = [str(item).strip() for item in candidates if str(item).strip()]
    if len(titles) != count or any(len(title) > 100 for title in titles):
        raise RuntimeError("BigPickle вернул некорректные заголовки")
    return titles


def _make_local_cover(frame: Path, output: Path, title: str) -> None:
    with Image.open(frame) as source:
        source.convert("RGB").save(output, "PNG")
    _finish_generated_cover(output, title)


def _manifest_items(covers_dir: Path) -> list[dict[str, str]]:
    manifest = covers_dir / "manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in data.get("covers", []) if isinstance(item, dict) and (covers_dir / Path(str(item.get("file") or "")).name).exists()]


def _save_manifest(covers_dir: Path, items: list[dict[str, str]]) -> None:
    (covers_dir / "manifest.json").write_text(json.dumps({"covers": items}, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_suggested_covers(
    job_id: str,
    video: Path,
    transcript: str,
    covers_dir: Path,
    progress: Callable[[float, str], None],
    manual_title: str = "",
) -> list[dict[str, str]]:
    covers_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = covers_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    duration = _duration(video)
    rng = random.Random(job_id)
    low, high = min(1.0, duration * 0.1), max(min(1.0, duration * 0.1), duration * 0.9)
    count = 5 if manual_title.strip() else 4
    progress(3, "BigPickle придумывает заголовки")
    try:
        titles = _bigpickle_titles(transcript, 4)
    except Exception as exc:
        (covers_dir / "title_generation_error.log").write_text(str(exc), encoding="utf-8")
        titles = _auto_titles(transcript, 4)
        progress(8, "BigPickle недоступен, использованы локальные заголовки")
    timestamps = sorted(rng.uniform(low, high) for _ in range(count))
    frames = [extract_frame(video, timestamp, frames_dir / f"frame_{index + 1:02d}.jpg") for index, timestamp in enumerate(timestamps)]
    progress(12, "Стоп-кадры подготовлены")
    outputs = [covers_dir / f"suggested_{index + 1:02d}.png" for index in range(count)]
    if manual_title.strip():
        titles.append(manual_title.strip())
    for index, (frame, output, title) in enumerate(zip(frames, outputs, titles)):
        _make_local_cover(frame, output, title)
        progress(15 + (index + 1) / count * 80, f"Создано обложек: {index + 1}/{count}")
    items = _manifest_items(covers_dir)
    known = {item.get("file") for item in items}
    for index, path in enumerate(outputs):
        if path.name not in known:
            source = "manual" if manual_title.strip() and index == 4 else "generated"
            items.append({"id": path.stem, "file": path.name, "name": titles[index], "source": source})
    _save_manifest(covers_dir, items)
    progress(100, f"Обложки готовы: {len(outputs)}")
    return items


def generate_custom_cover(video: Path, timestamp: float, title: str, covers_dir: Path, progress: Callable[[float, str], None]) -> list[dict[str, str]]:
    covers_dir.mkdir(parents=True, exist_ok=True)
    cover_id = f"custom_{uuid.uuid4().hex[:8]}"
    frame = extract_frame(video, timestamp, covers_dir / "frames" / f"{cover_id}.jpg")
    output = covers_dir / f"{cover_id}.png"
    progress(15, "Стоп-кадр подготовлен")
    _make_local_cover(frame, output, title)
    items = _manifest_items(covers_dir)
    items.append({"id": cover_id, "file": output.name, "name": title, "source": "custom"})
    _save_manifest(covers_dir, items)
    progress(100, "Обложка готова")
    return items


def save_uploaded_cover(source: Path, covers_dir: Path) -> list[dict[str, str]]:
    covers_dir.mkdir(parents=True, exist_ok=True)
    cover_id = f"upload_{uuid.uuid4().hex[:8]}"
    output = covers_dir / f"{cover_id}.png"
    shutil.copy2(source, output)
    _normalize_cover(output)
    items = _manifest_items(covers_dir)
    items.append({"id": cover_id, "file": output.name, "name": "Своя обложка", "source": "upload"})
    _save_manifest(covers_dir, items)
    return items
