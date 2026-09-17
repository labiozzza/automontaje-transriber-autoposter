from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


OPENCODE = os.environ.get("OPENCODE_BIN", "").strip() or shutil.which("opencode") or "opencode"
MODEL = os.environ.get("COVER_MODEL", "opencode/big-pickle")
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


def _cover_titles(path: Path, transcript: str, count: int) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        candidates = data.get("titles", []) if isinstance(data, dict) else data
    except (OSError, json.JSONDecodeError):
        candidates = []
    if not isinstance(candidates, list):
        candidates = []
    titles = [str(item).strip() for item in candidates if str(item).strip()][:count]
    words = transcript.split()
    while len(titles) < count:
        start = len(words) * len(titles) // max(1, count)
        fallback = " ".join(words[start:start + 5]).strip(" .,!?:;—-") or "Новый взгляд"
        titles.append(fallback)
    return titles


def _run_agent(prompt: str, workdir: Path, log_path: Path, expected: list[Path], progress: Callable[[float, str], None]) -> None:
    command = [
        OPENCODE, "run", prompt, "-m", MODEL, "--auto", "--format", "json", "--dir", str(workdir),
    ]
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        last_count = -1
        deadline = time.monotonic() + float(os.environ.get("COVER_GENERATION_TIMEOUT", "1200"))
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TimeoutError("Генерация обложек превысила лимит времени")
            count = sum(path.exists() for path in expected)
            if count != last_count:
                last_count = count
                progress(15 + count / max(1, len(expected)) * 80, f"Создано обложек: {count}/{len(expected)}")
            time.sleep(2)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Big Pickle завершился с кодом {code}; см. {log_path.name}")


def _manifest_items(covers_dir: Path) -> list[dict[str, str]]:
    manifest = covers_dir / "manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in data.get("covers", []) if isinstance(item, dict) and (covers_dir / Path(str(item.get("file") or "")).name).exists()]


def _save_manifest(covers_dir: Path, items: list[dict[str, str]]) -> None:
    (covers_dir / "manifest.json").write_text(json.dumps({"covers": items}, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_suggested_covers(job_id: str, video: Path, transcript: str, covers_dir: Path, progress: Callable[[float, str], None]) -> list[dict[str, str]]:
    covers_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = covers_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    duration = _duration(video)
    rng = random.Random(job_id)
    low, high = min(1.0, duration * 0.1), max(min(1.0, duration * 0.1), duration * 0.9)
    timestamps = sorted(rng.uniform(low, high) for _ in range(4))
    frames = [extract_frame(video, timestamp, frames_dir / f"frame_{index + 1:02d}.jpg") for index, timestamp in enumerate(timestamps)]
    progress(12, "Стоп-кадры подготовлены")
    outputs = [covers_dir / f"suggested_{index + 1:02d}.png" for index in range(4)]
    titles_path = covers_dir / "suggested_titles.json"
    calls = "\n".join(
        f"{index + 1}. images=['{frame}']; out='{output}'; quality='medium'; size='1088x1920'."
        for index, (frame, output) in enumerate(zip(frames, outputs))
    )
    prompt = f"""Ты создаёшь четыре разные обложки для Instagram Reels. Используй инструмент gpt_imagegen ровно четыре раза, один раз для каждого пункта ниже. Каждый исходный кадр передавай через images как Image 1.

ТРЕБОВАНИЯ К КАЖДОЙ ОБЛОЖКЕ:
- это вертикальная обложка Instagram Reels, рассчитанная на просмотр в ленте телефона;
- сохрани человека, предметы и узнаваемость исходного кадра;
- визуальный стиль спокойный и сдержанный: естественный свет, умеренный контраст, приглушённые натуральные цвета;
- исключи неон, чрезмерную насыщенность, пересветы, агрессивный HDR и кричащие цветовые акценты;
- НЕ РИСУЙ на изображении текст, буквы, логотипы или watermark: приложение добавит заголовок само;
- придумай для каждого варианта отдельный короткий заголовок на русском из 3–5 слов, без CAPS LOCK и лишних восклицательных знаков;
- до завершения сохрани точные четыре заголовка в UTF-8 JSON-файл `{titles_path}` в формате {{"titles": ["...", "...", "...", "..."]}};
- в prompt каждого gpt_imagegen явно повтори требования к спокойному фону без любого текста.

TOOL-ВЫЗОВЫ И ФАЙЛЫ:
{calls}

ПОЛНАЯ ТРАНСКРИБАЦИЯ РОЛИКА:
---
{transcript}
---

Не останавливайся после первого изображения. Создай все четыре файла. После четырёх tool-вызовов ответь кратко."""
    _run_agent(prompt, covers_dir, covers_dir / "generation.log", outputs, progress)
    existing = [path for path in outputs if path.exists()]
    if not existing:
        raise RuntimeError("Big Pickle не создал ни одной обложки")
    titles = _cover_titles(titles_path, transcript, len(existing))
    for path, title in zip(existing, titles):
        _finish_generated_cover(path, title)
    items = _manifest_items(covers_dir)
    known = {item.get("file") for item in items}
    for index, path in enumerate(existing):
        if path.name not in known:
            items.append({"id": path.stem, "file": path.name, "name": titles[index], "source": "generated"})
    _save_manifest(covers_dir, items)
    progress(100, f"Обложки готовы: {len(existing)}")
    return items


def generate_custom_cover(video: Path, timestamp: float, title: str, covers_dir: Path, progress: Callable[[float, str], None]) -> list[dict[str, str]]:
    covers_dir.mkdir(parents=True, exist_ok=True)
    cover_id = f"custom_{uuid.uuid4().hex[:8]}"
    frame = extract_frame(video, timestamp, covers_dir / "frames" / f"{cover_id}.jpg")
    output = covers_dir / f"{cover_id}.png"
    progress(15, "Стоп-кадр подготовлен")
    prompt = f"""Вызови gpt_imagegen ровно один раз. Image 1 — стоп-кадр ролика. Создай вертикальный фон обложки Instagram Reels, сохрани узнаваемость кадра. Стиль спокойный и сдержанный: естественный свет, умеренный контраст, приглушённые натуральные цвета. Исключи неон, чрезмерную насыщенность, пересветы и агрессивный HDR. НЕ РИСУЙ текст, буквы, логотипы или watermark: приложение добавит заголовок само. Параметры: images=['{frame}']; out='{output}'; quality='medium'; size='1088x1920'."""
    _run_agent(prompt, covers_dir, covers_dir / f"{cover_id}.log", [output], progress)
    if not output.exists():
        raise RuntimeError("Обложка не создана")
    _finish_generated_cover(output, title)
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
