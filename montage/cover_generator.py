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

from PIL import Image, ImageOps


OPENCODE = os.environ.get("OPENCODE_BIN", "").strip() or shutil.which("opencode") or "opencode"
MODEL = os.environ.get("COVER_MODEL", "opencode/big-pickle")
COVER_SIZE = (1080, 1920)


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
    calls = "\n".join(
        f"{index + 1}. images=['{frame}']; out='{output}'; quality='medium'; size='1088x1920'."
        for index, (frame, output) in enumerate(zip(frames, outputs))
    )
    prompt = f"""Ты создаёшь четыре разные обложки для Instagram Reels. Используй инструмент gpt_imagegen ровно четыре раза, один раз для каждого пункта ниже. Каждый исходный кадр передавай через images как Image 1.

ТРЕБОВАНИЯ К КАЖДОЙ ОБЛОЖКЕ:
- это вертикальная обложка Instagram Reels, рассчитанная на просмотр в ленте телефона;
- сохрани человека, предметы и узнаваемость исходного кадра, но сделай изображение контрастным и кликабельным;
- придумай отдельный короткий кликбейтный заголовок на русском на основании полной транскрибации ниже;
- заголовок должен быть крупным, читаемым, РОВНО ПО ЦЕНТРУ изображения;
- шрифт визуально должен быть Comfortaa Bold; не размещай никакой другой текст, логотипы или watermark;
- текст должен помещаться в safe-zone и хорошо читаться в маленькой карточке ленты Instagram;
- в prompt каждого gpt_imagegen явно напиши точный выбранный заголовок и все требования выше.

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
    for path in existing:
        _normalize_cover(path)
    items = _manifest_items(covers_dir)
    known = {item.get("file") for item in items}
    for index, path in enumerate(existing):
        if path.name not in known:
            items.append({"id": path.stem, "file": path.name, "name": f"Вариант {index + 1}", "source": "generated"})
    _save_manifest(covers_dir, items)
    progress(100, f"Обложки готовы: {len(existing)}")
    return items


def generate_custom_cover(video: Path, timestamp: float, title: str, covers_dir: Path, progress: Callable[[float, str], None]) -> list[dict[str, str]]:
    covers_dir.mkdir(parents=True, exist_ok=True)
    cover_id = f"custom_{uuid.uuid4().hex[:8]}"
    frame = extract_frame(video, timestamp, covers_dir / "frames" / f"{cover_id}.jpg")
    output = covers_dir / f"{cover_id}.png"
    progress(15, "Стоп-кадр подготовлен")
    prompt = f"""Вызови gpt_imagegen ровно один раз. Image 1 — стоп-кадр ролика. Создай вертикальную обложку Instagram Reels: контрастную и заметную в ленте, сохрани узнаваемость кадра. Наложи ТОЧНО этот текст без изменений: «{title}». Текст крупный, шрифт Comfortaa Bold, расположен РОВНО ПО ЦЕНТРУ, в safe-zone. Никакого другого текста, логотипов и watermark. Параметры: images=['{frame}']; out='{output}'; quality='medium'; size='1088x1920'."""
    _run_agent(prompt, covers_dir, covers_dir / f"{cover_id}.log", [output], progress)
    if not output.exists():
        raise RuntimeError("Обложка не создана")
    _normalize_cover(output)
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
