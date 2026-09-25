from __future__ import annotations

import json
import math
import re
from typing import Any

from .opencode_client import ask_json


_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?$")


def validate_drawing_overlay(raw: Any, index: int = 0) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Рисунок должен быть объектом")
    source = raw.get("drawing") if isinstance(raw.get("drawing"), dict) else raw
    width = min(2000, max(100, int(source.get("width") or 1000)))
    height = min(2000, max(100, int(source.get("height") or 1000)))
    raw_paths = source.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths or len(raw_paths) > 80:
        raise ValueError("Рисунок должен содержать от 1 до 80 линий")
    paths: list[dict[str, Any]] = []
    total_points = 0
    for raw_path in raw_paths:
        if not isinstance(raw_path, dict) or not isinstance(raw_path.get("points"), list):
            raise ValueError("Линия рисунка имеет неверный формат")
        points: list[list[float]] = []
        for point in raw_path["points"][:400]:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append([round(min(width, max(0.0, x)), 3), round(min(height, max(0.0, y)), 3)])
        if len(points) < 2:
            continue
        total_points += len(points)
        if total_points > 4000:
            raise ValueError("В рисунке слишком много точек")
        stroke = str(raw_path.get("stroke") or "#FFFFFF")
        if not _COLOR_RE.fullmatch(stroke):
            stroke = "#FFFFFF"
        paths.append({
            "points": points,
            "stroke": stroke.upper(),
            "stroke_width": round(min(40.0, max(1.0, float(raw_path.get("stroke_width") or 8))), 2),
            "opacity": round(min(1.0, max(0.1, float(raw_path.get("opacity") or 1))), 3),
            "closed": bool(raw_path.get("closed", False)),
        })
    if not paths:
        raise ValueError("GPT не создал пригодных линий")
    start = max(0.0, min(86399.0, float(raw.get("start") or 0)))
    end = max(start + 0.5, min(86400.0, float(raw.get("end") or start + 4)))
    return {
        "uid": str(raw.get("uid") or f"drawing-{index + 1}")[:80],
        "kind": "drawing",
        "name": str(raw.get("name") or raw.get("trigger_text") or f"Рисунок {index + 1}")[:100],
        "trigger_text": str(raw.get("trigger_text") or "")[:240],
        "start": round(start, 3),
        "end": round(end, 3),
        "x": round(min(1.0, max(0.0, float(raw.get("x", 0.5)))), 4),
        "y": round(min(1.0, max(0.0, float(raw.get("y", 0.35)))), 4),
        "size": min(1200, max(80, int(raw.get("size") or 360))),
        "draw_speed": round(min(3000.0, max(10.0, float(raw.get("draw_speed") or 350))), 2),
        "drawing": {"width": width, "height": height, "paths": paths},
    }


def generate_drawings(transcript: str) -> list[dict[str, Any]]:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Транскрипция пуста")
    prompt = (
        "Ты художник-раскадровщик короткого вертикального видео. Проанализируй SRT-транскрипцию "
        "и предложи 1-6 уместных схематичных детских рисунков, которые визуально поясняют сказанное. "
        "Не используй каталог готовых объектов: каждый рисунок придумай и построй сам свободными линиями. "
        "Например для фразы 'собака кушает корм' самостоятельно нарисуй контур собаки, наклонённую голову, "
        "миску и корм. Рисунок должен читаться без фотореализма, на прозрачном фоне, без текста.\n\n"
        "Верни СТРОГО JSON без markdown по схеме:\n"
        '{"drawings":[{"name":"...","trigger_text":"точная фраза","start":1.2,"end":5.2,'
        '"x":0.5,"y":0.35,"size":360,"draw_speed":350,"drawing":{"width":1000,"height":1000,'
        '"paths":[{"points":[[100,200],[120,180],[150,170]],"stroke":"#FFFFFF",'
        '"stroke_width":10,"opacity":1,"closed":false}]}}]}\n\n'
        "Правила: coordinates только 0..1000; каждый контур состоит из достаточно подробной ломаной; "
        "используй много отдельных линий в естественном порядке рисования; максимум 60 линий и 2000 точек "
        "на сцену; start/end бери из SRT; рисунок должен помещаться в холст; никаких HTML, SVG, CSS, JS, URL "
        "или файлов; не предлагай рисунок, если он не помогает пониманию.\n\nSRT:\n"
        + transcript[:35_000]
    )
    response = ask_json(prompt)
    raw_drawings = response.get("drawings")
    if not isinstance(raw_drawings, list):
        raise ValueError("GPT не вернул список рисунков")
    drawings: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_drawings[:6]):
        try:
            drawings.append(validate_drawing_overlay(raw, index))
        except (TypeError, ValueError, OverflowError):
            continue
    if not drawings:
        raise ValueError("GPT не предложил пригодных рисунков")
    return drawings
