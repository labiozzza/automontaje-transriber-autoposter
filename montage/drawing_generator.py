from __future__ import annotations

import json
import math
import re
from typing import Any

from .opencode_client import ask_json


_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?$")
_ALLOWED_VIZ_TYPES = {"chart", "process", "comparison", "hierarchy", "card"}
_CHART_TRENDS = {"up", "down", "flat", "wave"}


def _sanitize_label(value: Any, limit: int = 24) -> str:
    text = re.sub(r"[\x00-\x1f<>]", "", str(value or "").strip())
    return text[:limit]


def classify_request(text: str) -> str:
    """Определяет тип визуализации по тексту запроса (или явному префиксу)."""
    lower = (text or "").strip().lower()
    prefix_map = [
        ("график:", "chart"), ("диаграмм", "chart"), ("схема:", "abstract"),
        ("процесс:", "process"), ("карточка:", "card"), ("сравнение:", "comparison"),
        ("иерархия:", "hierarchy"), ("таблица:", "card"),
    ]
    for prefix, kind in prefix_map:
        if lower.startswith(prefix):
            return kind
    chart_words = ("график", "диаграм", "гистограмм", "столбчат", "ось", "оси", "абсцисс",
                   "ординат", "вертикал", "горизонтал", "координат", "тренд", "возрастающ",
                   "убывающ", "процент", "доли", "пирог")
    process_words = ("схема работы", "процесс", "этап", "шаг", "шаги", "последовательност",
                     "поток", "алгоритм", "как работает", "цикл", "стадия", "очередь")
    comparison_words = ("сравн", "до и после", "плюсы", "минусы", "за и против", "против",
                        "разница между", "vs", "лучше чем")
    hierarchy_words = ("иерархи", "дерево", "уровни", "вложенност", "структура компании", "подчинённост")
    card_words = ("карточка", "карточк", "таблица", "список", "статистика", "метрика", "цифра",
                  "число", "чек-лист", "чеклист", "шапка")
    rules = [
        ("chart", chart_words), ("process", process_words), ("comparison", comparison_words),
        ("hierarchy", hierarchy_words), ("card", card_words),
    ]
    for kind, words in rules:
        if any(word in lower for word in words):
            return kind
    return "scene"


def parse_chart_spec(text: str) -> dict[str, Any]:
    """Вынимает из запроса оси и тренд: «вертикали клиента горизонталь контент» → Y=Клиенты, X=Контент."""
    lower = (text or "").strip().lower()
    spec: dict[str, Any] = {"axes": {"x": "", "y": ""}, "trend": None, "kind": "line"}

    def axis_label(kind: str) -> str:
        starters = {
            "y": (r"вертикал\w*\s*(?:—|-|:|=\s*)?\s*", r"ось\s*y\s*[=:—]\s*", r"по\s+вертикал\w*\s*"),
            "x": (r"горизонтал\w*\s*(?:—|-|:|=\s*)?\s*", r"ось\s*x\s*[=:—]\s*", r"по\s+горизонтал\w*\s*"),
        }[kind]
        for starter in starters:
            match = re.search(starter, lower)
            if not match:
                continue
            rest = lower[match.end():]
            boundary = re.match(r"(.+?)(?:[;,.!?]| вертикал|\s+горизонтал|\s+ос[иь]|\s+график|$)", rest)
            label = (boundary.group(1) if boundary else rest).strip(" —:-")
            if label:
                return label
        return ""

    spec["axes"]["y"] = axis_label("y")
    spec["axes"]["x"] = axis_label("x")
    for trend, keywords in (
        ("up", (r"\bвозрастающ|\bрастёт|\bрастет|\bрастущ|\bрост|\bувеличива|\bповыша|\bвверх|\bв гору|\bвосходящ")),
        ("down", (r"\bубывающ|\bпадени|\bпада[ею]т?|\bснижа|\bуменьша|\bпонижа|\bвниз|\bнисходящ|\bспад")),
        ("flat", (r"\bровн|\bстабильн|\bплоск|\bнеизмен|\bбез изменений")),
        ("wave", (r"\bколебл|\bволн|\bскачк|\bзыгзаг|\bзигзаг")),
    ):
        if re.search(keywords, lower):
            spec["trend"] = trend
            break
    if re.search(r"гистограмм|столбчат|бар-|бары", lower):
        spec["kind"] = "bar"
    elif re.search(r"кругов|пирог|донут|доли |процент", lower):
        spec["kind"] = "donut"
    return spec


def chart_viz_for_request(text: str) -> dict[str, Any] | None:
    spec = parse_chart_spec(text)
    if not (spec["axes"]["x"] or spec["axes"]["y"] or spec["trend"]):
        return None
    return {
        "type": "chart",
        "axes": spec["axes"],
        "kind": spec["kind"],
        "trend": spec["trend"] or "up",
    }


def validate_drawing_overlay(raw: Any, index: int = 0) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Рисунок должен быть объектом")
    source = raw.get("drawing") if isinstance(raw.get("drawing"), dict) else raw
    viz = raw.get("viz") if isinstance(raw.get("viz"), dict) else {}
    if str(viz.get("type", "")).strip() in _ALLOWED_VIZ_TYPES:
        axes = viz.get("axes") if isinstance(viz.get("axes"), dict) else {}
        trend = str(viz.get("trend") or "").strip()
        normalized_viz = {
            "type": str(viz.get("type")).strip(),
            "axes": {
                "x": _sanitize_label(axes.get("x")),
                "y": _sanitize_label(axes.get("y")),
            },
            "kind": _sanitize_label(viz.get("kind") or "line", 20) or "line",
            "trend": trend if trend in _CHART_TRENDS else "up",
        }
    else:
        normalized_viz = None
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
    total_length = sum(
        math.hypot(points[position][0] - points[position - 1][0], points[position][1] - points[position - 1][1])
        for path in paths
        for points in [path["points"]]
        for position in range(1, len(points))
    )
    duration = end - start
    stroke_pause = min(0.03, duration * 0.12 / max(1, len(paths) - 1))
    default_draw_seconds = max(0.08, duration * 0.8 - max(0, len(paths) - 1) * stroke_pause)
    requested_speed = raw.get("draw_speed")
    draw_speed = float(requested_speed) if requested_speed is not None else total_length / default_draw_seconds
    drawing_payload = {
        "width": width,
        "height": height,
        "stroke_pause": round(stroke_pause, 4),
        "paths": paths,
    }
    if normalized_viz:
        drawing_payload["viz"] = normalized_viz
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
        "draw_speed": round(min(50000.0, max(10.0, draw_speed)), 2),
        "viz": normalized_viz,
        "drawing": drawing_payload,
    }


def generate_drawings(transcript: str, selected_fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Транскрипция пуста")
    targets: list[dict[str, Any]] = []
    for index, item in enumerate(selected_fragments[:12]):
        if not isinstance(item, dict):
            continue
        start = max(0.0, float(item.get("start") or 0))
        end = max(start + 0.05, float(item.get("end") or start + 1))
        text = str(item.get("text") or "").strip()[:500]
        if text:
            targets.append({
                "target_id": str(item.get("target_id") or index),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            })
    if not targets:
        raise ValueError("Не выбраны фрагменты для рисунков")

    request_text = " ".join(item["text"] for item in targets)
    intent = classify_request(request_text)
    targets_json = json.dumps(targets, ensure_ascii=False)
    context = transcript[:28_000]
    if intent == "chart":
        prompt = (
            "Ты готовишь ДЕКАРТОВСКИЙ ГРАФИК для короткого вертикального видео. Полная транскрипция дана только "
            "для контекста. Пользователь выбрал фрагменты: создай ровно ОДИН график для каждого TARGET и не создавай "
            "графики для остальных частей транскрипции. Не меняй target_id, start или end.\n"
            "Оси, стрелки и текстовые подписи осей нарисует система сама — рисовать их НЕ нужно. "
            "Рисуй ТОЛЬКО данные в координатах 0..1000.\n"
            "ГДЕ РАЗМЕЩАТЬ:\n"
            "- Нижняя ось X будет проведена по y=920 от x=120 до x=880, ось Y по x=120 от y=920 до y=80.\n"
            "- Область данных внутри [140..860] по X и [100..890] по Y.\n"
            "- Тренд up: линия монотонно идёт слева снизу вправо вверх; down — слева сверху вправо вниз; "
            "flat — почти горизонтально; wave — с мягкими колебаниями.\n"
            "- Линия: одна или две касающиеся между собой ломаные из 8-14 точек, уверенные штрихи.\n"
            "- kind=bar: отдельные вертикальные перекладины от нижней оси вверх.\n"
            "ЗАПРЕЩЕНО: люди, стикмены, руки, карточки, видеоиконки, окна, предметы, мысли-облака, HTML, SVG.\n\n"
            "Верни СТРОГО JSON без markdown по схеме:\n"
            '{"drawings":[{"target_id":"0","name":"...","trigger_text":"точный текст TARGET",'
            '"start":1.2,"end":5.2,"x":0.5,"y":0.35,"size":360,'
            '"viz":{"type":"chart","axes":{"x":"КОНТЕНТ","y":"КЛИЕНТЫ"},"kind":"line","trend":"up"},'
            '"drawing":{"width":1000,"height":1000,"paths":[{"points":[[140,880],[300,700],[500,520],[700,260]],'
            '"stroke":"#55D66B","stroke_width":14,"opacity":1,"closed":false}]}}]}\n\n'
            "В viz укажи подписи осей из смысла TARGET: ось Y (вертикаль) и ось X (горизонталь), "
            "kind и trend из текста. Подписи до 24 символов.\n\n"
            "TARGETS (рисовать только их):\n"
            + targets_json
            + "\n\nFULL_TRANSCRIPT_CONTEXT:\n"
            + context
        )
    else:
        intent_rules = {
            "process": "Р.S. ПОКАЖИ ПРОЦЕСС: последовательность из 2-5 блоков со стрелками между ними слева "
                       "направо, без текста внутри.\n",
            "comparison": "Р.S. ПОКАЖИ СРАВНЕНИЕ: две стороны слева и справа, разделённые вертикальной линией "
                          "или стрелкой, без текста внутри.\n",
            "hierarchy": "Р.S. ПОКАЖИ ИЕРАРХИЮ: 1-3 уровня, верхний элемент и связанные с ним элементы ниже, "
                         "без текста внутри.\n",
            "card": "Р.S. ПОКАЖИ КАРТОЧКУ: крупная рамка с заголовком и числом-строкой, без текста внутри.\n",
            "abstract": "Р.S. ПОКАЖИ АБСТРАКТНУЮ СХЕМУ: связи и зависимости стрелками и контурами, без текста внутри.\n",
        }
        prompt = (
            "Ты создаёшь простые схематические рисунки для короткого вертикального видео. Полная транскрипция дана "
            "только для понимания контекста. Пользователь уже сам выбрал фрагменты: создай ровно ОДИН рисунок для "
            "каждого TARGET и не создавай рисунки для остальных частей транскрипции. Не объединяй TARGET между собой, "
        "не меняй их target_id, start или end.\n\n"
            "СТИЛЬ:\n"
            "- Простая понятная схема маркером на прозрачном фоне, как объяснение на доске.\n"
            "- Человек — аккуратный стикмен: круглая голова, линия корпуса, простые руки и ноги. Эмоцию можно показать "
            "двумя точками глаз и линией рта. Не рисуй реалистичную анатомию.\n"
            "- Предметы изображай несколькими узнаваемыми контурами. Убирай декоративные детали, текст и фон.\n"
            "- В рисунке должен быть один ясный смысл. Используй стрелки и 2-3 линии движения только когда они "
            "помогают показать действие. Никаких странных абстрактных форм.\n"
            "- Если один герой встречается в нескольких TARGET, сохраняй того же стикмена: одинаковый размер головы, "
            "цвет и отличительный простой признак, например кепку или причёску.\n"
            "- Рисунок должен быть узнаваем за одну секунду и состоять из 6-20 длинных уверенных штрихов.\n\n"
            + intent_rules.get(intent, "")
            + "Верни СТРОГО JSON без markdown по схеме:\n"
            '{"drawings":[{"target_id":"0","name":"...","trigger_text":"точный текст TARGET",'
            '"start":1.2,"end":5.2,"x":0.5,"y":0.35,"size":360,'
            '"drawing":{"width":1000,"height":1000,"paths":[{"points":[[100,200],[120,180],[150,170]],'
            '"stroke":"#FFFFFF","stroke_width":10,"opacity":1,"closed":false}]}}]}\n\n'
            "ТЕХНИЧЕСКИЕ ПРАВИЛА: coordinates только 0..1000; 6-20 paths и 40-250 точек на рисунок; линии "
            "рисуются в естественном порядке; рисунок помещается в холст; никаких HTML, SVG, CSS, JS, URL или файлов.\n\n"
            "TARGETS (рисовать только их):\n"
            + targets_json
            + "\n\nFULL_TRANSCRIPT_CONTEXT:\n"
            + context
        )
    response = ask_json(prompt, model="opencode/big-pickle")
    raw_drawings = response.get("drawings")
    if not isinstance(raw_drawings, list):
        raise ValueError("BigPickle не вернул список рисунков")
    target_map = {item["target_id"]: item for item in targets}
    used_targets: set[str] = set()
    drawings: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_drawings[:24]):
        if not isinstance(raw, dict):
            continue
        target_id = str(raw.get("target_id", ""))
        target = target_map.get(target_id)
        if not target or target_id in used_targets:
            continue
        try:
            constrained = dict(raw)
            constrained.update({"start": target["start"], "end": target["end"], "trigger_text": target["text"]})
            if intent == "chart":
                forced_viz = chart_viz_for_request(request_text)
                if forced_viz:
                    constrained["viz"] = forced_viz
            drawing = validate_drawing_overlay(constrained, index)
            drawing["target_id"] = target_id
            drawings.append(drawing)
            used_targets.add(target_id)
        except (TypeError, ValueError, OverflowError):
            continue
    return drawings

_SRT_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _srt_timestamp_to_seconds(match: Any, offset: int) -> float:
    hours, minutes, seconds, millis = (
        int(match.group(offset + 1)),
        int(match.group(offset + 2)),
        int(match.group(offset + 3)),
        int(match.group(offset + 4)),
    )
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def parse_srt_segments(text: str) -> list[dict[str, Any]]:
    """Разбирает обычный SRT в сегменты с временем для автоподбора схем."""
    segments: list[dict[str, Any]] = []
    if not text or not text.strip():
        return segments
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        stamp_index = None
        match = None
        for index, line in enumerate(lines):
            found = _SRT_TIME_RE.search(line)
            if found:
                stamp_index = index
                match = found
                break
        if match is None or stamp_index is None:
            continue
        body = " ".join(lines[stamp_index + 1:]).strip()
        if not body:
            continue
        try:
            start = _srt_timestamp_to_seconds(match, 0)
            end = _srt_timestamp_to_seconds(match, 4)
        except (ValueError, AttributeError):
            continue
        if end < start:
            start, end = end, start
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": body[:400]})
    return segments


def generate_smart_drawings(transcript: str, segments: list[dict[str, Any]], max_drawings: int = 3) -> list[dict[str, Any]]:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Транскрипция пуста")
    if not segments:
        segments = parse_srt_segments(transcript)
    phrases = _segments_to_phrases(segments)
    context_phrases = phrases[:40]
    video_end = 0.0
    for item in segments:
        try:
            video_end = max(video_end, float(item.get("end") or 0))
        except (TypeError, ValueError):
            continue
    prompt = (
        "Ты — редактор схем для вертикального ролика. Твоя задача: выбрать 2–3 СМЫСЛОВЫХ момента в транскрипции "
        "и предложить простые объясняющие схемы (drawing) для каждого.\n"
        "Критерии: важный переход/сравнение/процесс/причина-следствие, моменты равномерно распределены по ролику, "
        "не предлагай одно слово, бери законченную мысль. Максимум 3 схемы.\n"
        "Рисуй в координатах 0..1000, 6-20 путей, 40-250 точек, простые формы, стрелки по делу. "
        "Для графиков рисуй ТОЛЬКО данные (оси нарисует система). Человек — аккуратный стикмен, без реалистичной анатомии.\n"
        "Верни СТРОГО JSON без markdown:\n"
        '{"suggestions":[{"text":"фраза","start":0.0,"end":2.5,"name":"название схемы",'
        '"rationale":"почему","drawing_type":"process","drawing":{"width":1000,"height":1000,'
        '"paths":[{"points":[[100,200],[200,150]],"stroke":"#FFFFFF","stroke_width":10}]}]}\n\n'
        "ФРАЗЫ:\n" + json.dumps(context_phrases, ensure_ascii=False) + "\n\nТРАНСКРИПЦИЯ (контекст):\n" + transcript[:20000]
    )
    response = ask_json(prompt, model="opencode/big-pickle")
    suggestions = response.get("suggestions")
    if not isinstance(suggestions, list):
        raise ValueError("BigPickle не вернул предложения")
    drawings: list[dict[str, Any]] = []
    used_intervals: list[tuple[float, float]] = []
    for idx, s in enumerate(suggestions[:max_drawings * 2]):
        if len(drawings) >= max_drawings:
            break
        if not isinstance(s, dict):
            continue
        text = str(s.get("text") or "").strip()[:500]
        if not text:
            continue
        try:
            start = float(s.get("start") or 0)
            end = float(s.get("end") or start + 0.8)
        except Exception:
            continue
        if start < 0:
            start = 0.0
        if end <= start:
            end = start + 1.5
        if video_end:
            if start >= video_end - 0.4:
                continue
            end = min(end, video_end)
        if end < start + 0.5:
            end = start + 0.5
        overlaps = any(start < used_end and end > used_start for used_start, used_end in used_intervals)
        if overlaps:
            continue
        d = s.get("drawing") or {}
        viz = s.get("viz")
        name = str(s.get("name") or "Схема")
        rationale = str(s.get("rationale") or "").strip()[:300]
        drawing_type = str(s.get("drawing_type") or "").strip()[:40]
        overlay = {
            "name": name,
            "start": round(max(0, start), 3),
            "end": round(max(start + 0.1, end), 3),
            "x": 0.5, "y": 0.35, "size": 360, "draw_speed": 900,
            "trigger_text": text,
        }
        if viz and isinstance(viz, dict):
            overlay["viz"] = viz
        if not isinstance(d, dict):
            d = {"width": 1000, "height": 1000, "paths": [{"points": [[100, 500], [900, 500]], "stroke": "#FFFFFF", "stroke_width": 10}]}
        overlay["drawing"] = d
        try:
            validated = validate_drawing_overlay(overlay, idx)
        except Exception:
            continue
        if rationale:
            validated["rationale"] = rationale
        if drawing_type:
            validated["drawing_type"] = drawing_type
        used_intervals.append((validated["start"], validated["end"]))
        drawings.append(validated)
    return drawings


def _segments_to_phrases(
    segments: list[dict[str, Any]],
    max_segments: int = 100_000,
    max_phrases: int = 60,
) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    current_text = ""
    current_start = 0.0
    current_end = 0.0
    for seg in segments[:max_segments]:
        try:
            s = float(seg.get("start") or 0)
            e = float(seg.get("end") or s)
        except Exception:
            continue
        t = str(seg.get("text") or "").strip()
        if not t:
            continue
        if not current_text:
            current_text, current_start, current_end = t, s, e
            continue
        if s - current_end > 2.5 or t.endswith(('.', '!', '?', '…')) or len(current_text) > 120:
            phrases.append({"text": current_text, "start": round(current_start, 3), "end": round(current_end, 3)})
            current_text, current_start, current_end = t, s, e
        else:
            current_text = (current_text + " " + t).strip()
            current_end = max(current_end, e)
    if current_text:
        phrases.append({"text": current_text, "start": round(current_start, 3), "end": round(current_end, 3)})
    if len(phrases) <= max_phrases:
        return phrases
    step = len(phrases) / max_phrases
    return [phrases[min(len(phrases) - 1, int(index * step))] for index in range(max_phrases)]


def revise_drawing(overlay: dict[str, Any], instruction: str, index: int = 0) -> dict[str, Any]:
    """До-рисовывает уже готовую схему: BigPickle меняет только содержимое поля drawing."""
    inner = overlay.get("drawing") if isinstance(overlay.get("drawing"), dict) else {}
    base = {
        "name": str(overlay.get("name") or "Схема"),
        "trigger_text": str(overlay.get("trigger_text") or ""),
        "start": float(overlay.get("start") or 0.0),
        "end": float(overlay.get("end") or 1.0),
        "x": float(overlay.get("x") if overlay.get("x") is not None else 0.5),
        "y": float(overlay.get("y") if overlay.get("y") is not None else 0.35),
        "size": int(overlay.get("size") or 360),
        "draw_speed": float(overlay.get("draw_speed") or 350),
    }
    viz = inner.get("viz") if isinstance(inner.get("viz"), dict) else (overlay.get("viz") if isinstance(overlay.get("viz"), dict) else None)
    if viz:
        base["viz"] = viz
    base["drawing"] = inner
    prompt = (
        "Ты дорисовываешь ОДНУ уже готовую схему для вертикального видео. Измени только содержимое поля drawing "
        "по инструкции пользователя. Фон не нужен, стиль «рисунок маркером», как объяснение на доске, линии "
        "рисуются в естественном порядке.\n"
        "СОХРАНИ без изменений: width, height, а если был виз-блок viz — его подписи осей, kind и trend.\n"
        "ИНСТРУКЦИЯ ПОЛЬЗОВАТЕЛЯ:\n"
        + (instruction or "")[:2000]
        + "\n\nТЕКУЩАЯ СХЕМА (JSON):\n"
        + json.dumps(base, ensure_ascii=False)[:30_000]
        + "\n\nТЕХНИЧЕСКИЕ ПРАВИЛА: coordinates только 0..1000; 6-20 paths и 40-250 точек; линии в естественном "
        "порядке; рисунок помещается в холст; никаких HTML, SVG, CSS, JS, URL или файлов. Верни СТРОГО JSON без "
        "markdown: тот же объект, но с новым полем drawing в прежней схеме: {\"drawing\":{\"width\":1000,\"height\":1000,"
        '"paths":[{"points":[[100,200],[120,180],[150,170]],"stroke":"#FFFFFF","stroke_width":10,"opacity":1,"closed":false}]}}'
    )
    response = ask_json(prompt, model="opencode/big-pickle")
    new_inner = response.get("drawing") if isinstance(response.get("drawing"), dict) else response
    if not isinstance(new_inner, dict) or not isinstance(new_inner.get("paths"), list):
        raise ValueError("BigPickle не вернул содержимое изменённой схемы")
    base["drawing"] = new_inner
    return validate_drawing_overlay(base, index)
