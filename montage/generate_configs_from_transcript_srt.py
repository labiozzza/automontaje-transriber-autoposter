#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate zoom/subtitle JSON configs from an SRT transcript.

Input:
    python generate_configs_from_transcript_srt.py path/to/subtitles_words.srt

Output near the SRT file:
    zoom_timeline_config.json
    subtitle_timeline_config.json              # sentence subtitles
    subtitle_timeline_config_words.json        # word-by-word subtitles

The script DOES NOT read or require the video file. The default render.video paths
are only written into JSON because the existing render scripts read them later.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPENCODE_BIN = os.getenv("OPENCODE_BIN", "").strip() or shutil.which("opencode") or "opencode"
BIGPICKLE_MODEL = os.getenv("BIGPICKLE_MODEL", "opencode/big-pickle").strip()
USE_BIGPICKLE = os.getenv("USE_BIGPICKLE", "1").strip().lower() not in {"0", "false", "no", "off"}
BIGPICKLE_TIMEOUT_SECONDS = float(os.getenv("BIGPICKLE_TIMEOUT_SECONDS", "180"))

# These are not inputs. They are default paths for the next render steps.
DEFAULT_SOURCE_VIDEO = "input/source.mp4"
DEFAULT_ZOOM_OUTPUT = "output/timeline_zoom_strong.mp4"
DEFAULT_SUBTITLE_OUTPUT = "output/timeline_zoom_subtitles.mp4"
DEFAULT_SUBTITLE_WORDS_OUTPUT = "output/timeline_zoom_subtitles_words.mp4"


@dataclass
class Cue:
    start: float
    end: float
    text: str


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def seconds_to_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_srt_time(value: str) -> float:
    raw = value.strip().replace(",", ".")
    match = re.match(r"^(\d+):(\d+):(\d+)(?:\.(\d+))?$", raw)
    if not match:
        fail(f"Bad SRT time: {value!r}")
    h, m, s, ms = match.groups()
    ms = (ms or "0")[:3].ljust(3, "0")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[Cue]:
    if not path.exists():
        fail(f"SRT not found: {path}")

    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues: list[Cue] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        time_line_index = None
        for i, line in enumerate(lines[:3]):
            if "-->" in line:
                time_line_index = i
                break
        if time_line_index is None:
            continue

        time_line = lines[time_line_index]
        left, right = [x.strip() for x in time_line.split("-->", 1)]
        start = parse_srt_time(left)
        end = parse_srt_time(right.split()[0])
        cue_text = " ".join(lines[time_line_index + 1:]).strip()
        cue_text = re.sub(r"<[^>]+>", "", cue_text)
        cue_text = re.sub(r"\s+", " ", cue_text).strip()

        if cue_text and math.isfinite(start) and math.isfinite(end) and end > start:
            cues.append(Cue(start=start, end=end, text=cue_text))

    cues.sort(key=lambda c: (c.start, c.end))
    if not cues:
        fail("No valid cues found in SRT")
    return cues


def split_cue_to_word_cues(cues: list[Cue]) -> list[Cue]:
    """If the input is already word-level, it stays mostly unchanged.
    If a cue contains several words, spread them evenly across the cue time.
    """
    out: list[Cue] = []
    for cue in cues:
        words = [w for w in cue.text.split() if w.strip()]
        if not words:
            continue
        if len(words) == 1:
            out.append(cue)
            continue
        duration = max(0.001, cue.end - cue.start)
        step = duration / len(words)
        for i, word in enumerate(words):
            out.append(Cue(start=cue.start + i * step, end=cue.start + (i + 1) * step, text=word))
    return out


def normalize_word(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"^[^a-zа-я0-9%]+|[^a-zа-я0-9%]+$", "", value, flags=re.I)
    return value


def normalize_phrase(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^a-zа-я0-9%]+", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def group_sentences(words: list[Cue]) -> list[list[Cue]]:
    groups: list[list[Cue]] = []
    current: list[Cue] = []

    max_duration = 3.8
    max_chars = 56
    max_words = 10
    hard_gap = 0.75

    for word in words:
        if current and word.start - current[-1].end >= hard_gap:
            groups.append(current)
            current = []

        current.append(word)
        text = " ".join(w.text for w in current)
        duration = current[-1].end - current[0].start
        ends_sentence = bool(re.search(r"[.!?…]$", word.text.strip()))

        if ends_sentence or duration >= max_duration or len(text) >= max_chars or len(current) >= max_words:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    # Merge too-short orphan groups into previous group when possible.
    merged: list[list[Cue]] = []
    for group in groups:
        if merged and len(group) <= 2:
            prev_text = " ".join(w.text for w in merged[-1] + group)
            prev_duration = group[-1].end - merged[-1][0].start
            if len(prev_text) <= max_chars and prev_duration <= max_duration + 1.2:
                merged[-1].extend(group)
                continue
        merged.append(group)
    return merged


def group_phrases(words: list[Cue]) -> list[list[Cue]]:
    """Group consecutive words into short, readable chunks of 2-5 words."""
    groups: list[list[Cue]] = []
    index = 0
    while index < len(words):
        remaining = len(words) - index
        size = min(5, remaining)
        if remaining - size == 1:
            size -= 1
        if size <= 1 and groups:
            groups[-1].append(words[index])
            break
        groups.append(words[index:index + max(1, size)])
        index += max(1, size)
    return groups


def _run_opencode(prompt: str, timeout: float = BIGPICKLE_TIMEOUT_SECONDS) -> str:
    cmd = [
        OPENCODE_BIN, "run", "-m", BIGPICKLE_MODEL, "--format", "json", prompt,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"opencode exit {result.returncode}: {result.stderr.strip()[-500:]}")
    return result.stdout


def _extract_text_from_ndjson(raw: str) -> str:
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") in {"text", "part"}:
            part = ev.get("part")
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(part, str) and part:
                parts.append(part)
        elif ev.get("type") == "assistant_text":
            text = ev.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    # Remove common markdown fences if the model adds them.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first:last + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return None


def ask_bigpickle(sentences: list[list[Cue]]) -> dict[str, Any] | None:
    if not USE_BIGPICKLE:
        return None

    compact = []
    for group in sentences[:140]:
        compact.append({
            "start": round(group[0].start, 3),
            "end": round(group[-1].end, 3),
            "text": " ".join(w.text for w in group),
        })

    prompt = (
        "Ты анализируешь русскую транскрибацию короткого вертикального видео. "
        "Нужно подготовить подсветку слов и события автозума.\n\n"
        "Верни СТРОГО JSON без markdown и без комментариев. Схема:\n"
        "{\n"
        '  "green": ["позитивные или выгодные слова/фразы"],\n'
        '  "yellow": ["важные смысловые слова/фразы, цифры, термины"],\n'
        '  "red": ["негативные, опасные, конфликтные слова/фразы"],\n'
        '  "zoom": [{"time": 0.0, "percent": 0}, {"time": 2.4, "percent": 40}]\n'
        "}\n\n"
        "Правила:\n"
        "- green/yellow/red: короткие слова или фразы из исходного текста, не выдумывай новые.\n"
        "- red важнее green, green важнее yellow.\n"
        "- zoom: 8-24 события на весь ролик, не чаще одного раза в 1.8 секунды.\n"
        "- percent только 0, 30, 40, 50, 60, 70.\n"
        "- на сильных тезисах ставь 50-70, на обычных переходах 30-40.\n\n"
        "Сегменты:\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )

    try:
        raw_stdout = _run_opencode(prompt)
    except Exception as e:
        print(f"WARN: BigPickle unavailable or failed: {e}", file=sys.stderr)
        return None

    assistant_text = _extract_text_from_ndjson(raw_stdout)
    parsed = extract_json_object(assistant_text) if assistant_text else None
    if parsed is None:
        print("WARN: BigPickle returned invalid JSON. Fallback to local heuristics.", file=sys.stderr)
    return parsed


def list_from_lm(data: dict[str, Any] | None, key: str) -> list[str]:
    if not isinstance(data, dict):
        return []
    value = data.get(key)
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out[:80]


def build_rules(words: list[Cue], sentences: list[list[Cue]], lm: dict[str, Any] | None) -> dict[str, list[str]]:
    green = set(map(normalize_phrase, list_from_lm(lm, "green")))
    yellow = set(map(normalize_phrase, list_from_lm(lm, "yellow")))
    red = set(map(normalize_phrase, list_from_lm(lm, "red")))

    heuristic_red = {
        "не", "нет", "нельзя", "плохо", "опасно", "ошибка", "уголовка", "штраф", "долг",
        "налоговая", "ндс", "проблема", "минус", "потеря", "страх", "сложно", "больно",
        "дорого", "убыток", "хаос", "забыли", "повесили", "не могу",
    }
    heuristic_green = {
        "выгода", "прибыль", "лучше", "улучшила", "рост", "сэкономил", "заплатил", "можно",
        "получилось", "результат", "приятное", "плюс", "работает", "решение", "готово",
    }
    heuristic_yellow = {
        "налог", "доход", "деньги", "рублей", "тысяч", "процент", "1%", "ндс", "ип", "ооо",
        "сейчас", "раньше", "первый", "важно", "вопрос", "ответ", "система", "клиент",
    }

    for word in words:
        n = normalize_word(word.text)
        if not n:
            continue
        if re.search(r"\d", n) or "%" in n:
            yellow.add(n)
        if n in heuristic_red:
            red.add(n)
        if n in heuristic_green:
            green.add(n)
        if n in heuristic_yellow:
            yellow.add(n)

    # Make sure sentence-level important repeated terms are highlighted.
    full_text = normalize_phrase(" ".join(w.text for w in words))
    for phrase in heuristic_red | heuristic_green | heuristic_yellow:
        if " " in phrase and phrase in full_text:
            if phrase in heuristic_red:
                red.add(phrase)
            elif phrase in heuristic_green:
                green.add(phrase)
            else:
                yellow.add(phrase)

    return {
        "green": sorted(x for x in green if x),
        "yellow": sorted(x for x in yellow if x),
        "red": sorted(x for x in red if x),
    }


def color_for_word(index: int, words: list[Cue], rules: dict[str, list[str]]) -> str:
    # Phrase priority: red > green > yellow.
    max_phrase_len = 5
    for color in ("red", "green", "yellow"):
        phrases = rules.get(color, [])
        phrase_set = {p for p in phrases if " " in p}
        for length in range(max_phrase_len, 1, -1):
            if index + length > len(words):
                continue
            candidate = normalize_phrase(" ".join(w.text for w in words[index:index + length]))
            if candidate in phrase_set:
                return color

    n = normalize_word(words[index].text)
    if not n:
        return "default"
    if n in set(rules.get("red", [])):
        return "red"
    if n in set(rules.get("green", [])):
        return "green"
    if n in set(rules.get("yellow", [])):
        return "yellow"
    return "default"


def build_colored_tokens(group: list[Cue], all_words: list[Cue], offset: int, rules: dict[str, list[str]]) -> list[dict[str, str]]:
    return [
        {"text": cue.text, "color": color_for_word(offset + i, all_words, rules)}
        for i, cue in enumerate(group)
    ]


def make_subtitle_items(groups: list[list[Cue]], all_words: list[Cue], rules: dict[str, list[str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    for group in groups:
        text = " ".join(w.text for w in group)
        item = {
            "start": round(group[0].start, 3),
            "end": round(group[-1].end, 3),
            "start_timecode": seconds_to_timecode(group[0].start),
            "end_timecode": seconds_to_timecode(group[-1].end),
            "text": text,
            "tokens": build_colored_tokens(group, all_words, offset, rules),
        }
        items.append(item)
        offset += len(group)
    return items


def sanitize_lm_zoom(lm: dict[str, Any] | None, duration: float) -> list[dict[str, Any]]:
    if not isinstance(lm, dict) or not isinstance(lm.get("zoom"), list):
        return []
    out: list[dict[str, Any]] = []
    allowed = [0, 30, 40, 50, 60, 70]
    last_t = -999.0
    for raw in lm["zoom"]:
        if not isinstance(raw, dict):
            continue
        try:
            t = float(raw.get("time", raw.get("start", raw.get("timecode"))))
            p = int(round(float(raw.get("percent", 40))))
        except Exception:
            continue
        if not math.isfinite(t) or t < 0 or t > duration + 0.25:
            continue
        if t - last_t < 1.8:
            continue
        p = min(allowed, key=lambda x: abs(x - p))
        out.append({"timecode": seconds_to_timecode(t)[3:], "percent": p})
        last_t = t
    return out[:24]


def sentence_importance(group: list[Cue], rules: dict[str, list[str]]) -> int:
    score = 0
    text = normalize_phrase(" ".join(w.text for w in group))
    for word in group:
        n = normalize_word(word.text)
        if re.search(r"\d", n):
            score += 2
        if n in rules.get("red", []):
            score += 4
        elif n in rules.get("green", []):
            score += 3
        elif n in rules.get("yellow", []):
            score += 2
    for color, add in (("red", 5), ("green", 4), ("yellow", 3)):
        for phrase in rules.get(color, []):
            if " " in phrase and phrase in text:
                score += add
    return score


def build_zoom_events(sentences: list[list[Cue]], rules: dict[str, list[str]], lm: dict[str, Any] | None, duration: float) -> list[dict[str, Any]]:
    lm_events = sanitize_lm_zoom(lm, duration)
    if lm_events:
        first_time = lm_events[0].get("timecode")
        if first_time != "00:00.000":
            lm_events.insert(0, {"timecode": "00:00.000", "percent": 0})
        return lm_events

    events: list[dict[str, Any]] = [{"timecode": "00:00.000", "percent": 0}]
    last_t = 0.0
    for group in sentences:
        t = group[0].start
        if t - last_t < 2.0:
            continue
        score = sentence_importance(group, rules)
        if score >= 12:
            percent = 70
        elif score >= 8:
            percent = 60
        elif score >= 5:
            percent = 50
        elif score >= 2:
            percent = 40
        else:
            percent = 30
        events.append({"timecode": seconds_to_timecode(t)[3:], "percent": percent})
        last_t = t
        if len(events) >= 22:
            break
    return events


def base_subtitle_render(output: str) -> dict[str, Any]:
    return {
        "video": DEFAULT_ZOOM_OUTPUT,
        "output": output,
        "font": None,
        "position": {"x": 0.12, "y": 0.66, "anchor": "left"},
        "box": {
            "enabled": True,
            "max_width": 0.76,
            "padding_x": 20,
            "padding_y": 12,
            "radius": 20,
            "background": "#00000088",
        },
        "style": {
            "font_size": 64,
            "line_spacing": 1.04,
            "text_color": "#FFFFFF",
            "stroke_color": "#000000",
            "stroke_width": 4,
            "highlight": {
                "green": "#00E676",
                "yellow": "#FFD54F",
                "red": "#FF3D00",
            },
        },
        "layout": {"max_lines": 2, "uppercase": False},
        "encoding": {"crf": 0, "preset": "medium"},
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python generate_configs_from_transcript_srt.py path/to/subtitles_words.srt")
        raise SystemExit(2)

    srt_path = Path(sys.argv[1]).expanduser().resolve()
    out_dir = srt_path.parent

    source_cues = parse_srt(srt_path)
    words = split_cue_to_word_cues(source_cues)
    sentences = group_sentences(words)
    phrases = group_phrases(words)
    duration = max(w.end for w in words)

    lm = ask_bigpickle(sentences)
    rules = build_rules(words, sentences, lm)

    sentence_items = make_subtitle_items(sentences, words, rules)
    phrase_items = make_subtitle_items(phrases, words, rules)
    word_items = make_subtitle_items([[w] for w in words], words, rules)
    zoom_events = build_zoom_events(sentences, rules, lm, duration)

    zoom_config = {
        "render": {
            "video": DEFAULT_SOURCE_VIDEO,
            "output": DEFAULT_ZOOM_OUTPUT,
            "anchor_mode": "smoothed_eye",
            "eye_screen": {"x": 0.5, "y": 0.42},
            "eye_follow_lag": {"enabled": True, "seconds": 0.35},
            "edge_mode": "reflect",
            "encoding": {"crf": 0, "preset": "medium"},
            "zoom": {
                "source": "timeline",
                "percent_mode": "direct",
                "default_zoom": 1.0,
                "transition": 0.15,
                "direct_strength": 2.0,
                "timeline": zoom_events,
            },
        }
    }

    subtitle_sentences_config = {
        "render": base_subtitle_render(DEFAULT_SUBTITLE_OUTPUT),
        "source_transcript": str(srt_path.name),
        "subtitle_mode": "sentences",
        "highlight_rules": {
            "green_words": rules["green"],
            "yellow_words": rules["yellow"],
            "red_words": rules["red"],
        },
        "subtitles": sentence_items,
    }

    subtitle_words_config = {
        "render": base_subtitle_render(DEFAULT_SUBTITLE_WORDS_OUTPUT),
        "source_transcript": str(srt_path.name),
        "subtitle_mode": "words",
        "highlight_rules": {
            "green_words": rules["green"],
            "yellow_words": rules["yellow"],
            "red_words": rules["red"],
        },
        "subtitles": word_items,
    }
    subtitle_phrases_config = {
        "render": base_subtitle_render("output/timeline_zoom_subtitles_phrases.mp4"),
        "source_transcript": str(srt_path.name),
        "subtitle_mode": "phrases",
        "highlight_rules": {
            "green_words": rules["green"],
            "yellow_words": rules["yellow"],
            "red_words": rules["red"],
        },
        "subtitles": phrase_items,
    }

    zoom_path = out_dir / "zoom_timeline_config.json"
    subtitle_path = out_dir / "subtitle_timeline_config.json"
    subtitle_words_path = out_dir / "subtitle_timeline_config_words.json"
    subtitle_phrases_path = out_dir / "subtitle_timeline_config_phrases.json"

    write_json(zoom_path, zoom_config)
    write_json(subtitle_path, subtitle_sentences_config)
    write_json(subtitle_words_path, subtitle_words_config)
    write_json(subtitle_phrases_path, subtitle_phrases_config)

    print("OK")
    print(f"input: {srt_path}")
    print(f"words: {len(words)}")
    print(f"sentence subtitles: {len(sentence_items)} -> {subtitle_path}")
    print(f"word subtitles: {len(word_items)} -> {subtitle_words_path}")
    print(f"phrase subtitles: {len(phrase_items)} -> {subtitle_phrases_path}")
    print(f"zoom events: {len(zoom_events)} -> {zoom_path}")
    print("next:")
    print("  python apply_face_zoom_v5.py --tracking data/face_tracking.json --config zoom_timeline_config.json")
    print("  python apply_auto_subtitles_v2.py --config subtitle_timeline_config.json")
    print("  python apply_auto_subtitles_v2.py --config subtitle_timeline_config_words.json")


if __name__ == "__main__":
    main()
