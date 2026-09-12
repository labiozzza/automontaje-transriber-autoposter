"""
Юнит-тесты модуля автомонтажа.

Запуск:
  /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 test_montage_unit.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from montage import engine as m  # noqa: E402

PASS = 0
FAIL = 0
FAILURES = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))


def main():
    # 1. build_words_srt
    segments = [
        {"start": 0.0, "end": 1.5, "text": "Привет мир"},
        {"start": 2.0, "end": 3.0, "text": "Текст"},
    ]
    srt = m.build_words_srt(segments)
    check("build_words_srt создаёт SRT с 2 блоками", srt.count("\n\n") == 1, repr(srt))
    check("SRT содержит тайминги", "00:00:00,000 --> 00:00:01,500" in srt)
    check("SRT содержит текст", "Привет мир" in srt and "Текст" in srt)
    check("Пустые сегменты пропускаются", len(m.build_words_srt([
        {"start": 0, "end": 0, "text": ""},
        {"start": 1, "end": 2, "text": "да"}]).split("\n\n")) == 1)

    # 2. ensure_tracking_has_anchors — synthetic fallback
    with tempfile.TemporaryDirectory() as td:
        tracking = Path(td) / "face_tracking.json"
        tracking.write_text(json.dumps({
            "source": {"fps": 30, "duration": 3.0},
            "camera_targets": [{"face_found": False, "frame": 0, "time": 0.0}],
        }), encoding="utf-8")
        ok = m.ensure_tracking_has_anchors(tracking)
        data = json.loads(tracking.read_text(encoding="utf-8"))
        check("ensure_tracking добавляет якоря", ok and data.get("_synthetic_anchors") is True)
        check("Все синтетические таргеты face_found", all(item["face_found"] for item in data["camera_targets"]))
        n = len(data["camera_targets"])
        check("Синтетических таргетов достаточно", n >= 25, f"n={n}")

    # 3. Генерация конфигов без BigPickle (fallback-эвристики)
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        srt_path = workdir / "fallback.srt"
        srt_path.write_text("""1
00:00:00,000 --> 00:00:02,000
Прибыль растёт.

2
00:00:02,000 --> 00:00:04,000
Ошибка случилась, но всё хорошо.
""", encoding="utf-8")
        cmd_out = m.run_cmd(
            [sys.executable, "-u", str(m.MONAGE_DIR / "generate_configs_from_transcript_srt.py"), str(srt_path)],
            cwd=workdir, log_path=workdir / "cfg.log", timeout=300,
        )
        check("Генератор конфигов без BigPickle завершился 0", cmd_out == 0, f"code={cmd_out}")
        zoom = workdir / "zoom_timeline_config.json"
        sub = workdir / "subtitle_timeline_config.json"
        check("zoom_timeline_config создан", zoom.exists())
        check("subtitle_timeline_config создан", sub.exists())
        if zoom.exists():
            data = json.loads(zoom.read_text(encoding="utf-8"))
            events = data["render"]["zoom"]["timeline"]
            check("Zoom timeline содержит события", isinstance(events, list) and len(events) >= 1, f"len={len(events) if isinstance(events,list) else 'n/a'}")
        if sub.exists():
            data = json.loads(sub.read_text(encoding="utf-8"))
            check("Есть правила подсветки", isinstance(data.get("highlight_rules"), dict))

    # 4. montage_status не падает и возвращает нужные ключи
    status = m.montage_status()
    for key in ("online", "opencode", "bigpickle_enabled", "face_model", "font", "media_repo", "animations", "publishers"):
        check(f"montage_status[{key}]", key in status, f"missing {key}")
    check("publishers содержат instagram", isinstance(status["publishers"].get("instagram_reels"), bool))

    # 5. ffprobe на синтетическом видео перехватывает ошибки
    try:
        probe = m.ffprobe(Path("/tmp/montage_test.mp4"))
        check("ffprobe читает синтетическое видео", float(probe["format"]["duration"]) > 0)
    except Exception as exc:
        check("ffprobe читает синтетическое видео", False, repr(exc))

    # 6. Экспорт текста
    ok_text = m.build_words_srt([{"start": 5.5, "end": 6.25, "text": "тест"}])
    check("таймкоды > минуты", "00:00:05,500" in ok_text, repr(ok_text))

    # 7. format_timecode для override таймлайна
    check("format_timecode сек<60", m.format_timecode(0) == "00:00.000", m.format_timecode(0))
    check("format_timecode секунды с миллисекундами", m.format_timecode(2.5) == "00:02.500", m.format_timecode(2.5))
    check("format_timecode минуты", m.format_timecode(65) == "01:05.000", m.format_timecode(65))
    check("format_timecode переполнение мс→сек", m.format_timecode(4.99999) == "00:05.000", m.format_timecode(4.99999))
    check("format_timecode отрицательное→0", m.format_timecode(-3) == "00:00.000", m.format_timecode(-3))

    print("\n==================================================")
    print(f"ИТОГО: {PASS} PASS, {FAIL} FAIL")
    if FAILURES:
        for f in FAILURES:
            print("  FAIL:", f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()