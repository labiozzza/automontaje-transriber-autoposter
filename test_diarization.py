"""
Тест диаризации и режима "по словам" через API.
"""
import os, sys, json, re
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000"
WAV_3SPEAKERS = "/tmp/diarization_test.wav"

PASS = 0
FAIL = 0
FAILURES = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"[FAIL] {name} — {detail}")


def main():
    global PASS, FAIL, FAILURES
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context()
        page = ctx.new_page()

        print("=== 1. Диаризация с 3 спикерами ===")
        with page.expect_response(lambda r: r.url.endswith("/api/upload") and r.ok) as up:
            page.goto(BASE_URL)
            page.set_input_files("#fileInput", WAV_3SPEAKERS)
            page.wait_for_timeout(1000)
        job = up.value.json()["job_id"]
        check("Файл загружен", bool(job), job)

        resp = ctx.request.post(f"{BASE_URL}/api/transcribe/{job}",
                                form={"model": "tiny", "language": "ru",
                                      "mode": "sentences", "diarization": "true",
                                      "num_speakers": "3"})
        check("Транскрибация запущена", resp.ok)

        # ждём выполнения
        result = None
        for _ in range(120):
            st = ctx.request.get(f"{BASE_URL}/api/status/{job}").json()
            if st["status"] == "done":
                result = ctx.request.get(f"{BASE_URL}/api/result/{job}").json()["segments"]
                break
            if st["status"] == "error":
                check("Ошибка не возникла", False, st["detail"])
                break
            page.wait_for_timeout(1000)

        check("Результат получен", result is not None)
        if result:
            speakers = set(s.get("speaker") for s in result)
            check("Диаризация определила спикеров", any(s for s in speakers), str(speakers))
            check("Найдено 3 спикера", len([s for s in speakers if s]) == 3, str(speakers))
            check("Все сегменты имеют спикера", all(s.get("speaker") for s in result),
                  str(set(s.get("speaker") for s in result)))

        # SRT с диаризацией
        srt = ctx.request.get(f"{BASE_URL}/api/export/{job}/srt?speakers=true&timecodes=true").text()
        check("SRT содержит [SPEAKER_", "[SPEAKER_" in srt)
        speaker_tags = set(re.findall(r"\[(SPEAKER_\d+)\]", srt))
        check("В SRT 3 тега спикеров", len(speaker_tags) == 3, str(speaker_tags))

        print("\n=== 2. Режим 'по словам' ===")
        with page.expect_response(lambda r: r.url.endswith("/api/upload") and r.ok) as up:
            page.set_input_files("#fileInput", "/tmp/test_upload.ogg")
            page.wait_for_timeout(1000)
        job2 = up.value.json()["job_id"]

        ctx.request.post(f"{BASE_URL}/api/transcribe/{job2}",
                         form={"model": "tiny", "language": "ru",
                               "mode": "words", "diarization": "false"})

        result2 = None
        for _ in range(120):
            st = ctx.request.get(f"{BASE_URL}/api/status/{job2}").json()
            if st["status"] == "done":
                result2 = ctx.request.get(f"{BASE_URL}/api/result/{job2}").json()["segments"]
                break
            if st["status"] == "error":
                check("Ошибка не возникла", False, st["detail"])
                break
            page.wait_for_timeout(1000)

        check("Результат (слова) получен", result2 is not None)
        if result2:
            check("Сегментов больше чем предложениями", len(result2) > 20, f"len={len(result2)}")
            check("Слова короткие", all(len(s["text"].split()) <= 5 for s in result2),
                  str([s["text"] for s in result2[:5]]))

        print("\n=== 3. Экспорт dialogue ===")
        dial = ctx.request.get(f"{BASE_URL}/api/export/{job}/dialogue?timecodes=true").text()
        check("Dialogue содержит разделители", "═══" in dial)
        check("Dialogue не пуст", len(dial) > 100)

        browser.close()

    print(f"\nИТОГО: {PASS} PASS, {FAIL} FAIL")
    if FAILURES:
        print("Провалы:")
        for f in FAILURES:
            print(f"  - {f}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())