"""
Полный E2E тест веб-интерфейса Transcriber в Playwright.

Запуск:
  SSL_CERT_FILE=... /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 test_ui.py
"""
import os, sys, json, time, re
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

BASE_URL = "http://127.0.0.1:8000"
TEST_AUDIO = "/tmp/test_upload.ogg"
SCREENSHOT_DIR = "/tmp/transcriber_screens"
Path(SCREENSHOT_DIR).mkdir(exist_ok=True)

PASS = 0
FAIL = 0
FAILURES = []


def check(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))


def upload_file(page, path=TEST_AUDIO):
    page.set_input_files("#fileInput", path)
    page.wait_for_timeout(1500)


def get_progress(page):
    return page.eval_on_selector("#progressBar", "el => el.style.width")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("console", lambda msg: errors.append(f"[console] {msg.type}: {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(f"[pageerror] {exc}"))

        # === 1. Загрузка страницы ===
        print("\n=== 1. Загрузка интерфейса ===")
        page.goto(BASE_URL, wait_until="networkidle")
        check("Страница загружается", page.title() == "Transcriber" or "Transcriber" in page.content(),
              f"title={page.title()}")
        check("Заголовок 'Transcriber'", page.locator("h1").text_content() == "Transcriber")
        check("Зона загрузки видна", page.locator("#uploadZone").is_visible())
        page.screenshot(path=f"{SCREENSHOT_DIR}/01_main.png")

        # === 2. Загрузка файла ===
        print("\n=== 2. Загрузка файла ===")
        upload_file(page)
        check("Имя файла отображается", page.locator("#fileName").text_content().strip() == "test_upload.ogg",
              f"got={page.locator('#fileName').text_content()}")
        check("Карточка настроек появилась", page.locator("#settingsCard").is_visible())
        check("Кнопка старта активна", page.locator("#startBtn").is_enabled())
        page.screenshot(path=f"{SCREENSHOT_DIR}/02_uploaded.png")

        # === 3. Настройки ===
        print("\n=== 3. Настройки ===")
        page.select_option("#modelSelect", "tiny")
        check("Выбрана модель tiny", page.locator("#modelSelect").input_value() == "tiny")
        page.select_option("#languageSelect", "ru")
        check("Выбран язык ru", page.locator("#languageSelect").input_value() == "ru")
        page.select_option("#modeSelect", "sentences")
        check("Режим по предложениям", page.locator("#modeSelect").input_value() == "sentences")

        # Диаризация включение -> блок количества спикеров
        check("Блок спикеров изначально скрыт",
              not page.locator("#speakerCountRow").is_visible())
        page.locator("#diarizationToggle + .toggle-slider").click()
        page.wait_for_timeout(300)
        check("Включение диаризации показывает списокеров",
              page.locator("#speakerCountRow").is_visible())
        page.fill("#numSpeakers", "3")
        check("Кол-во спикеров установлено", page.locator("#numSpeakers").input_value() == "3")
        page.locator("#diarizationToggle + .toggle-slider").click()
        page.wait_for_timeout(300)
        check("Выкл диаризации скрывает списокеров",
              not page.locator("#speakerCountRow").is_visible())
        page.screenshot(path=f"{SCREENSHOT_DIR}/03_settings.png")

        # === 4. Запуск транскрибации (без диаризации) ===
        print("\n=== 4. Транскрибация без диаризации ===")
        page.select_option("#modelSelect", "small")
        page.click("#startBtn")
        page.wait_for_timeout(500)
        check("Прогресс-бар появился", page.locator("#progressCard").is_visible())
        started = False
        for i in range(180):
            stage = page.locator("#progressStage").text_content()
            detail = page.locator("#progressDetail").text_content()
            if page.locator("#resultsCard").is_visible():
                started = True
                break
            page.wait_for_timeout(1000)
        check("Обработка завершилась и результат показался", started, f"stage={stage} detail={detail}")
        page.screenshot(path=f"{SCREENSHOT_DIR}/04_progress.png")

        # === 5. Вкладки результата ===
        print("\n=== 5. Отображение результата ===")
        check("Вкладка Диалог активна по умолчанию",
              page.locator(".result-tab.active").text_content() == "Диалог")
        check("Текст диалога не пуст", len(page.locator("#resultDisplay").text_content()) > 50)

        # switch to Текст
        page.click(".result-tab:has-text('Текст')")
        page.wait_for_timeout(200)
        check("Вкладка Текст активна", page.locator(".result-tab.active").text_content() == "Текст")
        check("Текст отображается", len(page.locator("#resultDisplay").text_content()) > 50)
        page.screenshot(path=f"{SCREENSHOT_DIR}/05_text.png")

        # switch to JSON
        page.click(".result-tab:has-text('JSON')")
        page.wait_for_timeout(200)
        check("Вкладка JSON активна", page.locator(".result-tab.active").text_content() == "JSON")
        json_content = page.locator("#resultDisplay").text_content()
        check("JSON содержит структуру", '"text"' in json_content)
        page.screenshot(path=f"{SCREENSHOT_DIR}/06_json.png")
        page.click(".result-tab:has-text('Диалог')")

        # === 6. Тумблеры в реальном времени ===
        print("\n=== 6. Переключение тумблеров ===")
        # Включить диаризацию на уже готовом result? Нет - меняем визуализацию только.
        tc_before = page.locator(".timecode").count()
        page.locator("#showTimecodes + .toggle-slider").click()
        page.wait_for_timeout(300)
        tc_after = page.locator(".timecode").count()
        check("Выкл тайм-кодов убирает их", tc_before > 0 and tc_after == 0,
              f"before={tc_before} after={tc_after}")
        page.locator("#showTimecodes + .toggle-slider").click()
        page.wait_for_timeout(300)
        check("Вкл тайм-кодов возвращает их", page.locator(".timecode").count() == tc_before)

        page.screenshot(path=f"{SCREENSHOT_DIR}/07_toggles.png")

        # === 7. Скачивание файлов ===
        print("\n=== 7. Скачивание файлов ===")
        job_id = page.evaluate("jobId")
        results = {}
        expected = {
            "srt": ".srt",
            "vtt": "WEBVTT",
            "txt": None,
            "json": "{",
            "tsv": "text",
            "dialogue": None,
        }
        for fmt in ["srt", "vtt", "txt", "json", "tsv", "dialogue"]:
            url = f"{BASE_URL}/api/export/{job_id}/{fmt}?speakers=true&timecodes=true"
            resp = page.context.request.get(url)
            body = resp.body().decode("utf-8")
            size = len(body.encode("utf-8"))
            results[fmt] = size
            check(f"Скачан {fmt} ({size} B)", resp.ok and size > 0, f"status={resp.status}")
            Path(f"/tmp/download_test_{fmt}").write_text(body, encoding="utf-8")

        # содержание файлов
        srt_content = Path("/tmp/download_test_srt").read_text(encoding="utf-8")
        check("SRT имеет тайм-коды", "-->" in srt_content)
        check("SRT имеет номер строки", re.search(r"^\d+$", srt_content, re.M) is not None)
        check("SRT не пуст текстом", any(line.strip() and not re.match(r"^[\d:,\.\-\s>]+$", line.strip())
                                         for line in srt_content.splitlines()))
        txt_content = Path("/tmp/download_test_txt").read_text(encoding="utf-8")
        check("TXT содержит текст", len(txt_content.strip()) > 50)
        json_content = Path("/tmp/download_test_json").read_text(encoding="utf-8")
        json_data = json.loads(json_content)
        check("JSON - массив сегментов", isinstance(json_data, list) and len(json_data) > 0)
        check("JSON элементы имеют text", all("text" in s for s in json_data))

        # === 8. API доступ ===
        print("\n=== 8. Проверка REST API ===")
        api_status = page.evaluate("""async () => {
            const res = await fetch('/api/jobs');
            return await res.json();
        }""")
        check("GET /api/jobs возвращает массив", isinstance(api_status, list))

        # === 9. Мультиформат: загрузка другого файла после результата ===
        print("\n=== 9. Повторная загрузка ===")
        page.set_input_files("#fileInput", TEST_AUDIO)
        page.wait_for_timeout(1500)
        check("Новый файл принят", "test_upload.ogg" in page.locator("#fileName").text_content())

        # === 10. Ошибки на странице ===
        print("\n=== 10. JS-ошибки ===")
        check("Нет ошибок JS в консоли", len(errors) == 0, "; ".join(errors[:5]))

        page.screenshot(path=f"{SCREENSHOT_DIR}/08_final.png", full_page=True)
        browser.close()

    print(f"\n{'='*50}")
    print(f"ИТОГО: {PASS} PASS, {FAIL} FAIL")
    if FAILURES:
        print("\nПроваленные проверки:")
        for f in FAILURES:
            print(f"  - {f}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())