"""Browser-level tests for the unified progress/operations UI (Playwright).

Covers spec items: no duplicate progress bars, main bar reflects the active
operation, indeterminate progress, stage errors are surfaced (not masked),
partial publish shown as «Частично завершено», per-target progress, switching
operations keeps montage state, concurrent operations, stale-version guard,
all-failed publish shown as «Ошибка публикации» (never masked as partial),
unknown outcomes require manual check, interrupted montage surfaces an error
after refresh, and transcription defaults actually sent to the server.

Run (server must be listening on :8000):
  python test_progress_ui.py
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000"
SHOT_DIR = Path("/tmp/transcriber_screens")
SHOT_DIR.mkdir(parents=True, exist_ok=True)

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


def inject(page, state):
    page.evaluate(
        """(s) => { window.__opsState = {}; window.__opsBodySig = null; mergeOpsState(s); renderOperations(window.__opsState); }""",
        state,
    )


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE_URL, wait_until="networkidle")
        page.evaluate("() => switchView('montage')")

        # 1. no duplicate progress bars
        check("single #progressBar", page.locator("#progressBar").count() == 1,
              f"count={page.locator('#progressBar').count()}")
        check("single #montageProgressCard", page.locator("#montageProgressCard").count() == 1)
        check("single #operationsBody", page.locator("#operationsBody").count() == 1)

        # 2. montage running -> main title + stages
        inject(page, {"operation": {"status": "running", "active_stage": "render",
                                    "progress": 42.5, "label": "Монтаж видео", "detail": "Рендер кадров",
                                    "stages": [{"key": "prepare", "label": "Подготовка", "status": "completed", "progress": 100},
                                               {"key": "render", "label": "Рендер", "status": "running", "progress": 40}]}})
        check("montage main title", page.text_content("#operationTitle") == "Монтаж видео",
              page.text_content("#operationTitle"))
        check("montage main percent", page.text_content("#operationPercent") == "42.5%",
              page.text_content("#operationPercent"))
        check("stage rows rendered", page.locator("#operationsBody .ops-stage").count() >= 2)

        # 3. indeterminate
        inject(page, {"operation": {"status": "running", "progress": None, "label": "Монтаж видео", "stages": []}})
        check("indeterminate class", "indeterminate" in (page.get_attribute("#operationMainBarWrap", "class") or ""))
        check("indeterminate percent ellipsis", page.text_content("#operationPercent") == "…",
              page.text_content("#operationPercent"))

        # 4. error not masked
        inject(page, {"operation": {"status": "error", "progress": 33, "label": "Монтаж видео",
                                    "error": "FFmpeg failed", "stages": []}})
        check("error percent", page.text_content("#operationPercent") == "ошибка",
              page.text_content("#operationPercent"))
        check("error detail shows message", "FFmpeg failed" in page.text_content("#operationDetail"))
        check("error styled", page.locator("#operationsBody .ops-op-error").count() >= 1)

        # 5. montage completed with warning -> title not "Видео готово"
        inject(page, {"operation": {"status": "completed", "progress": 100, "label": "Монтаж видео",
                                    "warning": "Публикация без звуковой дорожки", "stages": []}})
        check("warning hides ready title", page.text_content("#operationTitle") != "Видео готово",
              page.text_content("#operationTitle"))
        check("warning surfaced", "звуков" in page.text_content("#operationDetail"))

        # 6. montage completed clean -> "Видео готово"
        inject(page, {"operation": {"status": "completed", "progress": 100, "label": "Монтаж видео", "stages": []}})
        check("clean completion title", page.text_content("#operationTitle") == "Видео готово",
              page.text_content("#operationTitle"))

        # 7. partial publish
        inject(page, {"publish_progress": {
            "ig": {"label": "Instagram", "progress": 100, "status": "done", "detail": "Опубликовано"},
            "yt": {"label": "YouTube", "progress": 100, "status": "error", "detail": "Квота"},
        }})
        check("publish is main op", page.text_content("#operationTitle") == "Публикация",
              page.text_content("#operationTitle"))
        check("partial warning is shown", "Не все площадки" in page.text_content("#operationDetail"),
              page.text_content("#operationDetail"))
        check("partial label частично", page.locator("#operationsBody .ops-op-warn").count() >= 1)
        check("targets listed", page.locator("#operationsBody .ops-stage").count() >= 2)
        page.screenshot(path=str(SHOT_DIR / "ops_partial_publish.png"))

        # 7b. all-failed publish -> «Ошибка публикации», never «Частично завершено»
        inject(page, {"publish_progress": {
            "ig": {"label": "Instagram", "progress": 100, "status": "error", "detail": "Квота"},
            "yt": {"label": "YouTube", "progress": 100, "status": "error", "detail": "Сеть"},
        }})
        body7b = page.text_content("#operationsBody")
        check("all-failed title", 'Ошибка публикации' == page.text_content("#operationTitle"),
              page.text_content("#operationTitle"))
        check("all-failed status text", "Ошибка публикации" in body7b, body7b)
        check("all-failed not partial", "Частично завершено" not in body7b, body7b)
        check("all-failed error row", page.locator("#operationsBody .ops-op-error").count() >= 1)
        check("all-failed warning", "Ни одна площадка" in page.text_content("#operationDetail"),
              page.text_content("#operationDetail"))

        # 7c. mixed done/error/unknown -> needs manual check, no auto-republish
        inject(page, {"publish_progress": {
            "ig": {"label": "Instagram", "progress": 100, "status": "done", "detail": "Опубликовано"},
            "yt": {"label": "YouTube", "progress": 100, "status": "error", "detail": "Квота"},
            "tg": {"label": "Telegram", "progress": None, "status": "unknown", "detail": "Результат неизвестен"},
        }})
        body7c = page.text_content("#operationsBody")
        check("mixed needs-check title", "Требуется проверка" == page.text_content("#operationTitle"),
              page.text_content("#operationTitle"))
        check("mixed status text", "Требуется проверка" in body7c, body7c)
        check("mixed counts", "Опубликовано 1 из 3" in page.text_content("#operationDetail")
              and "нужна проверка: 1" in page.text_content("#operationDetail"),
              page.text_content("#operationDetail"))
        check("unknown target indeterminate", page.locator("#operationsBody .ops-op-bar.ind").count() >= 1)
        check("unknown target warn dot", page.locator("#operationsBody .ops-stage .dot.warn").count() >= 1)
        page.screenshot(path=str(SHOT_DIR / "ops_unknown_publish.png"))

        # 7d. interrupted montage after refresh -> error surfaced, start re-enabled
        page.evaluate("""() => {
          const orig = window.fetch;
          window.fetch = (url) => {
            if (String(url).includes('/api/montage/result/')) {
              return Promise.resolve({ok: true, json: () => Promise.resolve({
                status: 'failed', stage: 'interrupted', progress: 57,
                detail: 'Монтаж прерван перезапуском сервера',
                error: 'Монтаж прерван перезапуском сервера',
              })});
            }
            return orig(url);
          };
          activateMontageJob('ui_interrupted_test');
          window.fetch = orig;
        }""")
        page.wait_for_timeout(300)
        check("interrupt message shown", "Монтаж прерван перезапуском сервера" in page.text_content("#montageErrorBox"),
              page.text_content("#montageErrorBox"))
        check("interrupt start re-enabled", page.locator("#montageStartBtn").is_enabled())

        # 8. concurrent operations (montage + publish)
        inject(page, {
            "operation": {"status": "running", "progress": 20, "label": "Монтаж видео", "stages": []},
            "publish_progress": {"ig": {"label": "Instagram", "progress": None, "status": "publishing", "detail": "Загрузка"}},
        })
        body = page.text_content("#operationsBody")
        check("concurrent montage shown", "Монтаж видео" in body)
        check("concurrent publish shown", "Instagram" in body)

        # 9. switching operations preserves montage snapshot
        page.evaluate("""() => { window.__opsState = {operation: {status:'running', progress:55, label:'Монтаж видео', version:5, stages:[]}}; }""")
        page.evaluate("""() => { mergeOpsState({publish_progress: {"ig": {"label": "Instagram", "progress": 10, "status": "publishing", "detail": "Загрузка"}}}); renderOperations(window.__opsState); }""")
        kept = page.evaluate("() => window.__opsState.operation && window.__opsState.operation.progress")
        check("montage state preserved on switch", kept == 55, f"kept={kept}")

        # 10. stale version guard
        page.evaluate("""() => { window.__opsState = {operation: {status:'running', progress:80, label:'Монтаж видео', version:9, stages:[]}}; }""")
        page.evaluate("""() => { mergeOpsState({operation: {status:'running', progress:5, label:'Монтаж видео', version:2, stages:[]}}); }""")
        v = page.evaluate("() => window.__opsState.operation.version")
        p = page.evaluate("() => window.__opsState.operation.progress")
        check("stale snapshot rejected", v == 9 and p == 80, f"v={v} p={p}")

        # 11. collapse/expand
        page.evaluate("() => renderOperations(window.__opsState)")
        before = "closed" in (page.get_attribute("#operationsBody", "class") or "")
        page.evaluate("() => toggleOperationsBody()")
        after = "closed" in (page.get_attribute("#operationsBody", "class") or "")
        check("toggle collapses body", before != after)

        # 12. transcription defaults transmitted
        captured = {}

        def on_upload(route):
            route.fulfill(status=200, content_type="application/json", body='{"job_id":"fake-upload-1"}')

        def on_transcribe(route):
            captured["post"] = route.request.post_data
            route.fulfill(status=200, content_type="application/json", body='{"ok":true}')

        page.route("**/api/upload", on_upload)
        page.route("**/api/transcribe/**", on_transcribe)
        page.route("**/api/status/**", lambda r: r.fulfill(status=200, content_type="application/json", body='{"status":"processing","progress":1,"stage":"x","detail":""}'))
        page.evaluate("() => switchView('transcribe')")
        tmp = Path("/tmp/progress_ui_dummy.txt")
        tmp.write_text("dummy")
        page.set_input_files("#fileInput", str(tmp))
        page.wait_for_function("() => document.getElementById('startBtn') && !document.getElementById('startBtn').disabled", timeout=8000)
        page.evaluate("() => startTranscription()")
        page.wait_for_timeout(800)
        body = captured.get("post") or ""
        check("mode=words transmitted", 'name="mode"' in body and "words" in body, body[:200])
        check("diarization=false transmitted", 'name="diarization"' in body and "false" in body.split('name="diarization"')[1][:20], body[:200])

        page.screenshot(path=str(SHOT_DIR / "ops_desktop.png"))
        browser.close()

    print(f"\n=== progress UI: {PASS} passed, {FAIL} failed ===")
    if errors:
        print("PAGE ERRORS:", errors)
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print(" -", f)
    return 1 if (FAIL or errors) else 0


if __name__ == "__main__":
    import os
    _code = main()
    sys.stdout.flush()
    os._exit(_code)
