"""Browser-level tests for the publish-preparation UI (Playwright).

Covers the new spec items:
  * a dedicated «Подготовка к публикации» operation appears and becomes the main
    operation while preparation runs;
  * publish target checkboxes are never disabled by preparation (artefacts are
    prepared automatically), while mirrored targets follow the mirror state;
  * level-3 stage folders of different targets expand/collapse independently;
  * failed operations show «—» instead of 100%;
  * running stages expose an elapsed-time label.

Run (server must be listening on :8000):
  python test_prepare_ui.py
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


def _empty_status():
    import json
    import urllib.request
    with urllib.request.urlopen(BASE_URL + "/api/montage/status", timeout=30) as resp:
        status = json.loads(resp.read().decode("utf-8"))
    status["jobs"] = []
    return json.dumps(status)


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
        # Не даём странице подхватить реальную последнюю задачу: это перезапишет
        # инъецируемое состояние и сделает тест недетерминированным.
        empty_status = _empty_status()
        page.route("**/api/montage/status", lambda r: r.fulfill(
            status=200, content_type="application/json", body=empty_status))
        page.route("**/api/montage/result/**", lambda r: r.fulfill(
            status=404, content_type="application/json", body='{"detail":"no job"}'))
        page.goto(BASE_URL, wait_until="networkidle")
        page.evaluate("() => switchView('montage')")

        # 1. prepare op is surfaced as the main operation while running.
        inject(page, {"prepare_status": "running", "prepare_operation": {
            "status": "running", "progress": 30, "detail": "Нормализую видео",
            "stages": [
                {"key": "instagram", "label": "Instagram", "status": "running", "progress": 40},
                {"key": "youtube", "label": "YouTube", "status": "pending"},
            ]}})
        check("prepare main title", page.text_content("#operationTitle") == "Подготовка к публикации",
              page.text_content("#operationTitle"))
        check("prepare op rendered", page.locator("#operationsBody .ops-op").count() >= 1)
        check("prepare nested stages", page.locator("#operationsBody .ops-stage").count() >= 2,
              f"count={page.locator('#operationsBody .ops-stage').count()}")

        # 2. publish remains the main op when both publish and prepare are active.
        inject(page, {
            "prepare_status": "running",
            "publish_progress": {"ig": {"label": "Instagram", "progress": 50, "status": "publishing", "detail": "Загрузка"}},
        })
        check("publish wins over prepare", page.text_content("#operationTitle") == "Публикация",
              page.text_content("#operationTitle"))

        # 3. applyPrepareState surfaces readiness but never disables normal targets
        #    (preparation is automatic; publish prepares any missing artefact).
        page.evaluate("""() => {
          window.__opsState = {result: {result_video: 'x.mp4'}, prepare_status: 'done',
            publish_ready: {instagram_feed: true, youtube: true, instagram_trial: true,
                            instagram_feed_mirrored: false, youtube_mirrored: false, instagram_trial_mirrored: false}};
          document.querySelectorAll('.publish-target-checkbox').forEach(c => { c.checked = true; });
          applyPrepareState(window.__opsState);
        }""")
        ready = page.evaluate("() => { const cb = document.querySelector('.publish-target-checkbox[value=\"instagram_feed\"]'); return {d: cb.disabled, c: cb.checked}; }")
        disabled = page.evaluate("() => Array.from(document.querySelectorAll('.publish-target-checkbox:not(.publish-mirror-target)')).filter(c => c.disabled).map(c => c.value)")
        check("ready target enabled+checked", ready["d"] is False and ready["c"] is True, str(ready))
        check("preparation never disables targets", len(disabled) == 0, str(disabled))
        check("prepare info shows done", "подготовлено" in page.text_content("#montagePrepareInfo").lower(),
              page.text_content("#montagePrepareInfo"))

        # 4. level-3 folders of different targets collapse independently.
        inject(page, {"publish_progress": {
            "ig": {"id": "ig", "label": "Instagram", "progress": 50, "status": "publishing", "detail": "Загрузка",
                   "steps": [{"key": "s1", "label": "Нормализация", "status": "completed", "progress": 100},
                             {"key": "s2", "label": "Загрузка", "status": "running", "progress": 20}]},
            "yt": {"id": "yt", "label": "YouTube", "progress": 10, "status": "publishing", "detail": "Загрузка",
                   "steps": [{"key": "s3", "label": "Нормализация", "status": "completed", "progress": 100}]},
        }})
        folds = page.locator("#operationsBody .ops-fold")
        check("two independent folds", folds.count() == 2, f"count={folds.count()}")
        page.evaluate("() => toggleOpsFold('publish:ig')")
        st = page.evaluate("""() => {
          const ig = document.querySelector('.ops-fold[data-fold="publish:ig"]');
          const yt = document.querySelector('.ops-fold[data-fold="publish:yt"]');
          return {ig: ig.classList.contains('closed'), yt: yt.classList.contains('closed')};
        }""")
        check("ig fold closed", st["ig"] is True, str(st))
        check("yt fold stays open", st["yt"] is False, str(st))

        # 5. failed op shows «—», never 100%.
        inject(page, {"operation": {"status": "error", "progress": 100, "label": "Монтаж видео",
                                    "error": "boom", "stages": []}})
        pct = page.evaluate("""() => {
          const ops = [...document.querySelectorAll('#operationsBody .ops-op')];
          const failed = ops.find(o => o.querySelector('.ops-op-error'));
          if (!failed) return null;
          const spans = failed.querySelectorAll('.ops-op-head span');
          return spans[spans.length - 1].textContent.trim();
        }""")
        check("failed op percent is dash", pct == "—", f"pct={pct!r}")

        # 6. running stages expose elapsed time.
        page.evaluate("""() => {
          window.__opsState = {};
          mergeOpsState({operation: {status: 'running', progress: 10, label: 'Монтаж видео',
            stages: [{key: 'render', label: 'Рендер', status: 'running', progress: null,
                      started_at: Math.floor(Date.now()/1000) - 5}]}});
          renderOperations(window.__opsState);
        }""")
        elapsed_ok = page.evaluate("""() => {
          const rows = [...document.querySelectorAll('#operationsBody .ops-stage')];
          return rows.some(r => /\\d+([.,]\\d+)?с|м /.test(r.textContent));
        }""")
        check("running stage shows elapsed", elapsed_ok)

        page.screenshot(path=str(SHOT_DIR / "ops_prepare.png"))
        browser.close()

    print(f"\n=== prepare UI: {PASS} passed, {FAIL} failed ===")
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
