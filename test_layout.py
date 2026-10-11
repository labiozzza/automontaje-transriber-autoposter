"""Responsive layout checks + screenshots for the montage view.

Verifies mobile / tablet / desktop have no horizontal overflow and that the
reorganised 8-section workflow stacks correctly. Screenshots are written to
tests_artifacts/screens/.

Run (server on :8000):
  python test_layout.py
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000"
ART = Path(__file__).parent / "tests_artifacts" / "screens"
ART.mkdir(parents=True, exist_ok=True)

VIEWPORTS = [("mobile", 375, 812), ("tablet", 768, 1024), ("desktop", 1440, 900)]

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
    with sync_playwright() as p:
        browser = p.chromium.launch()
        errors = []
        for label, w, h in VIEWPORTS:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.on("pageerror", lambda e, acc=errors: acc.append(str(e)))
            page.goto(BASE_URL, wait_until="networkidle")
            page.evaluate("() => switchView('montage')")
            page.wait_for_timeout(300)

            overflow = page.evaluate(
                "() => ({doc: document.documentElement.scrollWidth, win: window.innerWidth})"
            )
            check(f"{label}: no horizontal overflow",
                  overflow["doc"] <= overflow["win"] + 2,
                  f"scrollWidth={overflow['doc']} innerWidth={overflow['win']}")

            sections = page.locator(".montage-section").count()
            check(f"{label}: 8 sections present", sections == 8, f"sections={sections}")

            wide = page.evaluate(
                """() => {
                  const w = window.innerWidth + 2;
                  const bad = [];
                  document.querySelectorAll('#viewMontage *').forEach(el => {
                    const r = el.getBoundingClientRect();
                    if (r.width > w) bad.push(el.id || el.className || el.tagName);
                  });
                  return bad.slice(0, 5);
                }"""
            )
            check(f"{label}: no oversized elements", len(wide) == 0, f"{wide}")

            page.screenshot(path=str(ART / f"montage_{label}.png"), full_page=False)
            page.close()
        browser.close()

    print(f"\n=== layout: {PASS} passed, {FAIL} failed ===")
    if errors:
        print("PAGE ERRORS:", errors)
    if FAILURES:
        for f in FAILURES:
            print(" -", f)
    return 1 if (FAIL or errors) else 0


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    os._exit(_code)
