"""Photograph every dashboard screen, so the UI can be judged by eye.

    python -m eval.shoot_app                 # all screens
    python -m eval.shoot_app today issues    # just these

Writes a full-page PNG per screen to results/screens/. Two uses: checking a
design change actually looks right before saying it does, and producing the
README screenshots.

This exists because there is no way to verify a *look* from a test. eval/check_app.py
proves each screen parses and renders without raising; it cannot tell that cards
are invisible against the background, that a sparkline is floating outside its
card, or that a chart is captioned in the wrong unit. All three were true of this
app while every automated check passed.

Needs the app already running (`streamlit run app/main.py`) and Google Chrome
installed; Playwright drives the real browser rather than downloading its own.
"""

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "results" / "screens"

# url_path -> filename. The default page is served at "/", not at its url_path,
# so requesting /today returns Streamlit's "Page not found" — that is normal
# behaviour for a page registered with default=True, not a routing bug.
SCREENS = {
    "": "today",
    "alerts": "alerts",
    "volume": "ratings",
    "issues": "topics",
    "changes": "trends",
    "find": "search",
    "how": "how-it-works",
}


def running() -> bool:
    try:
        urllib.request.urlopen(BASE, timeout=4)
        return True
    except (urllib.error.URLError, OSError):
        return False


def main() -> None:
    wanted = sys.argv[1:]
    screens = {p: n for p, n in SCREENS.items() if not wanted or n in wanted or p in wanted}
    if not screens:
        raise SystemExit(f"No screen matched {wanted}. Known: {sorted(SCREENS.values())}")

    if not running():
        raise SystemExit(f"Nothing is serving {BASE}. Start it with:\n"
                         f"    streamlit run app/main.py")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install playwright") from None

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        # A tall viewport, not full_page: Streamlit scrolls an inner container
        # rather than the document, so full_page captures only the first screen
        # and silently crops everything below it.
        page = browser.new_page(viewport={"width": 1600, "height": 2400},
                                device_scale_factor=1)
        for path, name in screens.items():
            page.goto(f"{BASE}/{path}", wait_until="networkidle")
            # Streamlit streams the page over a websocket after the shell loads,
            # so "document ready" is far too early — wait for real content, then
            # let charts finish drawing.
            try:
                page.wait_for_selector("[data-testid='stAppViewContainer'] h1",
                                       timeout=60000)
            except Exception:                                    # noqa: BLE001
                print(f"  {name}: no heading appeared — capturing anyway")
            page.wait_for_timeout(4500)
            shot = OUT / f"{name}.png"
            page.screenshot(path=str(shot), full_page=True)
            print(f"  {name:14} {shot.relative_to(Path.cwd()) if shot.is_relative_to(Path.cwd()) else shot}")
        browser.close()

    print(f"\n{len(screens)} screen(s) written to {OUT}")


if __name__ == "__main__":
    main()
