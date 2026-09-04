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

# Tab label -> output filename. There are no URLs any more: navigation is a
# session-state router with a top tab row, so a screen is reached by clicking
# its tab rather than by visiting a path.
SCREENS = {
    "This week": "this-week",
    "The fix list": "fix-list",
    "Ask anything": "ask",
    "Can I trust it": "trust",
}


def running() -> bool:
    try:
        urllib.request.urlopen(BASE, timeout=4)
        return True
    except (urllib.error.URLError, OSError):
        return False


def tabs_work(page) -> str:
    """Every tab is present and actually switches the screen.

    This replaces the old sidebar collapse/reopen assertion, which went away
    with the sidebar. The failure it guards against is the same shape: chrome
    that renders but does not function, which no render check can see.

    Returns "" on success, or a description of the failure.
    """
    for label in SCREENS:
        if not page.query_selector(f"button:has-text('{label}')"):
            return f"no tab labelled {label!r}"

    page.click("button:has-text('Can I trust it')")
    page.wait_for_timeout(2500)
    heading = page.eval_on_selector("h1", "e => e.textContent") if \
        page.query_selector("h1") else ""
    if "defend" not in heading.lower():
        return f"clicking a tab did not change the screen (h1 = {heading!r})"

    page.click("button:has-text('This week')")
    page.wait_for_timeout(2500)
    return ""


def containers_wrap(page) -> str:
    """The keyed container really does wrap its widgets.

    The reference design this is built from styles its tab row by opening a
    plain <div class="navrow">, and Streamlit closes that div before any button
    is emitted — its own DOM reports one .navrow holding zero buttons, so the
    tab styling never applied and nobody noticed. st.container(key=...) is the
    supported way; this asserts it is still working rather than assuming it.
    """
    n = page.eval_on_selector_all(".st-key-tabs .stButton", "e => e.length")
    if n < 4:
        return f".st-key-tabs wraps {n} buttons, expected 4"
    return ""


def main() -> None:
    wanted = sys.argv[1:]
    screens = {p: n for p, n in SCREENS.items() if not wanted or n in wanted}
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
        page = browser.new_page(viewport={"width": 1600, "height": 3000},
                                device_scale_factor=1)
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector(".ink", timeout=60000)
        page.wait_for_timeout(3000)

        for check in (containers_wrap, tabs_work):
            problem = check(page)
            label = {"containers_wrap": "keyed containers wrap their widgets",
                     "tabs_work": "every tab is present and switches screen"}[check.__name__]
            print(f"  {'FAIL' if problem else ' ok '} {label}"
                  f"{'  — ' + problem if problem else ''}")
            if problem:
                raise SystemExit(1)

        for label, name in screens.items():
            page.click(f"button:has-text('{label}')")
            # Streamlit streams the page over a websocket after the shell loads,
            # so "document ready" is far too early — wait for real content, then
            # let charts finish drawing.
            page.wait_for_timeout(1500)
            page.wait_for_timeout(4500)
            shot = OUT / f"{name}.png"
            page.screenshot(path=str(shot), full_page=True)
            print(f"  {name:14} {shot.relative_to(Path.cwd()) if shot.is_relative_to(Path.cwd()) else shot}")
        browser.close()

    print(f"\n{len(screens)} screen(s) written to {OUT}")


if __name__ == "__main__":
    main()
