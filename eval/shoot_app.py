"""Photograph the results board, so the UI can be judged by eye.

    python -m eval.shoot_app

Writes results/screens/board-1.png … board-4.png (the page in vertical sections)
plus board-search.png (Figure 07 with a real query run through the model). Two
uses: checking a design change actually looks right before saying it does, and
producing the README screenshots.

This exists because there is no way to verify a *look* from a test.
eval/check_app.py proves the page renders without raising; it cannot tell that a
band has lost its margin, that a chart is running off the right edge, or that a
label is sitting on top of the buttons below it. All three were true of the
supplied reference design while it served a perfectly healthy HTTP 200.

Three structural assertions run before any photograph is taken, because each
guards a failure that looks fine in a screenshot taken at the wrong moment.

Needs the app already running (`streamlit run app/main.py`).
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "results" / "screens"

# The keyed containers the layout depends on, and the least each must hold.
KEYED = {"split_topics": 3, "padr_topics": 3, "chips_series": 3,
         "band_search": 4, "chips_examples": 3}

DEMO_QUERY = "refund never came back to my account"

SCROLL = """() => document.querySelector('section.main')
            || document.querySelector('[data-testid="stMain"]')
            || document.scrollingElement"""


def running() -> bool:
    try:
        urllib.request.urlopen(BASE, timeout=4)
        return True
    except (urllib.error.URLError, OSError):
        return False


def containers_wrap(page) -> str:
    """Each keyed container really does wrap its widgets.

    The reference design this is built from styles its bands by opening a plain
    <div class="band">, and Streamlit closes that div before any widget is
    emitted — its own DOM reported two bands holding zero children, so the
    padding never applied and every figure sat flush against the window edge.
    st.container(key=...) is the supported way; this asserts it still works
    rather than assuming it.
    """
    for key, least in KEYED.items():
        n = page.eval_on_selector_all(
            f'[class*="st-key-{key}"] button, [class*="st-key-{key}"] input',
            "e => e.length")
        if n < least:
            return f'.st-key-{key} wraps {n} widgets, expected at least {least}'
    return ""


def nothing_overflows(page) -> str:
    """No element runs past the right edge.

    In the reference, Figure 02's scores and Figure 05's chart were both clipped
    off-screen because their padding wrapper was empty. Nothing in a screenshot
    of the left-hand column would have shown it.
    """
    width = page.evaluate("() => document.documentElement.clientWidth")
    over = page.evaluate("""(w) => [...document.querySelectorAll('*')]
        .filter(e => { const r = e.getBoundingClientRect();
                       return r.right > w + 1 && r.width > 20 && r.height > 0; })
        .slice(0, 4).map(e => (e.className || e.tagName).toString().slice(0, 40))""",
                         width)
    return f"{len(over)} element(s) past {width}px: {over}" if over else ""


def bands_align(page) -> str:
    """Every full-width band starts on the same left margin.

    A band whose wrapper collapsed keeps rendering — it just loses its padding,
    so its content jumps to x=0 while the bands around it stay at 56. That ragged
    edge is the single most visible symptom of the bug above.
    """
    xs = page.eval_on_selector_all(
        ".figlab .lab, .topbar .mark",
        "els => els.map(e => Math.round(e.getBoundingClientRect().x))")
    flush = [x for x in xs if x == 56]
    if len(flush) < 5:
        return f"only {len(flush)} of {len(xs)} band labels sit at x=56: {xs}"
    return ""


def main() -> None:
    if not running():
        raise SystemExit(f"Nothing is serving {BASE}. Start it with:\n"
                         f"    streamlit run app/main.py")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install playwright") from None

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(BASE, wait_until="networkidle")
        # Streamlit streams the page over a websocket after the shell loads, so
        # "document ready" is far too early — wait for real content.
        page.wait_for_selector(".topbar", timeout=60000)
        page.wait_for_timeout(3500)

        print("structure")
        for check in (containers_wrap, nothing_overflows, bands_align):
            problem = check(page)
            print(f"  {'FAIL' if problem else ' ok '} {check.__doc__.splitlines()[0]}"
                  f"{'  — ' + problem if problem else ''}")
            if problem:
                raise SystemExit(1)

        print("\nphotographs")
        height = page.evaluate(f"() => ({SCROLL})().scrollHeight")
        y, i = 0, 1
        while y < height and i <= 6:
            page.evaluate(f"(y) => {{ ({SCROLL})().scrollTop = y; }}", y)
            page.wait_for_timeout(900)
            shot = OUT / f"board-{i}.png"
            page.screenshot(path=str(shot))
            print(f"  board-{i}.png")
            y, i = y + 940, i + 1

        # Figure 07 runs the encoder and the reranker for real. The first call
        # loads a 90 MB cross-encoder, so this waits on the result rather than
        # on a fixed delay, which photographed the loading spinner instead.
        page.fill('[class*="st-key-band_search"] input', DEMO_QUERY)
        page.press('[class*="st-key-band_search"] input', "Enter")
        # Wait on a rendered result row, not on text. "text=of these" matched the
        # "Or try one of these" label above the box and returned instantly,
        # photographing the loading spinner.
        page.wait_for_selector(".review", timeout=180000)
        page.wait_for_timeout(2000)
        page.evaluate(f"() => {{ const e = ({SCROLL})(); e.scrollTop = e.scrollHeight; }}")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "board-search.png"))
        print("  board-search.png")
        browser.close()

    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
