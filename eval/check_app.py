"""Verify every dashboard screen actually loads. Run before claiming it works.

    python -m eval.check_app

Three checks, in order of how quietly each failure hides:

1. **Parse.** Every file under app/ is compiled with ast.parse. This exists
   because `streamlit.testing.v1.AppTest` does NOT report a SyntaxError: it
   prints one to stderr and then returns an AppTest whose `.exception` is None,
   so a broken file reports as passing. Four screens shipped broken that way.

2. **Render.** Each screen runs under AppTest with stderr captured, and the
   capture is inspected as well as `.exception`, for the same reason.

3. **Language.** The rendered text is scanned for statistics vocabulary and
   emoji, both of which the dashboard is meant to keep out of the reader's way.

An HTTP 200 from the Streamlit server proves nothing here — it serves the app
shell before any page code runs, so every route returns 200 even when every page
is broken. That was the other half of the false pass.
"""

import ast
import io
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

APP = Path("app")
VIEWS = sorted(APP.glob("views/*.py"))
# Method vocabulary belongs on exactly one screen, behind its expander. The
# other three are read by people who do not know what a cluster is, and a single
# leaked term is enough to make the whole page feel like someone's notebook.
JARGON = re.compile(r"\bz[- ]?scores?\b|\bz = |\bz ≥|baseline mean|"
                    r"standard deviation|Precision@\d|silhouette|"
                    r"statistically significant|Fisher|cosine similarity|"
                    r"cross-encoder|bi-encoder|Poisson|p=0\.|c-TF-IDF|"
                    r"HDBSCAN|UMAP|FAISS|embedding|centroid|\bcluster",
                    re.I)
# The one screen whose job is to explain the method. Jargon there is reported
# but does not fail the run.
JARGON_EXEMPT = {"accuracy.py"}
# Arrows (U+2190-U+21FF) are deliberately NOT here. "scraper -> Postgres" is
# typography, not an emoji, and the old range banned it on one screen while
# letting the identical glyph through on another purely because that one
# was written as the HTML entity &rarr;. Everything else in the symbol and
# dingbat blocks stays banned.
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2200-\u2BFF\u2600-\u27BF\uFE0F]")


def parses() -> list:
    bad = []
    for path in sorted(APP.rglob("*.py")):
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            bad.append(f"{path}:{exc.lineno} {exc.msg}")
    return bad


def renders(path: Path):
    """(error, visible_text). stderr is captured because AppTest writes a
    SyntaxError there and reports success anyway."""
    from streamlit.testing.v1 import AppTest
    sys.path.insert(0, str(APP.resolve()))
    from filters import Filters

    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            # Absolute: AppTest resolves a relative path against the file
            # that calls it, which is this module in eval/, not the caller's cwd.
            at = AppTest.from_file(str(path.resolve()), default_timeout=400)
            at.session_state["filters"] = Filters("All time", (), None)
            at.run()
    except Exception as exc:                      # noqa: BLE001
        return f"{type(exc).__name__}: {exc}", ""

    err = at.exception[0].value if at.exception else None
    if err and "url_pathname" in str(err):
        err = None          # st.page_link only resolves inside the nav shell
    noise = buf.getvalue()
    if "SyntaxError" in noise or "Traceback" in noise:
        err = err or noise.strip().splitlines()[-1]
    text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    return err, text


def main() -> None:
    print("1. parse")
    bad = parses()
    for line in bad:
        print(f"   FAIL {line}")
    if bad:
        raise SystemExit(f"\n{len(bad)} file(s) do not parse — fix before rendering.")
    print(f"   ok   {len(list(APP.rglob('*.py')))} files parse")

    print("\n2. render + 3. language")
    failures = 0
    for path in VIEWS:
        err, text = renders(path)
        jargon = sorted(set(JARGON.findall(text)))
        emoji = sorted(set(EMOJI.findall(text)))
        exempt = path.name in JARGON_EXEMPT
        # Emoji fail everywhere; jargon fails everywhere except the screen whose
        # subject is the method.
        broke = bool(err) or bool(emoji) or (bool(jargon) and not exempt)
        status = "FAIL" if broke else ("note" if jargon else " ok ")
        failures += broke
        detail = err or ("emoji=" + str(emoji) if emoji else "") \
            or ("jargon=" + str(jargon) if jargon else "")
        print(f"   {status} {path.name:14} {detail}")

    print(f"\n{len(VIEWS) - failures}/{len(VIEWS)} screens render clean")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
