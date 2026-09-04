"""Verify every dashboard screen actually loads. Run before claiming it works.

    python -m eval.check_app

Three checks, in order of how quietly each failure hides:

1. **Parse.** Every file under app/ is compiled with ast.parse. This exists
   because `streamlit.testing.v1.AppTest` does NOT report a SyntaxError: it
   prints one to stderr and then returns an AppTest whose `.exception` is None,
   so a broken file reports as passing. Four screens shipped broken that way.

2. **Render.** Screens are no longer standalone scripts — they are modules
   exposing `render()`, reached through a session-state router — so each is
   exercised by running app/main.py with `screen` preset. That also tests the
   router itself, which running the view files directly never did.

3. **Language.** The rendered text is scanned for statistics vocabulary and
   emoji. Method words are allowed on exactly one screen, the one whose subject
   IS the method; anywhere else they fail the run.

An HTTP 200 from the Streamlit server proves nothing here — it serves the app
shell before any page code runs, so every route returns 200 even when every
screen is broken. That was the other half of the false pass.
"""

import ast
import io
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

APP = Path("app")
MAIN = APP / "main.py"

# The four screens, as the router keys them.
SCREENS = {"week": "This week", "fix": "The fix list",
           "ask": "Ask anything", "trust": "Can I trust it"}

# Method vocabulary belongs on exactly one screen, behind its expander. The
# other three are read by people who do not know what a cluster is, and a single
# leaked term is enough to make the whole page feel like someone's notebook.
JARGON = re.compile(r"\bz[- ]?scores?\b|\bz = |\bz ≥|baseline mean|"
                    r"standard deviation|Precision@\d|silhouette|"
                    r"statistically significant|Fisher|cosine similarity|"
                    r"cross-encoder|bi-encoder|Poisson|p=0\.|c-TF-IDF|"
                    r"HDBSCAN|UMAP|FAISS|embedding|centroid|\bcluster",
                    re.I)
JARGON_EXEMPT = {"trust"}

# Emoji blocks only. Arrows, the true minus sign and box-drawing characters are
# typography and are deliberately NOT banned: "scraper -> Postgres" and "-21%"
# are correct typesetting, and an earlier, wider range failed the run on the
# stylesheet's own section rules.
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")


def parses() -> list:
    bad = []
    for path in sorted(APP.rglob("*.py")):
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            bad.append(f"{path}:{exc.lineno} {exc.msg}")
    return bad


def renders(screen: str):
    """(error, visible_text) for one screen, driven through the real router.

    stderr is captured because AppTest writes a SyntaxError there and reports
    success anyway.
    """
    from streamlit.testing.v1 import AppTest

    sys.path.insert(0, str(APP.resolve()))
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            at = AppTest.from_file(str(MAIN.resolve()), default_timeout=400)
            at.session_state["screen"] = screen
            at.run()
    except Exception as exc:                              # noqa: BLE001
        return f"{type(exc).__name__}: {exc}", ""

    err = at.exception[0].value if at.exception else None
    noise = buf.getvalue()
    if "SyntaxError" in noise or "Traceback" in noise:
        err = err or noise.strip().splitlines()[-1]
    # The stylesheet is itself an st.markdown call, so it lands in at.markdown.
    # Scanning it means checking CSS comments for jargon and section rules for
    # emoji, which is how "──" once failed every screen at once.
    body = [m.value for m in at.markdown if not m.value.lstrip().startswith("<style")]
    text = " ".join(body + [c.value for c in at.caption])
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
    for key, label in SCREENS.items():
        err, text = renders(key)
        jargon = sorted(set(JARGON.findall(text)))
        emoji = sorted(set(EMOJI.findall(text)))
        exempt = key in JARGON_EXEMPT
        broke = bool(err) or bool(emoji) or (bool(jargon) and not exempt)
        status = "FAIL" if broke else ("note" if jargon else " ok ")
        failures += broke
        detail = err or ("emoji=" + str(emoji) if emoji else "") \
            or ("jargon=" + str(jargon) if jargon else "")
        print(f"   {status} {label:16} {detail}")

    print(f"\n{len(SCREENS) - failures}/{len(SCREENS)} screens render clean")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
