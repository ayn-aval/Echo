"""Verify the results board loads and reads plainly. Run before claiming it works.

    python -m eval.check_app

Three checks, in order of how quietly each failure hides:

1. **Parse.** Every file under app/ is compiled with ast.parse. This exists
   because `streamlit.testing.v1.AppTest` does NOT report a SyntaxError: it
   prints one to stderr and then returns an AppTest whose `.exception` is None,
   so a broken file reports as passing. Four screens once shipped broken that way.

2. **Render.** app/main.py is run under AppTest, which exercises every figure
   including the database queries behind Figures 04 and 05.

3. **Language.** The rendered text is scanned for emoji and for mathematical
   notation. The second rule is the strict one and it has no exemption: this page
   is read by people who do not know what a correlation coefficient is, and the
   project owner has to be able to defend every number on it in their own words.
   A metric name that survives here is one they would have to explain in an
   interview, so the check fails the build rather than warning.

An HTTP 200 from the Streamlit server proves nothing here — it serves the app
shell before any page code runs, so the route returns 200 even when the page is
broken. That was the other half of an earlier false pass.
"""

import ast
import io
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

APP = Path("app")
MAIN = APP / "main.py"

# Notation and metric names, none of which belong in front of a reader. Each has
# a plain replacement already in use on the page: "Precision@10 of 0.7577" is
# "about 8 of every 10 results are relevant", "a 3.4σ spike" is "59 reviews in a
# week that normally sees about 20".
NOTATION = re.compile(
    r"\bz[- ]?scores?\b|\bstandard deviation|\bspearman|\bpearson|"
    r"precision@\d|recall@\d|\bmrr\b|silhouette|\bcosine|\bcentroid|"
    r"statistically significant|\bfisher\b|\bp\s*[=<]\s*0\.|\bpoisson|"
    r"c-TF-IDF|\bTF-?IDF|HDBSCAN|UMAP|FAISS|cross-encoder|bi-encoder|"
    r"\bembeddings?\b|ρ|σ|\bsigma\b",
    re.I)

# The one place a banned word is correct: the paper's own title, in the citation.
# It is removed before scanning rather than exempted by rule, so a stray
# "embedding" anywhere else still fails.
CITATION = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

# Emoji blocks only. Arrows, the true minus sign and box-drawing characters are
# typography and are deliberately NOT banned: "scraper -> Postgres" and "-21%"
# are correct typesetting, and an earlier, wider range failed the run on the
# stylesheet's own section rules.
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿️]")


def parses() -> list:
    bad = []
    for path in sorted(APP.rglob("*.py")):
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            bad.append(f"{path}:{exc.lineno} {exc.msg}")
    return bad


def renders():
    """(error, visible_text) for the board.

    stderr is captured because AppTest writes a SyntaxError there and reports
    success anyway.
    """
    from streamlit.testing.v1 import AppTest

    sys.path.insert(0, str(APP.resolve()))
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            at = AppTest.from_file(str(MAIN.resolve()), default_timeout=400)
            at.run()
    except Exception as exc:                              # noqa: BLE001
        return f"{type(exc).__name__}: {exc}", ""

    err = at.exception[0].value if at.exception else None
    noise = buf.getvalue()
    if "SyntaxError" in noise or "Traceback" in noise:
        err = err or noise.strip().splitlines()[-1]

    # The stylesheet is itself an st.markdown call, so it lands in at.markdown.
    # Scanning it would mean checking CSS comments for notation and section rules
    # for emoji, which is how "──" once failed every screen at once.
    body = [m.value for m in at.markdown if not m.value.lstrip().startswith("<style")]
    text = " ".join(body + [c.value for c in at.caption] +
                    [b.label for b in at.button])
    return err, text.replace(CITATION, "")


def main() -> None:
    print("1. parse")
    bad = parses()
    for line in bad:
        print(f"   FAIL {line}")
    if bad:
        raise SystemExit(f"\n{len(bad)} file(s) do not parse — fix before rendering.")
    print(f"   ok   {len(list(APP.rglob('*.py')))} files parse")

    print("\n2. render")
    err, text = renders()
    print(f"   {'FAIL' if err else ' ok '} board renders"
          f"{'  ' + str(err) if err else f'  ({len(text.split()):,} words of copy)'}")

    print("\n3. language")
    notation = sorted(set(NOTATION.findall(text)))
    emoji = sorted(set(EMOJI.findall(text)))
    print(f"   {'FAIL' if notation else ' ok '} no notation"
          f"{'  found: ' + str(notation) if notation else ''}")
    print(f"   {'FAIL' if emoji else ' ok '} no emoji"
          f"{'  found: ' + str(emoji) if emoji else ''}")

    if err or notation or emoji:
        raise SystemExit(1)
    print("\nthe board renders clean")


if __name__ == "__main__":
    main()
