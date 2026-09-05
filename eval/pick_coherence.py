"""Find real complaints the model groups together despite sharing no words.

    python -m eval.pick_coherence
    python -m eval.pick_coherence --theme 105 --min-similarity 0.75

Figure 03 of the results board claims the model reads meaning rather than
matching words. That claim needs evidence that was *found* under a stated rule,
not picked because it flattered the model. The rule:

    among complaint reviews the model put in the same topic, keep the triples
    that share no topic word at all, and return whichever of those the model
    placed closest together.

Ranking that way round matters. Maximising closeness first returns near-restatements
of one sentence, which prove nothing: the interesting evidence is the *most* similar
triple that still has no vocabulary in common.

Writes results/coherence_example.csv, including the true shared-word count, so
the figure reports the overlap it actually has rather than claiming zero.
"""

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.connection import connection

VECTORS = Path("data/vectors")
OUT = Path("results/coherence_example.csv")
MODEL = "sbert-domain"

MIN_WORDS = 6         # a very short review shares no words by accident, not meaning
MIN_SIMILARITY = 0.60 # a floor; the rule then maximises closeness
NEIGHBOURS = 60       # candidate partners per reference review
TOP_THEMES = 40       # search the largest complaint topics
MAX_STARS = 2         # complaints only; the rest of the board is about complaints

# Function words carry no topic, so overlapping on them is not "sharing a word".
# `app`, `application` and `swiggy` join them: they name the subject of every
# review in the corpus, so two reviews both using them have not shared anything.
STOP = set("""
a about after again all also am an and any are as at be because been before being
but by can cant cannot could did do does doing done dont for from get got had has
have he her here him his how i if in into is it its just like me more most my no
not now of off on once one only or other our out over own re said same she should
so some such than that the their them then there these they this those through to
too under until up very was we were what when where which while who why will with
would you your yours it's i'm ive is app application swiggy
""".split())

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")


SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ion",
            "ers", "er", "est", "ed", "es", "ly", "en", "s", "y", "e")


def stem(word: str) -> str:
    """Collapse a word to a crude stem, biased towards over-collapsing.

    No stemmer ships with this project's dependencies and adding one is not worth
    a new requirement, so this strips the common English endings and truncates.
    The bias is deliberate: the figure claims three reviews share no words, so a
    stem that merges two words that a reader would call different is safe, while
    one that keeps "taking" and "takes" apart would overstate the claim.
    """
    for _ in range(3):          # "ordering" needs two passes: -ing, then -er
        for suffix in SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                break
        else:
            break
    return word.rstrip("e")[:4] or word[:4]


def content_words(text: str) -> set[str]:
    """The topic-bearing words of a review, as stems."""
    words = re.findall(r"[a-z]+|[0-9]+", str(text).lower())
    # Any number counts, however short: three reviews all saying "45 mins" have
    # shared something, and a two-character token would fail a length floor.
    return {stem(w) for w in words
            if (w.isdigit() or len(w) >= 3) and w not in STOP}


def load():
    corpus = pd.read_parquet(VECTORS / "corpus.parquet")
    vectors = np.load(VECTORS / f"{MODEL}.npy")
    if len(corpus) != len(vectors):
        raise SystemExit(f"corpus {len(corpus)} != vectors {len(vectors)} — re-encode")
    vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)

    with connection() as conn:
        rows = pd.read_sql("""
            SELECT rt.review_id, rt.theme_id, r.score AS stars,
                   coalesce(t.display_name, t.label) AS theme
              FROM review_themes rt
              JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
              JOIN reviews r ON r.review_id = rt.review_id
             WHERE rt.model = %s AND t.actionable AND t.avg_rating <= 2.5
               AND r.score <= %s""", conn, params=(MODEL, MAX_STARS))
    return corpus, vectors, rows


def eligible(i: int, texts: list) -> bool:
    text = str(texts[i])
    return (len(text.split()) >= MIN_WORDS
            and not EMOJI.search(text)
            and len(content_words(text)) >= 3)


def best_triple(idx, vectors, texts, floor):
    """Fewest shared words among triples that clear the similarity floor.

    Searching every triple is O(n^3) and a topic holds up to 1,950 reviews, so
    each reference review only considers its own nearest neighbours. A triple all
    of whose pairs are close is one where the two partners are both near the
    reference, so nothing that could win is missed.
    """
    idx = [i for i in idx if eligible(i, texts)]
    if len(idx) < 3:
        return None

    words = {i: content_words(texts[i]) for i in idx}
    sims = vectors[idx] @ vectors[idx].T
    np.fill_diagonal(sims, -1)
    k = min(NEIGHBOURS, len(idx) - 1)

    best = None
    for a in range(len(idx)):
        near = [n for n in np.argpartition(sims[a], -k)[-k:] if sims[a, n] >= floor]
        for b, c in combinations(near, 2):
            if sims[b, c] < floor:
                continue
            pairs = ((a, b), (a, c), (b, c))
            shared = max(len(words[idx[x]] & words[idx[y]]) for x, y in pairs)
            lo = min(sims[x, y] for x, y in pairs)
            score = (shared, -lo)          # fewest shared words, then closest
            if best is None or score < best[0]:
                best = (score, [idx[a], idx[b], idx[c]],
                        float(sims[a, b]), float(sims[a, c]), shared)
    return best


def main(theme_filter=None, floor=MIN_SIMILARITY):
    corpus, vectors, rows = load()
    texts = corpus.content.tolist()
    # One vector serves every row sharing a text, so each review id has to
    # resolve to the row holding its distinct string.
    row_of = {rid: i for i, ids in enumerate(corpus.review_ids) for rid in ids}
    rows = rows[rows.review_id.isin(row_of)]

    candidates = ([theme_filter] if theme_filter else
                  rows.theme_id.value_counts().head(TOP_THEMES).index.tolist())

    overall = None
    for tid in candidates:
        theme_rows = rows[rows.theme_id == tid]
        found = best_triple(sorted({row_of[r] for r in theme_rows.review_id}),
                            vectors, texts, floor)
        if found and (overall is None or found[0] < overall[0]):
            overall = (*found, tid, theme_rows.theme.iloc[0], len(theme_rows))

    if overall is None:
        raise SystemExit(f"No triple reached {floor}. Lower --min-similarity and rerun.")

    _, picked, ab, ac, shared, tid, name, n_rows = overall
    ids = [corpus.review_ids.iloc[i][0] for i in picked]
    with connection() as conn:
        meta = pd.read_sql(
            "SELECT review_id, score AS stars FROM reviews WHERE review_id = ANY(%s)",
            conn, params=(ids,))
    stars = dict(zip(meta.review_id, meta.stars))

    out = pd.DataFrame([
        dict(position=p, review_id=r, stars=int(stars.get(r, 0)), content=texts[i],
             similarity_to_first=round(s, 4), theme_id=int(tid), theme=name,
             theme_reviews=int(n_rows), max_shared_topic_words=int(shared))
        for p, (i, r, s) in enumerate(zip(picked, ids, [1.0, ab, ac]), 1)])

    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"topic {tid} · {name} · {n_rows} complaint reviews")
    print(f"most topic words any pair shares: {shared}\n")
    for r in out.itertuples():
        print(f"  {r.position}. [{r.stars} star]  similarity {r.similarity_to_first:.2f}")
        print(f"     {r.content}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", type=int, default=None)
    ap.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    a = ap.parse_args()
    main(a.theme, a.min_similarity)
