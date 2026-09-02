"""Name each cluster, pick an example review, and persist themes to Postgres.

    python -m src.clustering.name_themes --model sbert-domain
    python -m src.clustering.name_themes --model sbert-domain --dry-run

Naming uses c-TF-IDF. Ordinary TF-IDF scores a word within one document; average
that over a cluster and every theme comes back as "app, order, food", because
those words are everywhere. c-TF-IDF glues each cluster into one pseudo-document
and runs TF-IDF *across clusters*, so a term ranks highly only when it is common
in this theme and rare in the others.

The example review is the one nearest the cluster centroid, measured in the
original embedding space rather than the UMAP space — UMAP distorts distances by
design, so "most typical" should be judged where the meaning actually lives.

Sizes are reported two ways because they answer different questions. n_texts
counts distinct strings; n_rows counts actual reviews, so the 15,576 people who
wrote "good" are counted 15,576 times. n_rows is the one a product manager wants.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)

from src.clustering.cluster import cluster, load_vectors, reduce
from src.db.connection import connection

CORPUS = Path("data/vectors/corpus.parquet")
STOP = "english"


def c_tfidf(texts, labels, top_n=8, min_reviews=25, min_share=0.04):
    """{cluster_id: [term, ...]} — terms distinctive to each cluster.

    Two filters, both learned from what went wrong without them.

    1. The vocabulary is fixed first from the review-level corpus, requiring a
       term in at least min_reviews reviews. Without it c-TF-IDF ranked
       misspellings top — "veri", "niceee", "coustmer", "zamato" are maximally
       distinctive precisely because almost nobody writes them.

    2. A term must also appear in at least min_share of *its own cluster's*
       reviews. c-TF-IDF rewards rarity, so filter 1 alone still surfaced
       "supar", "sarvice", "verry". A usable label needs a term that is both
       distinctive across clusters and common within one.

    Bigrams are included so a label can read "customer care" rather than two
    words that look unrelated side by side.
    """
    counts = CountVectorizer(stop_words=STOP, min_df=min_reviews,
                             ngram_range=(1, 2), binary=True)
    present = counts.fit_transform(texts)          # texts x vocab, 1 = term occurs
    names = counts.get_feature_names_out()

    ids = sorted({int(l) for l in labels if l >= 0})
    joined = [" ".join(t for t, l in zip(texts, labels) if l == cid) for cid in ids]
    matrix = TfidfVectorizer(vocabulary=names,
                             sublinear_tf=True).fit_transform(joined)

    out = {}
    for row, cid in enumerate(ids):
        member = np.flatnonzero(labels == cid)
        share = np.asarray(present[member].mean(axis=0)).ravel()
        scores = matrix[row].toarray().ravel()
        scores = np.where(share >= min_share, scores, 0.0)
        out[cid] = [names[i] for i in np.argsort(-scores)[:top_n] if scores[i] > 0]
    return out


def representatives(vectors, labels):
    """{cluster_id: row_index nearest that cluster's centroid, by cosine}."""
    unit = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    out = {}
    for cid in sorted({int(l) for l in labels if l >= 0}):
        idx = np.flatnonzero(labels == cid)
        centroid = unit[idx].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        out[cid] = int(idx[np.argmax(unit[idx] @ centroid)])
    return out


def build(model, mcs, ms):
    corpus = pd.read_parquet(CORPUS)
    texts = corpus.content.tolist()
    vectors = load_vectors(model)
    labels, probs = cluster(reduce(model), mcs, ms)

    terms = c_tfidf(texts, labels)
    reps = representatives(vectors, labels)

    # Expand distinct texts back to individual review rows.
    assign = corpus[["review_ids"]].copy()
    assign["theme_id"] = labels
    assign["strength"] = probs
    assign = assign.explode("review_ids").rename(columns={"review_ids": "review_id"})

    with connection() as conn:
        scores = pd.read_sql("SELECT review_id, score FROM reviews "
                             "WHERE app='swiggy' AND keep_for_themes", conn)
    assign = assign.merge(scores, on="review_id", how="left")

    rows = []
    for cid, group in assign[assign.theme_id >= 0].groupby("theme_id"):
        top = terms.get(int(cid), [])
        rows.append({
            "model": model, "theme_id": int(cid),
            # A label a person can read. The terms stay in top_terms for anyone
            # who wants the full evidence.
            "label": " / ".join(top[:3]) if top else f"theme {cid}",
            "top_terms": ", ".join(top),
            "n_rows": int(len(group)),
            "n_texts": int((labels == cid).sum()),
            "avg_rating": round(float(group.score.mean()), 2),
            "example_review_id": corpus.review_ids.iloc[reps[int(cid)]][0],
            "params": f"min_cluster_size={mcs},min_samples={ms}",
        })
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False), assign, labels


def persist(themes, assign, model):
    with connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM review_themes WHERE model=%s", (model,))
        cur.execute("DELETE FROM themes WHERE model=%s", (model,))
        execute_values(cur,
            "INSERT INTO themes (model, theme_id, label, top_terms, n_rows, "
            "n_texts, avg_rating, example_review_id, params) VALUES %s",
            [tuple(r) for r in themes[["model", "theme_id", "label", "top_terms",
                                       "n_rows", "n_texts", "avg_rating",
                                       "example_review_id", "params"]].to_numpy()])
        execute_values(cur,
            "INSERT INTO review_themes (app, review_id, model, theme_id, strength) "
            "VALUES %s",
            [("swiggy", r.review_id, model, int(r.theme_id), float(r.strength))
             for r in assign.itertuples()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="sbert-domain")
    ap.add_argument("--min-cluster-size", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    themes, assign, labels = build(args.model, args.min_cluster_size,
                                   args.min_samples)
    noise = float((labels < 0).mean()) * 100
    covered = int((assign.theme_id >= 0).sum())
    print(f"\n{args.model}: {len(themes)} themes · {noise:.1f}% of distinct texts "
          f"are noise · {covered:,} of {len(assign):,} review rows got a theme\n")

    show = themes.head(args.top)[["theme_id", "n_rows", "avg_rating", "label"]]
    print(show.to_string(index=False))

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return
    persist(themes, assign, args.model)
    print(f"\nwrote {len(themes)} themes and {len(assign):,} assignments to Postgres")


if __name__ == "__main__":
    main()
