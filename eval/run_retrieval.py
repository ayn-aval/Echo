"""Recall@10 and MRR on the hand-labelled review retrieval set.

    python -m eval.run_retrieval

Mirrors the STS harness: evaluate_retrieval() takes the same
encode(list[str]) -> ndarray callable, so Phases 3 and 4 measure themselves the
same way these baselines do.

Two metrics, because they answer different questions:

  Recall@10  of the reviews you marked relevant, what share appear in the top 10.
             "Did it find them?"
  MRR        1 / (rank of the first relevant review), averaged over queries.
             "Did it put a good one at the top?" A hit at rank 1 scores 1.0,
             at rank 10 scores 0.1.

Scoring assumptions, stated because they bound what these numbers mean:
  * Reviews outside the labelled pool count as non-relevant. Standard for pooled
    test collections, and it makes every score a slight underestimate.
  * Queries with zero relevant judgements are excluded — Recall has no denominator.
  * Retrieval runs over distinct review texts, not rows, matching the pool.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from eval.build_pool import cached, load_corpus
from src.db.connection import connection
from src.embeddings import baselines, glove, sbert

logging.getLogger("transformers").setLevel(logging.ERROR)

RESULTS = Path("results/baselines.csv")
K = 10


def load_judgements():
    with connection() as conn:
        j = pd.read_sql("""SELECT query_id, review_id FROM eval_judgements
                            WHERE relevant""", conn)
        q = pd.read_sql("SELECT query_id, query_text FROM eval_queries "
                        "ORDER BY query_id", conn)
    return q, j.groupby("query_id").review_id.apply(set).to_dict()


def score(sims, docs, queries, relevant, k=K):
    """sims: (n_queries, n_docs) similarity matrix."""
    recalls, precisions, rrs, ceilings, used = [], [], [], [], 0
    for i, qid in enumerate(queries.query_id):
        gold = relevant.get(qid)
        if not gold:
            continue                      # nothing marked relevant: unscoreable
        used += 1
        order = np.argsort(-sims[i])
        ranked = docs.review_id.to_numpy()[order]

        top = set(ranked[:k])
        hits = len(top & gold)
        recalls.append(hits / len(gold))
        precisions.append(hits / k)
        # With a median of 16 relevant reviews per query you cannot fit them all
        # into a top-10, so Recall@10 has a hard ceiling below 1. Reporting the
        # ceiling stops a "low" recall being read as a weak model.
        ceilings.append(min(k, len(gold)) / len(gold))

        hit = next((r for r, rid in enumerate(ranked, 1) if rid in gold), None)
        rrs.append(1.0 / hit if hit else 0.0)

    pct = lambda xs: round(float(np.mean(xs)) * 100, 2)
    return {"queries_scored": used,
            f"recall@{k}": pct(recalls),
            f"max_possible_recall@{k}": pct(ceilings),
            f"precision@{k}": pct(precisions),
            "mrr": pct(rrs)}


def unit(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


def evaluate_retrieval(encode, name, docs=None, k=K):
    """The reusable entry point — same interface as evaluate_sts()."""
    docs = load_corpus() if docs is None else docs
    queries, relevant = load_judgements()
    doc_vecs = encode(docs.content.tolist())
    sims = unit(encode(queries.query_text.tolist())) @ unit(doc_vecs).T
    return {"model": name, **score(sims, docs, queries, relevant, k)}


def main() -> None:
    docs = load_corpus()
    queries, relevant = load_judgements()
    if not relevant:
        print("No relevance judgements yet. Run:  streamlit run app/label.py")
        return
    print(f"{len(docs):,} distinct review texts · {len(relevant)} queries with "
          f"at least one relevant review\n")
    texts, qtexts = docs.content.tolist(), queries.query_text.tolist()

    rows = []

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    doc_m = tfidf.fit_transform(texts)
    sims = (tfidf.transform(qtexts) @ doc_m.T).toarray()
    rows.append({"model": "tfidf", **score(sims, docs, queries, relevant)})

    g_encode = glove.make_encoder(texts + qtexts, "reviews")
    d_glove = cached("reviews_glove", lambda: g_encode(texts))
    rows.append({"model": "glove-avg",
                 **score(unit(g_encode(qtexts)) @ unit(d_glove).T,
                         docs, queries, relevant)})

    b_encode = baselines.make_encoder("mean")
    d_bert = cached("reviews_bert_mean", lambda: b_encode(texts))
    rows.append({"model": "bert-mean",
                 **score(unit(b_encode(qtexts)) @ unit(d_bert).T,
                         docs, queries, relevant)})

    # One row per trained encoder on disk. Same encode() interface as every
    # baseline above — that is the whole point of building the harness first.
    trained = sbert.available()
    if not trained:
        print("(no trained encoder on disk — baselines only)")
    for name, (path, cache) in trained.items():
        s_encode = sbert.make_encoder(path)
        d_sbert = cached(cache, lambda e=s_encode: e(texts))
        rows.append({"model": name,
                     **score(unit(s_encode(qtexts)) @ unit(d_sbert).T,
                             docs, queries, relevant)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    tidy = df.melt(id_vars=["model", "queries_scored"],
                   var_name="dataset", value_name="spearman")
    tidy["task"] = "retrieval"
    tidy = tidy.rename(columns={"spearman": "score", "queries_scored": "n_pairs"})

    existing = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
    existing = existing[existing.get("task", "sts") != "retrieval"] if len(existing) else existing
    combined = pd.concat([existing, tidy], ignore_index=True)
    combined.to_csv(RESULTS, index=False)
    print(f"\nappended to {RESULTS}")


if __name__ == "__main__":
    main()
