"""Build the candidate pool for the retrieval evaluation.

You cannot label 50 queries against 64,280 reviews — that is 3.2 million
judgements. Pooling is how TREC has built test collections since 1992: take the
top results from several *different* retrieval systems, merge them, and label only
that pool. Anything outside it is treated as non-relevant.

The three systems are deliberately unalike:

  tfidf   pure word overlap — finds reviews that reuse the query's words
  glove   averaged word vectors — loose meaning, no word order
  bert    mean-pooled bert-base-uncased — different failure modes again

Pooling from a lexical system alone would build a test set that rewards string
matching, which is exactly what this project argues against.

Every trained model in src.embeddings.sbert.TRAINED is pooled in too, on top of
the three above. This is not optional. A model trained after the pool was built
retrieves reviews the earlier systems never surfaced — semantically close but
sharing no words — and those score as misses purely because nobody looked at
them. In Phase 3 that alone made the trained model look worse than untrained
BERT. Run with --augment after training anything new, then label the additions.

    python -m eval.build_pool --augment
"""

import logging
from pathlib import Path

# faiss MUST load before scikit-learn: both link their own OpenMP runtime and on
# macOS the wrong order segfaults the process (exit 139, no traceback) at the
# first faiss call. Only matters for --with-rerank, but the import is
# unconditional so the order cannot depend on a flag.
import faiss  # noqa: F401  (imported for load order, not used directly here)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.db.connection import connection
from src.embeddings import baselines, glove, sbert

logging.getLogger("transformers").setLevel(logging.ERROR)

CACHE = Path("data/embeddings")
TOP_PER_SYSTEM = 10


def load_corpus():
    """One row per distinct review text, with a representative review_id.

    Retrieval runs over distinct texts rather than rows: 64,280 kept reviews
    contain only 45,864 distinct strings, and judging the same sentence five
    times would waste the labelling budget. Scores are mapped back to rows later.
    """
    with connection() as conn:
        return pd.read_sql("""
            SELECT min(review_id) AS review_id, app, content
              FROM reviews
             WHERE app='swiggy' AND keep_for_themes
             GROUP BY app, content
             ORDER BY 1""", conn)


def cached(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists():
        print(f"  {name}: cached")
        return np.load(path)
    vectors = build()
    np.save(path, vectors)
    print(f"  {name}: {vectors.shape} -> {path}")
    return vectors


def top_k(query_vecs, doc_vecs, k):
    def unit(m):
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)
    sims = unit(query_vecs.astype(np.float32)) @ unit(doc_vecs.astype(np.float32)).T
    return np.argsort(-sims, axis=1)[:, :k]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--augment", action="store_true",
                    help="add a newly trained model's candidates to the existing "
                         "pool instead of rebuilding it from scratch")
    ap.add_argument("--with-rerank", action="store_true",
                    help="also pool the Phase 6 two-stage searcher (FAISS top-50 "
                         "reranked by a cross-encoder). Required before its "
                         "accuracy can be compared to anything.")
    args = ap.parse_args()
    docs = load_corpus()
    with connection() as conn:
        queries = pd.read_sql("SELECT query_id, query_text FROM eval_queries "
                              "ORDER BY query_id", conn)
    texts = docs.content.tolist()
    qtexts = queries.query_text.tolist()
    print(f"{len(docs):,} distinct review texts, {len(queries)} queries\n")

    hits = {}  # (query_id, review_id) -> set of system names

    print("tfidf")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    doc_m = tfidf.fit_transform(texts)
    q_m = tfidf.transform(qtexts)
    sims = (q_m @ doc_m.T).toarray()
    ranks = {"tfidf": np.argsort(-sims, axis=1)[:, :TOP_PER_SYSTEM]}

    print("glove")
    g_encode = glove.make_encoder(texts + qtexts, "reviews")
    d_glove = cached("reviews_glove", lambda: g_encode(texts))
    ranks["glove"] = top_k(g_encode(qtexts), d_glove, TOP_PER_SYSTEM)

    print("bert")
    b_encode = baselines.make_encoder("mean")
    d_bert = cached("reviews_bert_mean", lambda: b_encode(texts))
    ranks["bert"] = top_k(b_encode(qtexts), d_bert, TOP_PER_SYSTEM)

    # A model trained after the pool was built retrieves reviews the original
    # three never surfaced — semantically close but sharing no words. Those score
    # as misses purely because nobody looked at them, so it has to be pooled in
    # and judged before its numbers mean anything.
    for name, (path, cache) in sbert.available().items():
        print(name)
        s_encode = sbert.make_encoder(path)
        d_sbert = cached(cache, lambda e=s_encode: e(texts))
        ranks[name] = top_k(s_encode(qtexts), d_sbert, TOP_PER_SYSTEM)

    # The two-stage searcher is a genuinely different retrieval system: it
    # promotes reviews from ranks 11-50 that no bi-encoder ever put in its top
    # 10. Measured before pooling, 47.3% of its top-10 had never been judged
    # against FAISS's 96.2%, so its accuracy looked 20 points worse than it may
    # be. Same failure as the Phase 3 lesson, different system.
    if args.with_rerank:
        print("rerank (faiss top-50 + cross-encoder)")
        from src.search.query import embed_query, get_searcher
        from src.search.rerank import CANDIDATES, MODEL, get_cross_encoder
        index, corpus, encode, _ = get_searcher("sbert-domain", approximate=False)
        cross = get_cross_encoder(MODEL)
        picked = []
        for text in qtexts:
            cand = index.search(embed_query(text, encode), CANDIDATES)[1][0]
            scores = np.asarray(cross.predict(
                [(text, str(corpus.content.iloc[i])) for i in cand],
                show_progress_bar=False))
            picked.append(cand[np.argsort(-scores)][:TOP_PER_SYSTEM])
        ranks["rerank"] = np.vstack(picked)

    for system, idx in ranks.items():
        for qi, row in enumerate(idx):
            for di in row:
                key = (int(queries.query_id[qi]), docs.review_id[di])
                hits.setdefault(key, set()).add(system)

    with connection() as conn, conn.cursor() as cur:
        if not args.augment:
            cur.execute("DELETE FROM eval_pool")
        for (qid, rid), systems in hits.items():
            cur.execute("INSERT INTO eval_pool (query_id, app, review_id, sources) "
                        "VALUES (%s,'swiggy',%s,%s) ON CONFLICT DO NOTHING",
                        (qid, rid, ",".join(sorted(systems))))
        cur.execute("SELECT count(*), count(DISTINCT query_id) FROM eval_pool")
        n, nq = cur.fetchone()

    print(f"\npool: {n:,} candidates across {nq} queries "
          f"({n / max(nq, 1):.1f} per query)")
    overlap = pd.Series([len(s) for s in hits.values()]).value_counts().sort_index()
    print("candidates found by N systems:",
          {int(k): int(v) for k, v in overlap.items()})


if __name__ == "__main__":
    main()
