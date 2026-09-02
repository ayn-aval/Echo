"""Accuracy and latency of the search stack -> results/search_benchmark.csv

    python -m eval.benchmark_search
    python -m eval.benchmark_search --latency-queries 400

Two questions, measured separately.

ACCURACY — does the cross-encoder rerank actually retrieve better reviews? Scored
with the unchanged score() from eval/run_retrieval.py, on the same 26 hand-judged
queries used since Phase 2, so these numbers sit directly beside every earlier
one instead of merely resembling them.

LATENCY — p50 and p95 over the full corpus, timed per stage, because "search is
fast" is not a claim anyone can act on. The stages are separated so the trade is
visible: if reranking triples the total, you can see exactly which part grew.

The exact-FAISS row doubles as a correctness check: it must reproduce the Phase 4
Precision@10 of 61.15 and MRR of 83.81 for sbert-domain. If it does not, the index
rows and the corpus rows are misaligned — an off-by-one there returns confidently
wrong reviews that still look plausible. Recall is excluded from the check on
purpose; it legitimately moves whenever the pool is revised.
"""

import argparse
import time
from pathlib import Path

# faiss MUST be imported before scikit-learn (which eval.run_retrieval pulls in).
# Both link their own OpenMP runtime, and on macOS loading sklearn's first makes
# the first faiss search segfault the process — exit 139, no traceback, no output.
# Verified: sklearn-first crashes, faiss-first works, KMP_DUPLICATE_LIB_OK does
# not help. OMP_NUM_THREADS=1 also fixes it but forces faiss single-threaded,
# which would understate the latency this script exists to measure.
# isort keeps this correct by accident (faiss < numpy < pandas < sklearn), but
# do not rely on that — leave the import here.
import faiss  # noqa: F401  (import for its side effect on load order)
import numpy as np
import pandas as pd

from eval.build_pool import load_corpus
from eval.run_retrieval import load_judgements, score
from src.search.query import embed_query, get_searcher
from src.search.rerank import CANDIDATES, MODEL, get_cross_encoder

RESULTS = Path("results/search_benchmark.csv")
NEG = -1e9


def ranked_to_sims(n_docs, per_query):
    """Turn {query_row: [doc_idx ordered best-first]} into a matrix score() reads.

    score() expects a full similarity matrix and argsorts it. A reranker only
    produces an ordering over its candidates, so unranked documents get a large
    negative score and the ranked ones get descending values. The resulting top-k
    is identical to the ordering we were given.
    """
    sims = np.full((len(per_query), n_docs), NEG, dtype=np.float32)
    for row, docs in per_query.items():
        for rank, doc in enumerate(docs):
            sims[row, doc] = -float(rank)
    return sims


def accuracy(queries, relevant, docs, k=10, candidates=CANDIDATES):
    index, corpus, encode, _ = get_searcher("sbert-domain", approximate=False)
    ivf, _, _, _ = get_searcher("sbert-domain", approximate=True)
    cross = get_cross_encoder(MODEL)
    qtexts = queries.query_text.tolist()

    exact_hits, ivf_hits, rerank_hits = {}, {}, {}
    for row, text in enumerate(qtexts):
        vec = embed_query(text, encode)
        exact_hits[row] = index.search(vec, k)[1][0].tolist()
        ivf_hits[row] = ivf.search(vec, k)[1][0].tolist()

        cand = index.search(vec, candidates)[1][0]
        scores = cross.predict([(text, str(corpus.content.iloc[i])) for i in cand],
                               show_progress_bar=False)
        rerank_hits[row] = cand[np.argsort(-np.asarray(scores))][:k].tolist()

    n = len(docs)
    return pd.DataFrame([
        {"config": "faiss-exact",
         **score(ranked_to_sims(n, exact_hits), docs, queries, relevant, k)},
        {"config": "faiss-ivf-nprobe10",
         **score(ranked_to_sims(n, ivf_hits), docs, queries, relevant, k)},
        {"config": f"faiss-top{candidates}+cross-encoder",
         **score(ranked_to_sims(n, rerank_hits), docs, queries, relevant, k)},
    ])


def latency(queries, docs, n_queries, k=10, candidates=CANDIDATES, warmup=10):
    """p50/p95 per stage. Warm-up iterations are discarded, not averaged in —
    the first call pays model load and MPS kernel compilation, which would
    otherwise dominate p95 and make the number meaningless."""
    index, corpus, encode, _ = get_searcher("sbert-domain", approximate=False)
    ivf, _, _, _ = get_searcher("sbert-domain", approximate=True)
    cross = get_cross_encoder(MODEL)

    rng = np.random.default_rng(42)
    pool = queries.query_text.tolist()
    extra = docs.content.sample(max(n_queries - len(pool), 0),
                                random_state=42).tolist()
    texts = (pool + extra)[:n_queries]
    rng.shuffle(texts)

    timings = {s: [] for s in ("encode", "faiss_exact", "faiss_ivf",
                               "cross_encode_50", "total_1stage", "total_2stage")}
    for i, text in enumerate(texts):
        t0 = time.perf_counter(); vec = embed_query(text, encode)
        t1 = time.perf_counter(); index.search(vec, k)
        t2 = time.perf_counter(); ivf.search(vec, k)
        t3 = time.perf_counter()
        cand = index.search(vec, candidates)[1][0]
        cross.predict([(text, str(corpus.content.iloc[j])) for j in cand],
                      show_progress_bar=False)
        t4 = time.perf_counter()
        if i < warmup:
            continue
        timings["encode"].append((t1 - t0) * 1000)
        timings["faiss_exact"].append((t2 - t1) * 1000)
        timings["faiss_ivf"].append((t3 - t2) * 1000)
        timings["cross_encode_50"].append((t4 - t3) * 1000)
        timings["total_1stage"].append((t2 - t0) * 1000)
        timings["total_2stage"].append((t4 - t0) * 1000)

    return pd.DataFrame([{"stage": s,
                          "p50_ms": round(float(np.percentile(v, 50)), 2),
                          "p95_ms": round(float(np.percentile(v, 95)), 2),
                          "n": len(v)} for s, v in timings.items()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latency-queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    docs = load_corpus()
    queries, relevant = load_judgements()
    print(f"{len(docs):,} indexed texts · {len(relevant)} judged queries\n")

    acc = accuracy(queries, relevant, docs, args.k)
    print("ACCURACY\n" + acc.to_string(index=False))

    exact = acc[acc.config == "faiss-exact"].iloc[0]
    # Precision@10 and MRR only. Recall@10 is deliberately NOT checked: it moves
    # whenever the pool is revised, because newly judged relevant reviews enter
    # the denominator. This run is the case in point — pooling the reranker added
    # 91 relevant reviews and Recall@10 fell 27.86 -> 24.08 for an unchanged
    # system. Precision stayed at exactly 61.15. That is PROGRESS.md's Phase 3
    # lesson 2, and an earlier version of this check ignored it and reported a
    # false MISMATCH.
    expected = {"precision@10": 61.15, "mrr": 83.81}
    tol = {"precision@10": 0.0, "mrr": 0.05}
    drift = {m: (float(exact[m]), v) for m, v in expected.items()
             if abs(float(exact[m]) - v) > tol[m]}
    print("\ncheck vs Phase 4 retrieval: " +
          ("MATCHES — index and corpus rows are aligned" if not drift
           else f"MISMATCH {drift} — index/corpus rows may be misaligned"))

    lat = latency(queries, docs, args.latency_queries, args.k)
    print(f"\nLATENCY (ms, {args.latency_queries} queries, first 10 discarded "
          f"as warm-up)\n" + lat.to_string(index=False))

    RESULTS.parent.mkdir(exist_ok=True)
    out = pd.concat([acc.assign(measure="accuracy"),
                     lat.assign(measure="latency")], ignore_index=True)
    out.to_csv(RESULTS, index=False)
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
