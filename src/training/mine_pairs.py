"""Build the Phase 4 weak-supervision training pairs from the review corpus.

    python -m src.training.mine_pairs

Writes data/phase4_pairs.jsonl — one JSON object per line, {"a", "b", "source"}.
Upload that file to Kaggle as a Dataset; the training notebook reads it.

Two sources, chosen in the Phase 4 kickoff after the reply-based strategies in
PROJECT_PLAN.md were measured and rejected (see eval/reply_signal.py — Swiggy's
replies encode the star rating and nothing else, so pairing on them would teach
the model that a UPI outage and a late delivery mean the same thing).

  mined   two different reviews that BOTH tf-idf and the Phase 3 encoder rank in
          each other's top-k. Two systems with unrelated failure modes agreeing
          is evidence; the Phase 3 encoder agreeing with itself is an echo, which
          is why the tf-idf constraint is not optional. This is the source of
          real domain relations — "extra charge for express delivery" matched to
          "pay extra charges ... order was too late".

  simcse  every review paired with itself. Dropout stays on during training, so
          the two passes give slightly different vectors and the model learns to
          pull them together. A false positive is impossible by construction.
          Supplies the volume the mined set cannot.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from eval.build_pool import load_corpus
from src.db.connection import connection

SBERT_CACHE = Path("data/embeddings/reviews_sbert_300k.npy")
OUT = Path("data/phase4_pairs.jsonl")
BLOCK = 500


def words(text):
    return set(re.sub(r"[^a-z0-9 ]", "", str(text).lower()).split())


def mine(texts, sb, tf, k, min_sbert, min_tfidf, max_jaccard):
    """Pairs both systems independently rank in each other's top-k."""
    toks = [words(t) for t in texts]
    pairs, trivial = [], 0
    for start in range(0, len(texts), BLOCK):
        idx = np.arange(start, min(start + BLOCK, len(texts)))
        s_sim = sb[idx] @ sb.T
        t_sim = (tf[idx] @ tf.T).toarray()
        np.put_along_axis(s_sim, idx[:, None], -1, 1)   # never match yourself
        np.put_along_axis(t_sim, idx[:, None], -1, 1)
        s_top = np.argpartition(-s_sim, k, 1)[:, :k]
        t_top = np.argpartition(-t_sim, k, 1)[:, :k]
        for row, i in enumerate(idx):
            for j in map(int, set(s_top[row]) & set(t_top[row])):
                if i >= j or s_sim[row, j] < min_sbert or t_sim[row, j] < min_tfidf:
                    continue
                a, b = toks[i], toks[j]
                # One text being the other's word-set teaches nothing: these are
                # case and emoji variants, 60% of raw hits in the pilot.
                if a <= b or b <= a or len(a & b) / max(len(a | b), 1) > max_jaccard:
                    trivial += 1
                    continue
                pairs.append((i, j, round(float(s_sim[row, j]), 4)))
        print(f"\r  mined {start + len(idx):,}/{len(texts):,} "
              f"-> {len(pairs):,} pairs", end="", flush=True)
    print(f"  ({trivial:,} trivial duplicates discarded)")
    return pairs


def leakage_check(pairs, docs):
    """How often does a mined pair join two reviews judged relevant to one query?

    No judgement is ever read while mining, so this is not leakage in the strict
    sense. But if the mined pairs happened to reconstruct the answer key, the
    Phase 4 gain would be partly circular, so the overlap is measured and
    reported rather than assumed to be zero.
    """
    import pandas as pd
    with connection() as conn:
        j = pd.read_sql("SELECT query_id, review_id FROM eval_judgements "
                        "WHERE relevant", conn)
    by_query = j.groupby("query_id").review_id.apply(set).to_dict()
    rid = docs.review_id.to_numpy()
    hits = sum(any(rid[i] in g and rid[k] in g for g in by_query.values())
               for i, k, _ in pairs)
    pct = hits / max(len(pairs), 1) * 100
    print(f"\nleakage check: {hits:,} of {len(pairs):,} mined pairs ({pct:.2f}%) "
          f"join two reviews judged relevant to the same query")
    return pct


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-sbert", type=float, default=0.70)
    ap.add_argument("--min-tfidf", type=float, default=0.20)
    ap.add_argument("--max-jaccard", type=float, default=0.75)
    ap.add_argument("--no-simcse", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not SBERT_CACHE.exists():
        raise SystemExit(f"{SBERT_CACHE} missing — run `python -m eval.run_retrieval` "
                         "first to build the Phase 3 embedding cache.")

    docs = load_corpus()
    texts = docs.content.tolist()
    sb = np.load(SBERT_CACHE).astype(np.float32)
    sb /= np.clip(np.linalg.norm(sb, axis=1, keepdims=True), 1e-9, None)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    print(f"{len(texts):,} distinct review texts · sbert {sb.shape} · tfidf {tf.shape}\n")

    pairs = mine(texts, sb, tf, args.top_k, args.min_sbert, args.min_tfidf,
                 args.max_jaccard)
    leakage_check(pairs, docs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, j, sim in pairs:
            f.write(json.dumps({"a": texts[i], "b": texts[j],
                                "source": "mined", "sbert": sim}) + "\n")
        if not args.no_simcse:
            for t in texts:
                f.write(json.dumps({"a": t, "b": t, "source": "simcse"}) + "\n")

    n_simcse = 0 if args.no_simcse else len(texts)
    size = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out} — {len(pairs):,} mined + {n_simcse:,} simcse "
          f"= {len(pairs) + n_simcse:,} pairs, {size:.1f} MB")


if __name__ == "__main__":
    main()
