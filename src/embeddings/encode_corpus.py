"""Encode the review corpus once and persist vectors plus their id mapping.

    python -m src.embeddings.encode_corpus                 # all three models
    python -m src.embeddings.encode_corpus --model sbert-domain
    python -m src.embeddings.encode_corpus --force         # ignore the cache

Writes, under data/vectors/:

    corpus.parquet      row_index, content, n_rows, review_ids
    {model}.npy         (45864, dim) float32, row i <-> corpus row i

Why a mapping file at all: the .npy is a bare array. Row 0 is 768 numbers with
no record of which review it came from, and because we encode *distinct texts*
rather than rows, one vector serves every row sharing that text — 45,864 vectors
cover 64,280 rows. Without corpus.parquet a cluster assignment cannot be joined
back to Postgres.

The ordering comes from eval.build_pool.load_corpus(), the same function the
retrieval evaluation uses, so a vector's row index means the same thing in both
places. Existing embedding caches from Phase 4 are reused rather than recomputed.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from eval.build_pool import CACHE, load_corpus

OUT = Path("data/vectors")

# name -> (Phase 4 cache file, how to build it if that file is missing)
MODELS = {
    "sbert-domain": "reviews_sbert_domain",
    "bert-mean": "reviews_bert_mean",
    "glove-avg": "reviews_glove",
}
RATE = 186  # texts/sec measured on MPS, for the runtime estimate


def build_encoder(name, texts):
    """Only imported/constructed when a cache is actually missing."""
    from src.embeddings import baselines, glove, sbert
    if name == "glove-avg":
        return glove.make_encoder(texts, "reviews")
    if name == "bert-mean":
        return baselines.make_encoder("mean")
    path, _ = sbert.TRAINED["sbert-domain"]
    return sbert.make_encoder(path)


def vectors_for(name, texts, force=False):
    cached = CACHE / f"{MODELS[name]}.npy"
    if cached.exists() and not force:
        v = np.load(cached)
        print(f"  {name:13} reusing {cached} {v.shape}")
        return v
    mins = len(texts) / RATE / 60
    print(f"  {name:13} not cached — encoding {len(texts):,} texts, "
          f"expect about {mins:.0f} minute(s) on this machine")
    v = build_encoder(name, texts)(texts)
    np.save(cached, v)
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(MODELS), action="append",
                    help="repeatable; default is all three")
    ap.add_argument("--force", action="store_true",
                    help="re-encode even if a cached .npy exists")
    args = ap.parse_args()
    wanted = args.model or list(MODELS)

    docs = load_corpus()
    texts = docs.content.tolist()
    print(f"{len(texts):,} distinct review texts\n")

    OUT.mkdir(parents=True, exist_ok=True)

    # The mapping: every row of `reviews` that shares this exact text. Built
    # from the same WHERE clause load_corpus() uses, so no row is lost or gained.
    from src.db.connection import connection
    with connection() as conn:
        rows = pd.read_sql("""
            SELECT content, array_agg(review_id ORDER BY review_id) AS review_ids
              FROM reviews
             WHERE app='swiggy' AND keep_for_themes
             GROUP BY content""", conn)
    mapping = docs[["content"]].merge(rows, on="content", how="left")
    assert len(mapping) == len(docs), "merge changed the row count"
    assert mapping.review_ids.notna().all(), "a text has no rows behind it"
    mapping.insert(0, "row_index", np.arange(len(mapping)))
    mapping["n_rows"] = mapping.review_ids.str.len()
    mapping.to_parquet(OUT / "corpus.parquet", index=False)
    print(f"corpus.parquet: {len(mapping):,} texts covering "
          f"{int(mapping.n_rows.sum()):,} rows\n")

    for name in wanted:
        v = vectors_for(name, texts, args.force)
        assert v.shape[0] == len(texts), f"{name}: {v.shape[0]} != {len(texts)}"
        dest = OUT / f"{name}.npy"
        if not dest.exists() or args.force:
            shutil.copyfile(CACHE / f"{MODELS[name]}.npy", dest)
        print(f"  {name:13} -> {dest}  {v.shape}")


if __name__ == "__main__":
    main()
