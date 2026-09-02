"""Build and persist FAISS indexes over the review vectors.

    python -m src.search.index                    # exact + approximate
    python -m src.search.index --model bert-mean

Writes data/index/{model}.faiss (exact), data/index/{model}.ivf.faiss
(approximate) and data/index/{model}.json — a sidecar recording the model, vector
count, dimension and index type, so a stale index cannot be served against
vectors it was not built from.

What FAISS is doing, since it is easy to assume it is magic. Comparing a query to
45,864 reviews in a Python loop is slow not because the arithmetic is hard but
because of overhead: a function call and a scattered memory read per review.
FAISS keeps every vector in one contiguous block and does the whole comparison as
one tight loop of SIMD instructions, where a single CPU instruction multiplies 8
to 16 numbers at once. That alone is the speedup at this corpus size.

Vectors are L2-normalised and stored in an IndexFlatIP, because inner product on
unit vectors *is* cosine similarity — the same thing every other part of this
project measures, so search scores and evaluation scores mean the same thing.

Two indexes are built on purpose:

  IndexFlatIP    exact. Checks every vector, always correct.
  IndexIVFFlat   approximate. Partitions the vectors into nlist cells around
                 centroids and searches only the nprobe nearest cells, so it
                 looks at a fraction of the corpus and can miss true neighbours.

At 45,864 vectors the exact index is already fast, and the approximate one is not
needed. It is built anyway so the benchmark can show what it costs in recall and
what it saves in time — the honest version of "we used FAISS" is knowing when its
approximate structures start to pay, not implying they were required here.
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

VECTORS = Path("data/vectors")
INDEX = Path("data/index")
NLIST = 200          # partitions for the IVF index; ~sqrt(n) is the usual heuristic


def load(model: str) -> np.ndarray:
    path = VECTORS / f"{model}.npy"
    if not path.exists():
        raise SystemExit(f"{path} missing — run `python -m src.embeddings.encode_corpus`")
    vectors = np.load(path).astype(np.float32)
    # copy() because normalize_L2 works in place and np.load may hand back a
    # read-only or memory-mapped array.
    vectors = np.ascontiguousarray(vectors.copy())
    faiss.normalize_L2(vectors)
    return vectors


def build_exact(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_ivf(vectors, nlist=NLIST, nprobe=10):
    quantiser = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFFlat(quantiser, vectors.shape[1], nlist,
                               faiss.METRIC_INNER_PRODUCT)
    # Unlike the exact index, this one must first *learn* where the cell
    # centroids are before anything can be added.
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="sbert-domain")
    ap.add_argument("--nlist", type=int, default=NLIST)
    ap.add_argument("--nprobe", type=int, default=10)
    args = ap.parse_args()

    vectors = load(args.model)
    n, dim = vectors.shape
    INDEX.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: {n:,} vectors x {dim}d")

    exact = build_exact(vectors)
    faiss.write_index(exact, str(INDEX / f"{args.model}.faiss"))
    print(f"  exact       IndexFlatIP   {exact.ntotal:,} vectors")

    ivf = build_ivf(vectors, args.nlist, args.nprobe)
    faiss.write_index(ivf, str(INDEX / f"{args.model}.ivf.faiss"))
    print(f"  approximate IndexIVFFlat  {ivf.ntotal:,} vectors "
          f"(nlist={args.nlist}, nprobe={args.nprobe})")

    (INDEX / f"{args.model}.json").write_text(json.dumps({
        "model": args.model, "n_vectors": int(n), "dim": int(dim),
        "metric": "inner_product_on_unit_vectors_equals_cosine",
        "nlist": args.nlist, "nprobe": args.nprobe,
        "source": str(VECTORS / f"{args.model}.npy"),
    }, indent=2))

    # A vector must be its own nearest neighbour at similarity 1.0. If this fails
    # the vectors were not normalised and every score downstream is wrong.
    sims, ids = exact.search(vectors[:5], 1)
    assert (ids.ravel() == np.arange(5)).all(), "self-search did not return self"
    assert np.allclose(sims.ravel(), 1.0, atol=1e-4), f"self-similarity {sims.ravel()}"
    print("  self-search check passed (each vector is its own top hit at 1.0)")


if __name__ == "__main__":
    main()
