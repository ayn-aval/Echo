"""Search the reviews by meaning.

    python -m src.search.query "app keeps crashing"
    python -m src.search.query "delivery was late" --k 5 --approximate

    from src.search.query import search
    hits = search("refund not received", k=10)

Returns a DataFrame of rank, score, review_id, content. The score is cosine
similarity: 1.0 is identical, 0 is unrelated.

The index and encoder are loaded lazily and kept, because loading the encoder
costs a second or two and a dashboard must not pay that per keystroke. Phase 7
wraps get_searcher() in @st.cache_resource for the same reason — the difference
between the two Streamlit caches being that @st.cache_data memoises a *value*
(a DataFrame of results) while @st.cache_resource holds a live *object* that
cannot be pickled, like a model or an open index.
"""

import argparse
import functools
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

INDEX = Path("data/index")
CORPUS = Path("data/vectors/corpus.parquet")


@functools.lru_cache(maxsize=4)
def get_searcher(model: str = "sbert-domain", approximate: bool = False):
    """(index, corpus, encode) — built once per process, then reused."""
    meta_path = INDEX / f"{model}.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing — run `python -m src.search.index`")
    meta = json.loads(meta_path.read_text())

    suffix = ".ivf.faiss" if approximate else ".faiss"
    index = faiss.read_index(str(INDEX / f"{model}{suffix}"))
    corpus = pd.read_parquet(CORPUS)

    # A mismatch here means the index was built from different vectors than the
    # mapping describes, which would return confidently wrong reviews.
    if index.ntotal != len(corpus):
        raise SystemExit(f"index holds {index.ntotal:,} vectors but corpus.parquet "
                         f"has {len(corpus):,} rows — rebuild the index")

    from src.embeddings import sbert
    path, _ = sbert.TRAINED[model] if model in sbert.TRAINED else (None, None)
    if path is None:
        raise SystemExit(f"no trained encoder registered for {model}")
    encode = sbert.make_encoder(path, quiet=True)
    return index, corpus, encode, meta


def embed_query(text, encode):
    vec = np.ascontiguousarray(np.asarray(encode([text]), dtype=np.float32))
    faiss.normalize_L2(vec)          # the index stores unit vectors; match it
    return vec


def search(text: str, k: int = 10, model: str = "sbert-domain",
           approximate: bool = False) -> pd.DataFrame:
    index, corpus, encode, _ = get_searcher(model, approximate)
    scores, ids = index.search(embed_query(text, encode), k)
    rows = corpus.iloc[ids[0]]
    return pd.DataFrame({
        "rank": np.arange(1, len(rows) + 1),
        "score": np.round(scores[0], 4),
        # One text can stand for several identical reviews; the first id is the
        # representative, n_rows says how many people wrote it.
        "review_id": [ids_[0] for ids_ in rows.review_ids],
        "n_rows": rows.n_rows.to_numpy(),
        "content": rows.content.to_numpy(),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="+")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="sbert-domain")
    ap.add_argument("--approximate", action="store_true",
                    help="use the IVF index instead of exact search")
    args = ap.parse_args()

    query = " ".join(args.text)
    hits = search(query, args.k, args.model, args.approximate)
    print(f"\n{query!r} — {'approximate' if args.approximate else 'exact'} "
          f"search, top {len(hits)}\n")
    for r in hits.itertuples():
        dupes = f"  (x{r.n_rows})" if r.n_rows > 1 else ""
        print(f"  {r.rank:>2}. {r.score:.3f}{dupes}  "
              f"{' '.join(str(r.content).split())[:88]}")


if __name__ == "__main__":
    main()
