"""Two-stage search: FAISS retrieves 50 candidates, a cross-encoder reorders them.

    python -m src.search.rerank "my refund never arrived"

WHY NOT JUST CROSS-ENCODE EVERY REVIEW?
This is the question the whole project answers, so it is worth stating precisely.

A **bi-encoder** — what we trained in Phases 3 and 4 — puts the query through the
model and each review through the model *separately*, producing one vector each,
and compares the vectors. Because the review and the query never meet inside the
model, all 45,864 review vectors can be computed once, offline, and stored. A
query then costs one forward pass plus one matrix multiply.

A **cross-encoder** puts the query and one review through BERT *together*, as a
single glued input. Every layer can attend across both, so it sees interactions a
pair of separate vectors cannot represent — "not delivered" against "delivered
late" differ in a way two independent summaries blur. That makes it more accurate.
The cost is that it precomputes nothing: a score exists only for a specific
(query, review) pair, so ranking the whole corpus means 45,864 forward passes
**per query**, every query.

That is exactly the problem Sentence-BERT was written to solve. The paper opens
with it: finding the most similar pair among 10,000 sentences takes about 65
hours of BERT cross-encoding and about 5 seconds with SBERT embeddings, because
the first is quadratic in pairs and the second is linear in sentences with the
expensive part done in advance.

Two-stage search uses each model where it is strong. The bi-encoder cheaply
narrows 45,864 reviews to 50 — it only has to be good enough not to *lose* the
right answers, a recall job. The cross-encoder then carefully orders those 50 — a
precision job, on a small enough set to afford. 50 forward passes per query
instead of 45,864.

The catch, which the benchmark measures rather than assumes: this cross-encoder
is trained on English web search. These reviews are short, misspelled and partly
Hinglish, and Phase 4 established the model never bridged Hinglish to English. A
reranker can only reorder what stage one already found, so if stage one misses a
review, no amount of reranking recovers it.
"""

import argparse
import functools

import numpy as np
import pandas as pd

from src.search.query import get_searcher, embed_query

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 50


@functools.lru_cache(maxsize=2)
def get_cross_encoder(name: str = MODEL):
    """Loaded once and kept — about 90 MB, downloaded on first use."""
    from sentence_transformers import CrossEncoder
    from src.utils.device import get_device
    return CrossEncoder(name, device=str(get_device()), max_length=256)


def search_reranked(text, k=10, candidates=CANDIDATES, model="sbert-domain",
                    approximate=False, cross_name=MODEL):
    """Stage one retrieves `candidates`; stage two reorders them; top k returned."""
    index, corpus, encode, _ = get_searcher(model, approximate)
    scores, ids = index.search(embed_query(text, encode), candidates)
    rows = corpus.iloc[ids[0]]

    cross = get_cross_encoder(cross_name)
    pairs = [(text, str(c)) for c in rows.content]
    cross_scores = np.asarray(cross.predict(pairs, show_progress_bar=False),
                              dtype=np.float32)

    order = np.argsort(-cross_scores)[:k]
    picked = rows.iloc[order]
    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "cross_score": np.round(cross_scores[order], 4),
        "faiss_score": np.round(scores[0][order], 4),
        # Where this review sat before reranking — the clearest way to see
        # whether stage two actually changed anything.
        "faiss_rank": order + 1,
        "review_id": [i[0] for i in picked.review_ids],
        "content": picked.content.to_numpy(),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="+")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=CANDIDATES)
    args = ap.parse_args()

    query = " ".join(args.text)
    hits = search_reranked(query, args.k, args.candidates)
    print(f"\n{query!r} — top {args.candidates} from FAISS, reranked to "
          f"{len(hits)}\n")
    for r in hits.itertuples():
        moved = r.faiss_rank - r.rank
        arrow = f"{'up' if moved > 0 else 'down'} {abs(moved)}" if moved else "same"
        print(f"  {r.rank:>2}. cross {r.cross_score:>7.3f}  "
              f"(faiss #{r.faiss_rank:>2}, {arrow:>7})  "
              f"{' '.join(str(r.content).split())[:70]}")


if __name__ == "__main__":
    main()
