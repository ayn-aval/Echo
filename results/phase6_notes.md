# Phase 6 — Semantic search

FAISS index over the 45,864 `sbert-domain` review vectors, a query function, and
a two-stage option that reranks with a cross-encoder.

**Headline: the two-stage searcher reaches 75.77 Precision@10 — the first time in
this project that the neural stack beats TF-IDF's 65.00, which had won since
Phase 2. Single-stage search answers in 8.56 ms at p50.**

---

## Accuracy

`python -m eval.benchmark_search` -> `results/search_benchmark.csv`. Scored with
the unchanged `score()` from `eval/run_retrieval.py` on the same 26 hand-judged
queries used since Phase 2, so these sit directly beside every earlier number.

| system | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| tfidf | 30.69 | 65.00 | 85.71 |
| glove-avg | 23.17 | 58.85 | 78.22 |
| bert-mean | 17.25 | 43.46 | 74.27 |
| sbert-distilroberta-300k (Phase 3) | 17.72 | 45.77 | 72.25 |
| sbert-domain, single-stage (Phase 4) | 24.08 | 61.15 | 83.81 |
| faiss-ivf, nprobe=10 | 20.85 | 53.08 | 83.80 |
| **faiss top-50 + cross-encoder** | **30.64** | **75.77** | **86.54** |

Reranking buys **+14.62 Precision@10** over single-stage and **+10.77 over
TF-IDF**. Recall@10 ceiling on this collection is 49.78.

## The number this phase almost reported was wrong

The first benchmark said reranking *dropped* Precision@10 from 61.15 to **40.77**.
That number was an artifact and is recorded here because catching it was the main
work of the phase.

Looking at what the reranker promoted on its worst query gave it away:

```
query:    "the delivery took far longer than the app promised"
promoted: "delivery taking longer more than 45 minutes"
          "takes more time than what it should"
```

Both obviously relevant. The coverage check explained it:

| system | top-10 ever judged, before re-pooling |
|---|---|
| faiss-exact | 96.2% |
| cross-encoder | **47.3%** |

A reranker reorders 50 candidates, so it promotes reviews from ranks 11–50 that
no bi-encoder ever placed in a top-10. Those were never pooled, and the scorer
counts anything unpooled as irrelevant. **More than half the reranker's results
were being marked wrong because nobody had looked at them.** This is exactly the
Phase 3 lesson — "any new model evaluated on this collection must be re-pooled
first" — applied to a system that is not a new model but a new *ranking*.

Fixed by extending `eval/build_pool.py` with `--with-rerank`, adding 133
candidates, and judging them. After re-pooling, coverage is symmetric:

| system | top-10 ever judged, after |
|---|---|
| faiss-exact | 96.2% |
| cross-encoder | **96.2%** |
| faiss-ivf | 88.8% |

`faiss-ivf` is slightly under-pooled, so its 53.08 is a mild underestimate.

**Precision@10 was unchanged for every previously measured system across the pool
revision** — 65.00 / 58.85 / 43.46 / 45.77 / 61.15 to the decimal — while Recall
fell for all of them because 91 more relevant reviews entered the denominator
(sbert-domain 27.86 -> 24.08). That is Phase 3's lesson 2 reproducing exactly,
and it is why the alignment check in `eval/benchmark_search.py` now verifies
Precision and MRR and deliberately **not** Recall. An earlier version of that
check hardcoded the old Recall figure and reported a false MISMATCH.

## Latency

p50/p95 over 200 queries against the full corpus, first 10 discarded as warm-up
(the first call pays model load and MPS kernel compilation).

| stage | p50 ms | p95 ms |
|---|---|---|
| query encode (bi-encoder, MPS) | 6.89 | 17.58 |
| faiss exact search | **1.52** | 2.48 |
| faiss IVF search | 0.41 | 1.22 |
| cross-encode 50 candidates | 61.20 | 185.67 |
| **total, single-stage** | **8.56** | **19.20** |
| **total, two-stage** | **71.26** | **200.51** |

The brief asked for "well under a second". Single-stage is 8.56 ms — two orders
of magnitude inside it. Reranking costs **8.3x** the latency for **+14.62
Precision@10**, and at 71 ms p50 / 200 ms p95 it is still comfortably
interactive, so the trade is worth taking for a dashboard search box.

Note that FAISS is *not* the bottleneck in either configuration: exhaustive
search over 45,864 vectors takes 1.52 ms, while encoding the query takes 6.89 ms.

### FAISS's approximate index is not needed at this scale

IVF is 3.7x faster than exact search (0.41 ms vs 1.52 ms) and costs **8.07
Precision@10** (53.08 vs 61.15). Saving 1.1 ms out of an 8.56 ms query by giving
up accuracy is a bad trade. Approximate indexes earn their place at millions of
vectors; at 45,864 the exact index is already far cheaper than the encoder in
front of it. Claiming ANN was "needed" here would be false, and the honest
version of "we used FAISS" is knowing when its approximate structures start to
pay.

## Why not cross-encode all 45,864 reviews

This is the question the project exists to answer, so it is worth stating
precisely.

A **bi-encoder** — what Phases 3 and 4 trained — puts the query and each review
through the model *separately*, producing one vector each. Because they never
meet inside the model, all 45,864 review vectors are computed once, offline, and
a query costs one forward pass plus one matrix multiply.

A **cross-encoder** puts the query and one review through BERT *together*, so
every layer attends across both. It sees interactions two independent summaries
blur — "not delivered" against "delivered late" — which is why it is more
accurate. But it precomputes nothing: a score exists only for a specific
(query, review) pair, so ranking the corpus means 45,864 forward passes **per
query**.

The measurement here puts a number on it. 50 candidates cost 61.20 ms, so
45,864 would cost roughly **56 seconds per query** at the same rate, against
8.56 ms for the bi-encoder — about 6,500x. That is the Sentence-BERT paper's
opening argument reproduced on this corpus: the paper reports finding the most
similar pair among 10,000 sentences taking ~65 hours with BERT cross-encoding
versus ~5 seconds with SBERT embeddings.

Two-stage search uses each where it is strong: the bi-encoder cheaply narrows
45,864 to 50 (a recall job — it only has to avoid *losing* the right answers),
the cross-encoder carefully orders those 50 (a precision job, on a set small
enough to afford).

## A macOS bug worth recording

FAISS and scikit-learn each link their own OpenMP runtime. On macOS, importing
scikit-learn **before** FAISS makes the first FAISS call **segfault the process**
— exit 139, no traceback, no output, no partial results. Verified:

| condition | result |
|---|---|
| sklearn imported first | segfault |
| faiss imported first | works |
| `OMP_NUM_THREADS=1` | works, but forces FAISS single-threaded |
| `KMP_DUPLICATE_LIB_OK=TRUE` | still segfaults |

`eval/benchmark_search.py` and `eval/build_pool.py` now import faiss at the top
of their third-party block with the reason written down. `OMP_NUM_THREADS=1` was
rejected because it would understate the very latency the benchmark measures.

## Limitations

1. **The cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`, pretrained on
   English web search** — nothing here was trained on Swiggy data. It still helps
   substantially, but it inherits the Hinglish weakness: searching *"khana thanda
   tha"* (the food was cold) returns *"kahana bht acha tha"* (the food was very
   good) at rank 1, matching surface form rather than meaning.
2. **A reranker can only reorder what stage one found.** If FAISS misses a review
   in its top 50, no amount of reranking recovers it, which is why recall
   improves far less than precision.
3. **26 queries.** Every number here rests on that, and 24 of the 50 evaluation
   queries remain unlabelled.
4. **20 pooled candidates on one query** (*"my Instamart order had a problem"*)
   are still unjudged, and `faiss-ivf` sits at 88.8% coverage rather than 96.2%.
5. **Latency is single-query on an idle Mac** — no concurrency, no cold-start,
   no network. A deployed service would be slower.

## Reproducing

```bash
python -m src.search.index                        # build exact + IVF indexes
python -m src.search.query "app keeps crashing"   # ranked reviews with scores
python -m src.search.rerank "my refund never arrived"
python -m eval.build_pool --augment --with-rerank # MANDATORY before scoring
streamlit run app/label.py                        # judge the new candidates
python -m eval.benchmark_search                   # accuracy + latency
```
