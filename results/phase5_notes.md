# Phase 5 — Theme discovery

UMAP + HDBSCAN over the 45,864 distinct review texts, run identically on three
embeddings so the only thing varying is the model.

**Headline: the domain-adapted model is the only one of the three that produces
usable themes — and the metric that looks most flattering to the alternatives is
the one that misleads.**

---

## The comparison

`python -m eval.clustering_comparison` -> `results/clustering_comparison.csv`.
Same UMAP settings, same HDBSCAN settings (`min_cluster_size=60`,
`min_samples=None`), same naming, for all three.

| model | dim | clusters | noise % | biggest cluster | silhouette | audit |
|---|---|---|---|---|---|---|
| glove-avg | 300 | 47 | **15.04** | **58.64%** | −0.0330 | 44.12% |
| bert-mean | 768 | 42 | 25.45 | 41.15% | 0.0502 | 73.53% |
| **sbert-domain** | 768 | **110** | 40.30 | **5.88%** | **0.1138** | **82.35%** |

### Noise percentage on its own ranks the worst model first

GloVe posts the best noise figure, 15.04%. It earns it by refusing to separate
anything: its largest cluster holds **58.6% of the entire corpus**, and contains

```
cluster 25 — terms: cancellation, cancelled, refund, cancel, care
  · varry 👍👍 good swiggy                      <- 5-star praise
  · Worst nightmare to have swiggy              <- 1-star rant
  · cashback not recived yet                    <- refund issue
  · Very poor customer service. For an incomplete order...
```

That is a bucket, not a theme. `largest_pct` was added to the comparison output
specifically so this cannot be read past; the trap is documented in
`eval/clustering_comparison.py` rather than left for a reader to notice.

`sbert-domain`'s largest theme is 5.88%. It actually carved the corpus into
topics.

### Silhouette is a within-model diagnostic, not a comparison

It is computed inside each model's own embedding space, and those spaces differ
in geometry and dimensionality (GloVe is 300d, the others 768d). It is reported
because it is useful for tuning one model, and it should **not** be quoted as
proof that one model beats another.

## The blind hand-audit — the measure that settles it

102 judgements, 34 per model, via `streamlit run app/audit.py`. Which model
produced an assignment was never shown; verified with `streamlit.testing.v1`
that no model name reaches the rendered page. Rows were interleaved by a fixed
shuffle and drawn deterministically (md5 of review_id + model), so the sample
cannot be redrawn until it flatters someone.

| model | correct | accuracy | 95% CI |
|---|---|---|---|
| sbert-domain | 28/34 | 82.4% | [66.5, 91.7] |
| bert-mean | 25/34 | 73.5% | [56.9, 85.4] |
| glove-avg | 15/34 | 44.1% | [28.9, 60.5] |

Fisher exact, pairwise:

| comparison | p | verdict |
|---|---|---|
| sbert-domain vs glove-avg | 0.0022 | **significant** |
| bert-mean vs glove-avg | 0.0258 | **significant** |
| **sbert-domain vs bert-mean** | **0.5597** | **NOT significant** |

**This is the honest limit of the result.** The trained model beats averaged
GloVe decisively. Its 8.8-point lead over plain mean-pooled BERT **is not
statistically significant at 34 judgements per model**, and must not be claimed
as one. Settling that would need a few hundred more judgements.

The case for `sbert-domain` over `bert-mean` therefore rests on structure rather
than on audit accuracy, and that evidence is not a sampling question: 110 themes
against 42, and 5.88% of the corpus in the largest theme against 41.15%. A theme
containing 41% of all reviews is not something a product manager can act on,
whatever its audit score.

### Where each model's errors actually live

| model | audit rows from its biggest cluster | accuracy inside | outside |
|---|---|---|---|
| glove-avg | 20/34 (58.8%) | **15.0%** | 85.7% |
| bert-mean | 10/34 (29.4%) | 70.0% | 75.0% |
| sbert-domain | 1/34 (2.9%) | 100.0% | 81.8% |

GloVe's smaller clusters are **fine** — 85.7% accurate. Its entire failure is
concentrated in the one giant bucket, which is also why that bucket dominates its
audit sample. This is a sharper diagnosis than the headline 44.1%: GloVe can
group reviews, it just cannot decide where the biggest group ends.

## The themes

`python -m src.clustering.name_themes --model sbert-domain`. 110 themes, 44,332
of 64,280 review rows assigned, the rest noise.

| # | reviews | avg★ | label |
|---|---|---|---|
| 105 | 1,950 | 1.16 | mins / 45 / 15 |
| 92 | 856 | 1.14 | bad / experience / service |
| 103 | 761 | 1.09 | waited / hours / hour |
| 64 | 713 | 1.03 | worst / support / service |
| 62 | 600 | 3.12 | far / better / zomato |
| 88 | 538 | 1.31 | chicken / egg / fried |
| 109 | 537 | 1.11 | cancelled / charged / canceled |
| 39 | 2,855 | 2.47 | mere / karne / karte *(Hinglish, see below)* |
| 57 | 3,402 | 4.63 | good |
| 49 | 2,785 | 4.73 | nice / good / service |
| 56 | 2,246 | 4.80 | excellent / good |
| 51 | 1,971 | 4.76 | good / service / nice |

The complaint themes separate cleanly by rating — delivery timing 1.16★, long
waits 1.09★, support failures 1.03★, cancellations and wrong charges 1.11★, food
quality 1.31★ — and nothing was told to look for them.

### Naming took three attempts, and the failures are the interesting part

c-TF-IDF's first output was `wark / nice / niceee`. It ranks **misspellings**
highest, because "veri", "coustmer" and "zamato" are maximally distinctive
*precisely because* almost nobody writes them. Two filters were needed, both now
in `src/clustering/name_themes.py` with the reasoning:

1. the term must appear in at least 25 reviews corpus-wide, and
2. it must appear in at least 4% of *its own cluster's* reviews.

Filter 1 alone still produced `supar / sarvice / verry`. A usable label needs a
term that is both distinctive across clusters and common within one.

## Limitations

1. **The largest theme is a language, not a topic.** Theme 39, 2,855 reviews,
   top terms `mere / karne / karte` — Hindi function words. It groups Hinglish
   reviews regardless of what they complain about, so a food-quality complaint
   and a late-delivery complaint land together because both are romanised Hindi.
   This is the Phase 4 limitation surfacing in the product: the model never
   learned to bridge Hinglish to English, and it now costs a theme. Reported
   rather than patched, by decision.
2. **Seven of the top fifteen themes are generic praise** — roughly 13,600
   reviews of "good / nice / excellent". This is the predicted cost of the Phase 1
   decision to keep 2+ word reviews rather than 4+, recorded in PROGRESS.md at
   the time. Arguably a finding: over a fifth of the corpus carries no actionable
   content.
3. **40.3% of distinct texts are noise** — the highest of the three models. The
   trade is deliberate: the alternative settings that lowered noise did so by
   merging everything. `mcs=120, ms=None` reached 0.53% noise with a single
   cluster of 45,222.
4. **sbert-domain vs bert-mean is not statistically separated** by the audit.
5. **One audit sample per model, one judge.** No inter-annotator agreement.

## Reproducing

```bash
python -m src.db.init_db                      # themes, review_themes, theme_audit
python -m src.embeddings.encode_corpus        # vectors + review-id mapping
python -m src.clustering.tune --model sbert-domain    # the parameter sweep
python -m src.clustering.name_themes --model sbert-domain
python -m eval.clustering_comparison --persist
streamlit run app/audit.py                    # blind audit
python -m eval.clustering_comparison          # folds the audit in
```

UMAP uses a fixed `random_state`, so the themes reproduce exactly. UMAP warns
that this disables its parallelism; a clustering nobody can reproduce is not a
result, so the speed was traded away deliberately.
