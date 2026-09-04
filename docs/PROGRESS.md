# Progress

## Current status

**Phases 0 through 9 complete.**

| phase | state |
|---|---|
| 0 — Setup | done (Streamlit hello-world skipped; not needed until Phase 7) |
| 1 — Data collection | done — 100,000 Swiggy reviews in Postgres |
| 2 — Baselines + eval harness | done — STS and review-retrieval baselines measured |
| 3 — Reproduce SBERT | **done. 72.17 vs paper's 74.21; ablation run** |
| 4 — Domain adaptation | **done. Precision@10 45.77 -> 61.15, STS 72.17 -> 74.54** |
| 5 — Theme discovery | **done. 110 themes; audit 82.4% vs GloVe's 44.1%** |
| 6 — Semantic search | **done. 75.77 P@10 two-stage — first system to beat TF-IDF** |
| 7 — Streamlit dashboard | **done. 4 screens, top-tab shell; `streamlit run app/main.py`** |
| 8 — Trends and alerts | **done. 16 alerts at z>=3; Power BI guide written** |
| 9 — Write-up and polish | **done. README, cleanup, traceability check** |

**Immediate next step:** nothing is blocking. The highest-value remaining work is
**Phase 4b** (`notebooks/phase4b_ablation_kaggle.ipynb`, ~20 min on Kaggle), because
it settles the circularity question the README raises and leaves open — mined pairs
required TF-IDF to agree, so the model may have partly learned to imitate the system
it is being compared against.

After that: deployment (nobody can currently run this without a local Postgres), and
~200 more theme-audit judgements to separate sbert-domain from bert-mean.

---

## Phase 1 — Data collection

100,000 Swiggy reviews (`in.swiggy.android`), 18 Jan – 26 Aug 2026, 220 days,
zero duplicates, 46 MB. Scraped entirely from the unfiltered stream — the five
per-rating backup streams were never needed.

- **99.9% carry a Swiggy reply** — but Phase 4 measured the signal and it is
  **star rating only**, not complaint category. See the Phase 4 section; the
  replies were rejected as a training signal. Coverage was never the problem.
- Ratings are bimodal: 30.7% one-star, 53.8% five-star.
- ~270 reviews/day. Volume dipped to ~150/day for two weeks in late April 2026 and
  recovered — **this is real, not a scraping gap** (zero days missing). Phase 8
  alerting will fire on that window.

### Cleaning rules (full rationale in PROJECT_PLAN.md)

Nothing is deleted; `word_count`, `lang` and `keep_for_themes` are set on every
row. Dropped: 2,766 emoji-only and 35,646 single-word reviews. **Kept 64,280 of
100,000 (64.3%)**, containing 45,864 distinct texts.

Threshold set at 2+ words — the user's call over the recommended 4+, to retain
short genuine complaints like "worst service". **Phase 5 should therefore expect
one or more large generic-praise clusters and report them rather than hide them.**

The cleaned corpus is **not sentiment-representative** (filtering short reviews
raises the 1-star share). Rating and volume charts must use all 100,000; only
themes use the 64,280.

**Encode distinct texts, not rows** — 64,280 rows are 45,864 unique strings.

---

## Phase 2 — Baselines and the evaluation harness

`eval/sts_eval.py` takes any `encode(list[str]) -> ndarray` callable. Phases 3–5
call it unchanged, which is what makes the numbers comparable.

STS (Spearman x100, seven datasets):

| model | Avg |
|---|---|
| glove-avg | 53.88 |
| bert-mean | 52.64 |
| bert-cls | 31.44 |

**The paper's central motivation reproduces:** averaged GloVe beats mean-pooled
BERT. CLS at 31.44 sits close to the paper's 29.19.

Review retrieval: 26 hand-labelled queries, 935 judgements, 479 relevant. Pool
built from TF-IDF + GloVe + BERT + (later) the trained model.

**Honesty note for the README:** our GloVe baseline is 7.4 points below the
paper's (53.88 vs 61.32) because we compute one Spearman over all pairs while
SentEval averages per sub-dataset (measured: 44.11 vs 50.07 on STS-B). So any
"improvement over GloVe" we quote is inflated — **the paper's own margin is
+12.89, which is the right sanity check.** Rejected explanations: punctuation
tokenisation makes it worse; vocabulary construction is not at fault.

---

## Phase 3 — Reproduce Sentence-BERT

Siamese loop written by hand in raw PyTorch. `sentence-transformers` is imported
nowhere; `src/training/train.py` raises if it is.

**72.17 average Spearman across seven STS datasets against the paper's 74.21 for
SRoBERTa-NLI-base — a 2.04 gap on 32% of its training data.** We exceed the paper
on STS13 (+1.22). Full table in `results/phase3_notes.md` and
`results/table1_comparison.csv`.

Trained on Colab: `distilroberta-base`, 300k stratified SNLI+MultiNLI pairs,
paper-exact config (1 epoch, batch 16, Adam 2e-5, 10% warmup, mean pooling,
`(u,v,|u-v|)`). Encoder at `models/sbert-distilroberta-300k/encoder` (gitignored),
verified to reproduce the Colab numbers exactly on MPS.

### Retrieval on the review corpus — TF-IDF still wins

| model | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| tfidf | 38.70 | **65.00** | 85.71 |
| glove-avg | 30.41 | 58.85 | 78.22 |
| sbert-distilroberta-300k | 23.46 | 45.77 | 72.25 |
| bert-mean | 22.69 | 43.46 | 74.10 |

**This is the finding, not a failure.** 72.17 on STS and 45.77 on reviews is
precisely the gap PROJECT_PLAN.md opens Phase 4 with: generic sentence-similarity
ability does not transfer to three-word misspelled Hinglish reviews when the model
was trained on clean grammatical prose. **Phase 4 must beat 45.77 Precision@10.**

### Phase 3b — the Table 6 ablation

Nine runs on Kaggle (T4), 100k pairs each, scored on STS-B. Full table and
reasoning in `results/phase3_notes.md`; raw scores in `results/ablation.csv`.

**Claim 1 — `|u-v|` is the critical component. HOLDS.** `(u,v)` 52.98 ->
`(u,v,|u-v|)` 68.18, a margin of **+15.20 against the paper's +14.74** — slightly
larger than the paper's, on a third of the data. Stronger still as a group: every
configuration containing `|u-v|` scores 62.22-70.93, every one without it scores
52.98-60.38, **with no overlap.** The paper's central architectural claim
reproduces independently.

**Claim 2 — adding `u*v` hurts. DOES NOT HOLD.** `(u,v,|u-v|)` 68.18 -> `+u*v`
70.93, so **+2.75 where the paper reports -0.34.** Our effect is eight times the
paper's margin and opposite in sign, so this is not simply noise. Hypothesis, not
measurement: at 100k pairs the model is undertrained and richer features still
help, where the paper's fully-trained encoder finds the product redundant. **One
seed per configuration — we cannot fully separate this from variance.** Report it
as a finding to explain, never as a correction to the paper.

**Pooling — right winner, wrong loser.** MEAN 68.18 > MAX 66.12 > CLS 63.05. MEAN
wins as in the paper, but the paper has CLS *second* (0.98 spread); ours has it
last by 5.13. Consistent with the Phase 2 baseline, where `bert-cls` scored 31.44
against `bert-mean`'s 52.64: the CLS token is not a sentence representation until
training makes it one.

### Two methodology lessons — do not repeat

1. **Pooling bias nearly produced a wrong headline.** The first retrieval run gave
   the trained model 7.56 Recall@10, worse than untrained BERT. Cause: the pool was
   built from TF-IDF, GloVe and BERT, so 86% of what the trained model retrieved
   had never been judged and scored as irrelevant by default. Fixed with
   `python -m eval.build_pool --augment` plus 224 new judgements.
   **Any new model evaluated on this collection must be re-pooled first.**
2. **Quote Precision@10, never Recall@10, across pool revisions.** Precision was
   identical for all baselines before and after re-pooling (65.00 / 58.85 / 43.46);
   recall fell for everyone (TF-IDF 46.01 -> 38.70) purely because 89 more relevant
   reviews entered the denominator.

### Verified rather than assumed

- step-1 loss 1.0901 vs theoretical ln(3) = 1.0986
- overfit test: 192 pairs, 40 epochs -> loss 0.0011, accuracy 1.00
- params total == encoder + head, so weight sharing is real, not two encoders
- mean and max pooling both ignore padding
- resume continues at the right step with optimizer state intact
- 785 SNLI rows carry label -1 and are filtered; MultiNLI has none
- encoder reproduces Colab numbers on MPS to two decimals

**Scaling curve** (`results/phase3_debug_scaling.csv`) rules out an implementation
bug: untrained 49.58 -> 50k @ 2e-5 44.83 -> 50k @ 5e-5 49.14 -> 150k @ 2e-5 50.53.
Still climbing at every point. Two findings: **partial NLI training is worse than
none**, and the paper's 2e-5 is tuned for `bert-base` and too low for a
quarter-size model.

---

## Phase 4 — Domain adaptation

Full write-up in `results/phase4_notes.md`.

**Precision@10 45.77 -> 61.15 and STS 72.17 -> 74.54 at the same time.** The
phase target was to beat 45.77. TF-IDF still wins at 65.00, so the gap narrowed
from 19.23 points to 3.85 rather than closing.

### The project plan's premise was false, and had to be measured

`PROJECT_PLAN.md` assumes Swiggy's replies are templated by complaint category.
They are templated by **star rating** and carry nothing else — 0.7% of replies
name any complaint topic, 73 of 91 frequent replies go to exactly one rating, and
with the rating held fixed a TF-IDF classifier cannot predict which of 23
one-star templates was sent (8.12% vs 8.63% majority). Reproducible with
`python -m eval.reply_signal`. Pairing on replies would have taught the model
that a UPI outage and a late delivery mean the same thing.

**Used instead** (`python -m src.training.mine_pairs`, 53,061 pairs):
7,197 **mined** pairs where TF-IDF *and* the Phase 3 encoder independently agree
— the TF-IDF constraint is what stops the encoder echoing itself, and dropping it
yields ~85k pairs that drift back to matching on sentiment — plus 45,864
**simcse** self-pairs, where a false positive is impossible by construction.
Mined-pair precision is **~80%, not clean**; the failure mode is a shared
syntactic frame ("not giving discount" / "not giving cod option").

### Retrieval — the trained model finally beats GloVe, but not TF-IDF

| model | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| tfidf | 34.88 | **65.00** | 85.71 |
| glove-avg | 26.90 | 58.85 | 78.22 |
| bert-mean | 19.83 | 43.46 | 74.10 |
| sbert-distilroberta-300k | 20.57 | 45.77 | 72.25 |
| **sbert-domain** | 27.86 | **61.15** | 83.81 |

**Two checks that make this trustworthy.** Every previously-measured model
returned *identical* Precision@10 and MRR to Phase 3, to the decimal — only
Recall moved, and only because 67 more relevant reviews entered the denominator,
exactly as the Phase 3 lesson predicts. And pooling coverage is symmetric: 100%
for the three lexical/baseline systems, **96.15% for both trained models**, so
their comparison is fair and both are slightly understated against TF-IDF.

### STS went up, not down — the prediction was wrong

72.17 -> **74.54**, six of seven datasets improving. Specialisation was expected
to cost generic performance and did not. Contrastive training improves sentence
embeddings largely independently of the corpus: it spreads out the vector space,
the same anisotropy problem behind `bert-mean`'s 52.64 in Phase 2. Training on
Swiggy reviews fixed a defect that had nothing to do with Swiggy.

Checked because the result was surprisingly good: **zero overlap** between the
28,663 STS sentences and the 44,978 training texts, and Phase 3 reproduces its
Colab numbers exactly on MPS, so this is not a cross-machine artifact.

**Do not write "we beat the paper."** 74.54 exceeds the paper's 74.21 for
SRoBERTa-NLI-base, but their number is NLI training alone and ours adds a second
stage they never ran. Different recipe, not a better result on the same one.

### Open limitations

- **Part of the retrieval gain may be circular** — the mined pairs required
  TF-IDF to agree, so the model may have partly learned to imitate the system
  that wins this benchmark. Phase 4b is built to settle it.
- **The two pair sources are not separated.** `--sources mined|simcse` and
  `notebooks/phase4b_ablation_kaggle.ipynb` are ready; retrieval for the variants
  needs one more re-pool and labelling round, STS needs none.
- **Hinglish is still not bridged.** "khana thanda tha" vs "the food was cold"
  scores 0.066 against 0.049 for an unrelated pair. Phase 3 was no better.
- 14 pooled candidates on *"my Instamart order had a problem"* remain unjudged,
  10 of them since Phase 3.

### Lessons worth carrying

1. **Check a plan's stated premise against the data before building on it.**
   Two of Phase 4's three proposed strategies were dead, and one SQL query plus
   one classifier settled it in minutes.
2. **Smoke-test the training path locally before spending a GPU session.** Doing
   so caught three Kaggle-fatal problems: `sentence-transformers` 6 needs
   `accelerate`, its `losses`/`models` import paths are deprecated, and the
   Phase 3 anti-library guard needed confirming inert.
3. **Check tensor names after a cross-machine round trip.** Kaggle saved
   LayerNorm as `gamma`/`beta` where the local transformers uses `weight`/`bias`.
   The remapping happened, but had it not, those layers would have loaded
   randomly initialised and still produced plausible-looking vectors.

## Phase 5 — Theme discovery

Full write-up in `results/phase5_notes.md`.

UMAP + HDBSCAN over 45,864 distinct texts, run identically on three embeddings.
**110 themes from `sbert-domain`; 44,332 of 64,280 review rows assigned.**

| model | clusters | noise % | biggest cluster | silhouette | audit |
|---|---|---|---|---|---|
| glove-avg | 47 | **15.04** | **58.64%** | -0.0330 | 44.12% |
| bert-mean | 42 | 25.45 | 41.15% | 0.0502 | 73.53% |
| **sbert-domain** | 110 | 40.30 | **5.88%** | 0.1138 | **82.35%** |

### Noise percentage alone ranks the WORST model first

GloVe posts the best noise figure by refusing to separate anything: 58.6% of the
corpus sits in one cluster mixing "varry 👍👍 good swiggy" with refund complaints.
`largest_pct` was added to the comparison output so this cannot be read past.
**Never quote noise % without it.**

**Silhouette is a within-model diagnostic, not a comparison** — each is computed
inside that model's own space, and GloVe is 300d against the others' 768d.

### The blind audit settles it, but only partly

102 judgements, 34/model, model identity hidden (verified via `AppTest` that no
model name reaches the page), fixed-seed shuffle, deterministic sample.

| comparison | p (Fisher) | verdict |
|---|---|---|
| sbert-domain vs glove-avg | 0.0022 | significant |
| bert-mean vs glove-avg | 0.0258 | significant |
| **sbert-domain vs bert-mean** | **0.5597** | **NOT significant** |

**Do not claim the trained model beats plain BERT on the audit.** The 8.8-point
gap is inside the noise at n=34. The case over `bert-mean` rests on structure,
which is not a sampling question: 110 themes vs 42, and 5.88% vs 41.15% of the
corpus in the largest theme. A theme holding 41% of reviews is not actionable.

**Where GloVe actually fails:** 20 of its 34 audit rows came from its mega-cluster,
scoring 15.0% there against 85.7% outside it. Its small clusters are fine; it
simply cannot decide where the biggest group ends.

### Open limitations

- **The largest theme is a language, not a topic.** Theme 39 (2,855 reviews,
  `mere / karne / karte`) groups Hinglish reviews regardless of complaint. The
  Phase 4 finding that Hinglish was never bridged, now surfacing in the product.
  Reported rather than patched, by the user's decision.
- **Seven of the top fifteen themes are generic praise** (~13,600 reviews) — the
  predicted cost of the Phase 1 2+ word threshold.
- **40.3% noise**, the highest of the three. Deliberate: settings that lowered it
  did so by merging everything (`mcs=120, ms=None` gave 0.53% noise and one
  cluster of 45,222).
- One judge, one sample, no inter-annotator agreement.

### Lesson worth carrying

**A metric that rewards doing nothing will rank "nothing" first.** Noise
percentage looked like a coverage measure and was really a laziness measure.
The fix was not a better metric but a second one alongside it — `largest_pct` —
plus reading actual cluster contents during tuning rather than chasing
silhouette. Tuning purely on scores would have selected `mcs=120, ms=None`,
which puts 98.6% of the corpus in one "theme" and scores a merely mediocre
-0.03 silhouette rather than an obviously broken one.

## Phase 6 — Semantic search

Full write-up in `results/phase6_notes.md`.

**The two-stage searcher reaches 75.77 Precision@10 — the first system in this
project to beat TF-IDF's 65.00, which had won since Phase 2. Single-stage search
answers in 8.56 ms p50.**

| system | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| tfidf | 30.69 | 65.00 | 85.71 |
| sbert-domain, single-stage | 24.08 | 61.15 | 83.81 |
| faiss-ivf nprobe=10 | 20.85 | 53.08 | 83.80 |
| **faiss top-50 + cross-encoder** | **30.64** | **75.77** | **86.54** |

### The number this phase almost reported was WRONG

The first benchmark said reranking *dropped* Precision@10 to **40.77**. It was an
artifact. A reranker reorders 50 candidates, so it promotes reviews from ranks
11-50 that no bi-encoder ever put in a top-10 — never pooled, therefore scored as
irrelevant by default. Coverage was **47.3%** against faiss-exact's 96.2%.

Fixed with `eval/build_pool.py --augment --with-rerank` plus 127 new judgements;
coverage is now 96.2% for both. **The Phase 3 lesson applies to a new *ranking*,
not just a new *model*.**

**Precision@10 was unchanged for all five previously measured systems across the
pool revision** (65.00 / 58.85 / 43.46 / 45.77 / 61.15) while Recall fell for
every one of them — 91 more relevant reviews entered the denominator. Lesson 2
reproducing exactly. The alignment check in `eval/benchmark_search.py` therefore
verifies Precision and MRR and **deliberately not Recall**; an earlier version
hardcoded the old Recall value and reported a false MISMATCH.

### Latency (p50/p95 ms, 200 queries, warm-up discarded)

| stage | p50 | p95 |
|---|---|---|
| query encode | 6.89 | 17.58 |
| faiss exact | **1.52** | 2.48 |
| faiss IVF | 0.41 | 1.22 |
| cross-encode 50 | 61.20 | 185.67 |
| total 1-stage | **8.56** | 19.20 |
| total 2-stage | **71.26** | 200.51 |

Reranking costs 8.3x latency for +14.62 Precision@10 — worth it, and still
interactive. **FAISS is not the bottleneck**: exhaustive search over 45,864
vectors takes 1.52 ms while encoding the query takes 6.89 ms.

**FAISS's approximate index is not needed at this scale.** IVF is 3.7x faster and
costs 8.07 Precision@10. Saving 1.1 ms out of 8.56 ms by giving up accuracy is a
bad trade; ANN earns its place at millions of vectors, not 45,864. Do not imply
it was required here.

### The interview answer, with a measured number

50 candidates cost 61.20 ms, so cross-encoding all 45,864 would cost roughly
**56 seconds per query** against 8.56 ms for the bi-encoder — about 6,500x. A
bi-encoder embeds query and review separately so review vectors precompute once;
a cross-encoder scores a *pair*, so it precomputes nothing. That is the SBERT
paper's opening argument reproduced on this corpus.

### macOS bug — do not lose this

**FAISS and scikit-learn each link their own OpenMP runtime. Importing sklearn
BEFORE faiss segfaults the process — exit 139, no traceback, no output.**
Verified: faiss-first works; `OMP_NUM_THREADS=1` works but forces faiss
single-threaded (rejected — it would understate the latency being measured);
`KMP_DUPLICATE_LIB_OK=TRUE` does **not** help. `eval/benchmark_search.py` and
`eval/build_pool.py` import faiss at the top of their third-party block.

### Open limitations

- The cross-encoder is pretrained on English web search, not trained here. It
  inherits the Hinglish weakness: *"khana thanda tha"* (food was cold) returns
  *"kahana bht acha tha"* (food was very good) at rank 1.
- A reranker only reorders what stage one found, which is why recall improves far
  less than precision.
- 20 pooled candidates on one query remain unjudged; `faiss-ivf` sits at 88.8%
  coverage, so its 53.08 is a mild underestimate.
- Latency is single-query on an idle Mac — no concurrency or cold start.

## Phase 7 — The Streamlit dashboard

`streamlit run app/main.py`. Five pages, no new analysis — presentation of what
Phases 1-6 produced, plus the caching that makes it usable.

```
app/main.py               landing page; Streamlit auto-discovers app/pages/*.py
app/shared.py             sys.path fix, faiss-before-sklearn, both caches
app/pages/1_Overview.py   100,000 reviews, ratings, volume over time
app/pages/2_Themes.py     110 themes ranked, drill-down into any theme
app/pages/3_Trends.py     weekly volume, date + version filters, movers table
app/pages/4_Search.py     semantic search, single vs two-stage toggle
app/pages/5_Model_comparison.py   every results/ table with its caveats
```

`app/label.py` and `app/audit.py` stay **outside** `pages/` on purpose — they are
internal annotation tools that write to the database and have no place in a
dashboard someone is reading.

### The caching rule, since it is the difference between usable and not

Streamlit re-runs the entire script on every interaction, so anything expensive at
module level runs again on each keystroke.

- **`@st.cache_data`** memoises a returned *value*; Streamlit serialises it and
  hands each caller its own copy. For DataFrames from SQL and CSV.
- **`@st.cache_resource`** holds one *live object*, shared, never copied or
  serialised. For the encoder, the FAISS index, the cross-encoder.

Using `cache_data` on a model tries to serialise it; using `cache_resource` on a
DataFrame lets one page mutate what another sees.

**`get_search()` also runs one throwaway encode inside the cached call.** The
first query on MPS pays kernel compilation — measured at 1,944 ms against a warm
24 ms — and without the warm-up that cost is displayed to the user as the query
time. Same warm-up the Phase 6 benchmark discards.

### Correctness points enforced on the pages

- **Overview uses all 100,000 rows, never the 64,280 themed subset.** Filtering
  short reviews raises the 1-star share, so a rating chart on the subset is
  simply untrue. `shared.ALL_REVIEWS` / `shared.THEMED_REVIEWS` make the choice
  explicit, and `shared.corpus_note()` states both numbers wherever theme counts
  appear.
- **"Days covered" is `count(DISTINCT reviewed_at::date)`, not max minus min** —
  the counted version is what would expose a collection gap. Both give 221, so
  there are none. (PROGRESS previously said 220; 221 is the inclusive count.)
- **Trends ranks movers by share of reviews, not raw count.** Overall volume
  moves week to week, so a theme can gain reviews without becoming more of a
  problem. Themes under 30 reviews across both periods are excluded as too noisy.
- **The version filter offers only versions with 1,000+ reviews** — 24 of 285,
  covering 90.7%. Most versions have a single review from someone on an ancient
  build; a dropdown of 285 is unusable. 10,925 reviews have no version at all.
- **Every caveat from the phase notes is on the Model comparison page**, not in a
  footnote: noise % ranking the worst model first, silhouette not being
  comparable across models, sbert-domain vs bert-mean not being significant, and
  the GloVe baseline being 7.4 points below the paper's.

### Found while building

The Trends movers table immediately surfaced a real emergent complaint: a theme
labelled **"rain / gst / fee" grew from 15 to 83 reviews** across the midpoint
split, the largest proportional jump on the board. Phase 8's alerting should fire
on exactly this shape.

## Phase 8 — Trends and alerts

`theme_weekly` (3,342 rows: 110 topics x 31 complete weeks) and `theme_alerts`
in Postgres. Detection is a z-score against a trailing 8-week mean, kept simple
on purpose — the user's brief was that a defensible simple method beats an
unexplainable sophisticated one.

**Partial weeks are dropped at the source** (`src/analytics/weekly.py`), not
patched downstream. The corpus ends on a Wednesday, so the newest bucket held 3
days and ~1,095 reviews against a normal ~3,000; a naive weekly rule reports that
every topic collapsed at once. A second partial week at the start (2026-01-12,
1 day) was found the same way.

### The threshold is 3.0, and the data says why

110 topics are tested every week, so the multiple-comparison arithmetic decides
this, not taste. Measured across thresholds:

| threshold | expected from chance | alerts found |
|---|---|---|
| z >= 3.0 | 3.4 | **16** |
| z >= 2.5 | 15.7 | 28 |
| z >= 2.0 | 57.6 | 46 |
| z >= 1.5 | 169.0 | 67 |

**At z >= 2 the list would contain fewer alerts than chance alone predicts** — it
would be entirely noise. At z >= 3 the yield is roughly five times chance.

### Four guards, each for a named failure

- **MIN_REVIEWS = 15.** A topic going 1 -> 4 has z > 3 and means nothing; weekly
  counts are Poisson-ish and small numbers are mostly noise.
- **Zero variance.** A flat topic has sd = 0 and z is undefined. Falls back to
  `sd = sqrt(mean)`, the Poisson standard deviation — the right scale for counts,
  rather than an arbitrary epsilon.
- **EFFECT = 1.5x the baseline mean.** Statistically unusual and operationally
  trivial are different things.
- **share_z.** The same test on the topic's *share* of the week, reported beside
  the count. Two of the 16 alerts are marked "share flat — the week was simply
  busier", which is the volume confound caught rather than hidden.

### Four failure modes no guard fixes — state them, do not pretend

- **Slow burn is invisible.** A topic growing 10% a week drags its own trailing
  mean up and never trips. This catches jumps; the Trends screen catches trends.
- **Counts are right-skewed**, so a positive z overstates its rarity. "1 in 740"
  is an order of magnitude, not a probability.
- **Weeks are not independent.** A two-week incident raises the baseline and
  suppresses later alerts about the same problem.
- **No seasonality model.** A festival week lifts everything and the rule blames
  the topics rather than the calendar.

### What it found

16 alerts across 9 weeks. **Six topics fired in the single week of 2026-07-20**,
including "Poor customer support" at z = 19.73 (59 reviews against a baseline of
19.6 +/- 2.0). Several topics firing together is usually one incident rather than
several problems, and the Alerts screen says so rather than listing six rows.

### Also delivered

`docs/POWERBI.md` — connection steps, model advice, a four-visual executive
layout, and **which visuals mislead on this data**: pie of topic share (110
slices, and share is of the filtered subset), dual-axis count-vs-rating, stacked
area over a changing topic set, word clouds (they rank by raw frequency and
return "order, app, food" — exactly what this project exists to look past), and
week-on-week percentage change without a count floor.

## Design pass — making it look and read like a product

Four rounds of UI feedback had not landed. The turning point was screenshotting
every screen with a real browser and looking at them, instead of trusting that
they were fine because they returned HTTP 200. That surfaced three separate
problems, only one of which was taste.

**Two things were broken.** `app/views/home.py` contained its whole body twice,
so the landing page rendered everything double — leftover damage from the same
bad string surgery that broke four files the session before. And `.pill`, used on
the Topics screen, was never defined in the stylesheet, so that badge rendered as
raw unstyled text.

**84 of the 110 topics had no name.** `themes.display_name` was NULL for all but
26, so every screen fell back to raw c-TF-IDF word soup: `poor / customer /
support`, `late / delivery / bad`, `gst / fee / charges`, `नह / कर / डर`. This was
the real cause of "labels nobody can understand" — a data problem that no amount
of CSS would have fixed. `src/clustering/theme_names.py` now names all 110, each
written from that topic's terms plus a sample of its actual reviews, and each
still guarded by a term that must remain in `top_terms` so a re-clustering cannot
silently misname one. `--check` reports 110 of 110 applied, 0 refused.

**Nothing on any screen was louder than anything else.** The plane `#f7f7f5` and
the cards `#ffffff` sat 1.07:1 apart — invisible — and all seven screens shared
one rhythm: bar, three identical tiles, heading, full-width chart. The plane is
now `#eef1f6` (1.13:1) with real shadows, and each screen opens with `design.hero()`
— one dark band carrying one sentence and one large number, so the reader lands
on the finding rather than on three numbers of equal weight.

Also done: the four non-actionable topics (two grouping by language, two holding
angry reviews with no detail) are excluded from the business screens through a new
`themes.actionable` column, which Power BI inherits, while still counting in every
total and staying visible on *How it works*; nav labels became plain nouns
(Ratings, Topics, Trends, Search); `.streamlit/config.toml` sets the accent to the
brand blue, which Streamlit's own widgets take from there rather than from CSS;
and *How it works* now uses one unit throughout, having previously shown `65%` in a
chart and `6.5 in 10` in the tile directly below it for the same quantity.

Verified: `eval/check_app.py` 7/7; `theme_names --check` 110/110; Delivery's
figures cross-checked against a direct SQL count and unchanged (2,088 unhappy of
3,091); every screen photographed and looked at.

## Phase 9 — the write-up

`README.md`, written for a hiring manager with two minutes: the business problem
first, a screenshot second, the model third. Every claim in it is checked by a
script rather than trusted.

**`eval/check_readme.py` is the piece worth keeping.** It reads 33 figures out of
the CSVs in `results/` and asserts each appears verbatim in the README, so a later
re-run of an eval script cannot leave a stale number sitting in the write-up. It
caught two things immediately: an audit figure where Python's `format()` rounds
82.35 down to 82.3 while a person computing 28/34 correctly writes 82.4 (fixed with
`Decimal(ROUND_HALF_UP)`), and an ablation score the README only ever quotes as a
delta.

It also caught a number I had invented. The README first said encoding takes
"roughly 6 minutes"; nothing had measured that. Measured properly — 2,048 texts
after warm-up — it is **184 texts/second on `mps`**, so about 4 minutes for 45,864.

Cleanup done in the same pass: six dead functions removed (`tiles`, `rating_chip`,
`trend`, `sparkline` in `app/design.py`, superseded by `hero`/`rank_rows`; `page`
and `corpus_note` in `app/shared.py`, superseded by `design.appbar`) along with
their orphaned CSS, plus `load_all` in `eval/sts_data.py`. `requirements.txt` is now
fully pinned, including `numpy` — imported directly, previously arriving only via
pandas — and `playwright==1.62.0`.

## Second usability pass — the app now explains itself

The user's report: *"Once I close the sidebar, I am not able to open it back...
It is not able to convey the idea it is supposed to show... The labels are so
vague that it is not making any sense... If you can provide an opening page, that
would be more helpful."* Three separate problems, one of them a real bug.

### The sidebar was a trap, and no existing check could have caught it

`app/design.py` carried `#MainMenu, footer, header { visibility: hidden; }` —
boilerplate copied into most Streamlit apps to remove the Deploy button. When the
sidebar is collapsed, Streamlit 1.62 renders the reopen control as
`[data-testid="stExpandSidebarButton"]` **inside `<header data-testid="stHeader">`**.
Hiding the header hid it too. Confirmed with Playwright: after a collapse the
button sits at (67, 16) with `visibility: hidden`, and a sweep of the top-left
300x120 region returns **zero visible buttons**. Reloading the page was the only
way back.

The first attempted fix — hide `[data-testid="stToolbar"]` instead — **failed the
same way**, because the reopen button is nested three divs inside the toolbar.
The working fix hides only `[data-testid="stToolbarActions"]` and
`[data-testid="stAppDeployButton"]`, leaving both the header and the toolbar in
place, and gives the reopen button card styling so it reads as a control.

Why nothing caught it: `eval/check_app.py` renders every screen and they all
rendered; `eval/shoot_app.py` photographs the default state, where the sidebar
starts expanded and everything looks perfect. The bug only exists in a state no
check ever entered. `eval/shoot_app.py` now has `sidebar_reopens()`, which
collapses the sidebar, asserts the reopen button is visible, clicks it for real,
and asserts the sidebar comes back. It fails loudly if the CSS regresses.

### A fourth screen, "Start here"

Every other screen assumed the reader already knew whose reviews these are and
what was done to them. `app/views/start.py` is the default page now: a hero
stating the corpus and its span, three cards naming the questions the app answers
(each linking to the screen that answers it), and three numbered steps describing
what the system actually does. No controls, nothing to interpret, under 200 words.
Every figure on it is queried, not typed, so it cannot go stale.

It deliberately reports **106 topics, not 110** — the browsable count after the
four non-actionable groups are excluded — so the opening screen agrees with the
screen it links to. "How it works" remains the one place that names the four.

### Labels now say what the thing is

Navigation titles name a question rather than a location: Overview -> **What to
fix**, Explore -> **Topics & search**. Section headings followed: "Suddenly worse"
-> "Problems that spiked in the last 4 weeks", "Ratings overall" -> "How
customers rate the app", "Most talked about" -> "Every topic customers raised".
The two rating charts on What to fix had **no titles at all** and now have them.

One label was not just vague but wrong: the search screen's headline number was
`len(hits)`, which is always 25 because that is the `k` passed to `search()`. It
now sums `n_rows` — how many people actually wrote something that close.

`url_path` values were left unchanged, so existing links and the screenshot
filenames in the README still resolve.

## Third design pass — rebuilt to a supplied reference

The user dropped a zip into ~/Downloads and said "this is how I want my
streamlit app to be". It was not a mockup: it was a working four-screen
Streamlit app with its own design system, information architecture and copy
rules, running on invented demo data. It was unpacked, run on port 8599 and
photographed before any of it was adopted, so the decisions below are against
what it renders rather than what its README claims.

### What was taken

An editorial, flat, Swiss look — Archivo at weight 800, `border-radius: 0`
everywhere, a warm off-white ground, one hot red accent, 2px rules instead of
cards and shadows. Statement headlines. A stat row. And the single best idea in
it: the *groups by meaning* demonstration — three differently-worded quotes, an
arrow, one problem box — which explains the project in five seconds and which
none of the previous designs attempted.

The demonstration uses **real** reviews, queried from the theme they actually
belong to. "worst delivery time 1hr delay", "waits for more than an hour and
then says food is cancelled", "Wasnt able to connect to customer care more than
half an hour" → *Waited an hour or more*, 761 reviews at 1.1 stars.

### What was rejected, and why

**Its navigation does not work.** `theme.py` sets
`[data-testid="stSidebarNav"] { display: none }`. Confirmed in its own running
DOM: computed display is `none`, so no screen is reachable except through one
button. Our nav is kept and restyled; the wordmark is drawn as CSS pseudo-
elements on the sidebar header, because Streamlit fixes the sidebar's child
order and anything written with `st.sidebar` lands *below* the menu.

**Every number in it is invented**, at roughly 26x the real volume, and several
contradict the measured results. Corrected on the way in: "2 in 3 search results
on topic" → 7.6 in 10; "7 in 8 reviews in a sensible group" → 8 in 10; "a random
sample of 400 reviews" → 102, blind; "P@10 65.0 vs 40.0 for BM25" → 65.0 *is*
the TF-IDF baseline and there is no BM25 run; "1 review in 8 filed elsewhere" →
about 1 in 6.

**Its weekly frame does not survive real data.** The user was shown this before
deciding: the last complete week holds 2,889 reviews and puts 47 behind the
biggest problem at -16% against normal, so a weekly headline reads as nothing
happening. The ranked list is eight weeks; spikes stay weekly, because that is
what a spike is.

**Its sentence "they fall into the problems below" is an overclaim** at real
volumes. 26,010 reviews arrived in the window and 10,481 of them could be placed
in a problem at all. The screen now says both numbers.

### The palette was measured, not eyeballed

There is no JS runtime on this machine, so the validator was ported to Python
rather than skipped. One result changed a decision: **accent-400 #ff9783 is
1.88:1 against the ground**, under the 3:1 floor, and is the reference's bar
fill. A contrast warning is dischargeable only by a mandatory second cue, so it
stays for bars — which always print their name and number — and is replaced by
the full accent for sparklines, where a 5px mark has no label to fall back on.
Stars are drawn in two states rather than three, because a diverging scale needs
two hues and this system supplies one.

### Bugs found in our own code on the way

- Forcing Archivo onto every element broke Streamlit's Material icon ligatures,
  so the sidebar collapse control rendered the literal text
  "keyboard_double_arrow_left". Icon spans are now exempt from the font rule.
- `st.session_state["query"] = example` raised `StreamlitAPIException`: a key
  owned by an instantiated widget cannot be assigned. The seed now lives under
  its own key and is passed as `value=`, with the input given no key so its
  identity follows its arguments.
- Sparklines dated back from `max(week_start)` returned a ragged bar count — the
  caption said eight while the chart showed thirteen. Trimmed in pandas instead.

### The copy rule is now machine-checked

The reference's rule that method vocabulary appears nowhere outside screen 4 is
enforced rather than hoped for: `eval/check_app.py` fails the run on jargon in
screens 1-3 and merely notes it on `accuracy.py`, and the pattern was extended
to cluster, embedding, HDBSCAN, UMAP, FAISS and c-TF-IDF. Arrows were removed
from the emoji pattern — "scraper -> Postgres" is typography, and the old range
banned it on one screen while letting the identical glyph through on another
purely because that one was written as `&rarr;`.

## Fourth design pass — the v2 reference: top tabs and a weekly briefing

A second zip arrived and the user asked for "this exact design and UI". It was
again a working Streamlit app rather than a mockup, so it was unpacked, run on
port 8598 and photographed before anything was adopted.

The palette was unchanged from v1 and stands. What changed was the shell: a dark
**ink header** carrying a permanent one-line definition, **top tabs instead of a
sidebar** (which retires the reopen bug rather than maintaining it), a **weekly
briefing** as screen one, and an **evidence rail** on the fix list.

### The bug in the reference, and the fix

**Streamlit cannot wrap its own widgets in arbitrary HTML.** The reference opens
`<div class="navrow">`, renders its tab buttons, then closes it. Streamlit
auto-closes the div first, so the wrapper comes out empty — its own DOM reports
one `.navrow` holding **zero buttons**, and none of the tab styling ever applied.
Three visible consequences in its own screenshot: tabs rendered as ordinary
outlined buttons, the call-to-action button sat hard against the window edge, and
a white gap tore through the red decision poster where two columns met.

`st.container(key=...)` emits a real element carrying `st-key-<key>`. That was
proved with a standalone probe before any of this file was written, and
`eval/shoot_app.py` now asserts it on every run — the exact check the reference
fails. The second rule that came out of this: **any block with no widget in it is
one `st.markdown` call**, never `st.columns`, which is what removes the poster
seam.

### Two honesty decisions the user made

**Owners and "Do next" actions were dropped.** The reference puts a suggested
owner and a recommended action on every row and footnotes both as illustrative.
There is no owner or action data in this database. That costs the design its
"decision list" framing, and the user chose it anyway: nothing on the site should
need a footnote saying it is invented.

**The briefing reports the real week**, 17-23 Aug 2026 — 2,889 reviews, 932 of
them 1-2 stars — named by date, because the corpus is static and a screen headed
"this week" would be a fiction. The decision block uses the sharpest spike *on
record* with its true date rather than manufacturing urgency for a quiet week.

### Bugs found in our own work while building it

- **`app/copy.py` shadowed the stdlib `copy` module**, so every screen failed
  with `module 'copy' has no attribute 'ONE_LINER'`. Renamed to `prose.py`.
- **The briefing named a compliment as its largest problem.** The query filtered
  on `actionable` but not on rating, and the biggest group that week is
  "Praise: 'good'". Now filtered to `avg_rating <= 2.5`.
- **The movers were noise.** A floor of five reviews produced "+47%" on a base of
  six. Raised to ten a week, and the screen now says so.
- **The browse rail listed praise as problems** — nine of its top fourteen were
  compliments. Filtered to the 57 complaint groups, with the 106 total stated.
- **The active tab lagged the content by one click.** A button reports its click
  on the same run, after the tab row has already been drawn from the old key, so
  the content switched while the underline stayed put. Fixed with `st.rerun()`.
- **`gap: 0`, needed to make the full-bleed bands sit flush, also collapsed the
  spacing inside every fix-list row**, overlapping rank, title and button. Rows
  are keyed containers with their own padding.
- **`check_app` failed every screen on "──"**, because the stylesheet is itself
  an `st.markdown` call and its section rules were being scanned as emoji. Style
  blocks are now excluded, and the emoji pattern narrowed to emoji blocks — a
  true minus sign in "−21%" is typography.

### Measured, not eyeballed

The reference uses `#bab6b6` for the fix-list sparklines. At **1.80:1** against
the ground that is invisible, and a 5px mark carries no label to fall back on, so
`spark()` uses `neutral-700` at 5.83:1 instead.

## Decisions made

| decision | why |
|---|---|
| GitHub account `ayn-aval`, repo `ayn-aval/Echo` | personal portfolio account, not the work identity in the global git config; set repo-locally |
| PostgreSQL, not CSV | Phase 5 needs joined tables; the scraper's resume needs transactional writes; SQL matters for the target roles |
| Corpus capped at 100,000 | user's call, keeps scrape and Phase 5 encoding time down |
| Cleaning threshold 2+ words | user's call over the recommended 4+, to keep short genuine complaints |
| glove.840B.300d | matches the paper; streamed and vocabulary-filtered so RAM is ~40 MB not 2.6 GB |
| Paper-exact training config | so a shortfall cannot be blamed on batch size or LR |
| Debug runs local, real runs on Colab | faster iteration; no Colab GPU spent on a broken loop |
| Training code in `src/`, notebook as thin driver | notebooks are JSON and undiffable; the loop is the portfolio artifact |
| GPU runs on Kaggle, not Colab | user's call; `CLAUDE.md` and `PROJECT_PLAN.md` still say Colab. Kaggle mounts data read-only at `/kaggle/input/`, writes to `/kaggle/working/`, and has no Drive |
| Phase 4 pairs: mined + SimCSE | user's call after seeing real example pairs from all four candidate strategies; the two reply-based strategies in the plan were measured and rejected |
| Clustering at `min_cluster_size=60, min_samples=None` | user's call from the sweep, chosen by reading real cluster contents; best silhouette and coherent themes, at the cost of the highest noise |
| Hinglish language-cluster reported, not patched | user's call; splitting the corpus by `lang` would recover the themes but makes the three-way comparison non-comparable |
| Standalone `hdbscan`, not sklearn's | it compiled cleanly on Apple Silicon, so no substitution from the stated stack was needed |
| Exact FAISS index in production, IVF measured only | at 45,864 vectors ANN saves 1.1 ms and costs 8.07 Precision@10 |
| Two-stage rerank kept despite 8.3x latency | +14.62 Precision@10 and still 71 ms p50, comfortably interactive |
| `app/pages/` auto-discovery over `st.navigation` | simplest thing that works for five pages; the tradeoff is less control over sidebar titles |
| Dashboard shows `sbert-domain` themes only | model switching belongs on the comparison page, not scattered through the product pages |

## Known rough edges

- The scraper overshoots `--limit` by up to one batch (200).
- `data/scrape.log` is unreadable in an editor — tqdm uses carriage returns.
  Watch progress in SQL: `SELECT ..., now() - updated_at AS idle FROM
  scrape_checkpoints`. The `idle` column is the reliable signal; the progress bar
  keeps redrawing even when nothing arrives.
- 24 of the 50 evaluation queries were never labelled (only 26 were needed).
  Labelling the rest would steady the retrieval averages.
- Ten pooled candidates on one query remain unjudged (~1%).

## Process notes for future sessions

**A Playwright `full_page` screenshot of a Streamlit app captures only the first
screen.** Streamlit scrolls an inner container rather than the document, so
`full_page=True` silently crops everything below the fold and gives no warning.
Use a tall viewport instead — `eval/shoot_app.py` uses 1600x2400. The first
version of that script looked like it was working and had never once shown me the
bottom half of any page.

**Streamlit does not reload `app/design.py` on change.** It re-runs the entry
script but keeps imported modules in `sys.modules`, so after editing a module the
running server serves the old one and the page fails with a stale `AttributeError`
that reads like a code bug. Restart the server after touching anything under
`app/` that is not a view.


- **`AppTest` does NOT report a SyntaxError.** It prints one to stderr and then
  returns an AppTest whose `.exception` is `None`, so a file that does not even
  parse reports as passing. Four screens shipped broken this way, verified "ok"
  by a check that suppressed stderr. **Run `python -m eval.check_app`**, which
  parses every file with `ast.parse` first, captures stderr during render, and
  exits non-zero. Verified by deliberately breaking a file and confirming it
  fails.
- **An HTTP 200 from Streamlit proves nothing about a page.** The server returns
  the app shell before any page code runs, so every route answers 200 even when
  every page is broken. `curl` is not a test of a Streamlit screen.
- **Index-based string surgery on source files is how those four broke.**
  Replacing from a found offset to a computed end left the `)` that closed the
  original call. When patching code, re-parse the file afterwards.

- **Use `streamlit.testing.v1.AppTest` to check any page under `app/`.** Two
  Streamlit bugs shipped because they were verified in modes that could not
  reproduce them: `sys.path[0]` is the script's directory under Streamlit (not the
  project root), and bare mode skips duplicate-element-key registration.
  **AppTest clicks write to the real database — do not click Save when testing.**
- **When patching a file with `str.replace()`, assert the old text was found.**
  This file sat stale for two whole phases because non-matching edits reported
  success.
- **A guard that silently skips is the same failure in another costume.**
  `eval/ablation.py` gated its CLAIM summary on a row named `concat:u,v,|u-v|`
  that `configs()` deliberately never produces — that configuration *is* the
  `pooling:mean` run. The condition was never true, so the ablation printed its
  table and silently omitted both findings. Fixed at `eval/ablation.py:92-96`.
  **When a block of output is conditional, make the else branch say why it
  skipped.**

## Exact next step

**Phase 9 is complete, and the app has had two usability passes.** Nothing is
blocking. The one unfinished piece of *delivery* is the **Streamlit Community
Cloud deploy**, which last failed on Python 3.14 resolving `requirements.txt`;
the app must be deleted and recreated with Python 3.11 chosen in Advanced
settings before the first build, with the secret from `.streamlit/secrets.toml`.
Once it is live, its URL goes in the README.

**Three credentials were pasted into chat and should be rotated:** the Hugging
Face write token named `Echo` (highest priority), the Neon `neondb_owner`
password, and the `echo_readonly` password.

Still outstanding:
- **Phase 4b**, the pair-source ablation — `notebooks/phase4b_ablation_kaggle.ipynb`,
  ~20 min on Kaggle, never run.
- **~200 more theme-audit judgements** would settle sbert-domain vs bert-mean
  (currently p=0.56).
- 20 unjudged pooled candidates on one query; 24 of 50 eval queries never labelled.
- `CLAUDE.md` and `PROJECT_PLAN.md` still say Colab for training; runs use Kaggle.
  Ask before editing — they are portfolio documents.
- The user asked whether generic-praise topics should be hidden entirely from the
  dashboard, and whether to add business-impact estimates. Both need their input.
  (The four *non-actionable* topics are now hidden via `themes.actionable`; the
  generic-praise question is separate and still open.)

## Commands

```bash
source venv/bin/activate
python -m eval.run_sts          # three baselines, seven STS datasets
python -m eval.run_retrieval    # Recall@10 / Precision@10 / MRR
python -m eval.compare_paper    # side-by-side against the paper's Table 1
python -m eval.diagnostics      # structural checks + scaling curve
python -m eval.build_pool --augment   # re-pool after training a new model
streamlit run app/label.py      # relevance labelling
python -m eval.reply_signal     # Phase 4: are replies category- or rating-templated?
python -m eval.run_sts_trained  # STS for every trained encoder on disk
python -m src.training.mine_pairs        # rebuild the Phase 4 training pairs
python -m src.training.train_domain      # Phase 4 fine-tuning (also runs on Kaggle)
python -m src.embeddings.encode_corpus   # Phase 5: vectors + review-id mapping
python -m src.clustering.tune --model sbert-domain      # HDBSCAN parameter sweep
python -m src.clustering.name_themes --model sbert-domain
python -m eval.clustering_comparison --persist          # three-way theme comparison
streamlit run app/audit.py      # blind hand-audit of theme assignments
python -m src.search.index                       # Phase 6: build FAISS indexes
python -m src.search.query "app keeps crashing"  # semantic search
python -m src.search.rerank "refund not received"   # two-stage search
python -m eval.build_pool --augment --with-rerank    # pool the reranker BEFORE scoring it
python -m eval.benchmark_search  # accuracy + p50/p95 latency
streamlit run app/main.py       # the dashboard
python -m src.clustering.theme_names       # readable topic names
python -m src.clustering.theme_categories  # roll topics into business areas
python -m src.analytics.weekly             # Phase 8: weekly series (drops partial weeks)
python -m src.analytics.alerts --explain   # z-score alerts with their baselines
```
