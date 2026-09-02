# Progress

## Current status

**Phases 0, 1, 2, 3 and 4 complete.**

| phase | state |
|---|---|
| 0 — Setup | done (Streamlit hello-world skipped; not needed until Phase 7) |
| 1 — Data collection | done — 100,000 Swiggy reviews in Postgres |
| 2 — Baselines + eval harness | done — STS and review-retrieval baselines measured |
| 3 — Reproduce SBERT | **done. 72.17 vs paper's 74.21; ablation run** |
| 4 — Domain adaptation | **done. Precision@10 45.77 -> 61.15, STS 72.17 -> 74.54** |
| 5 — Theme discovery | not started |

**Immediate next step:** the user's call between Phase 5 (theme discovery) and
Phase 4b (the pair-source ablation, set up but not run —
`notebooks/phase4b_ablation_kaggle.ipynb`). The README write-up is still
undrafted; material is in `results/phase3_notes.md` and `results/phase4_notes.md`.

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

**Phase 4 is complete.** Two options, the user's call:

**A — Phase 4b, the pair-source ablation.** Set up but not run:
`notebooks/phase4b_ablation_kaggle.ipynb`, two runs on the existing
`echo-phase4` Kaggle dataset, ~20 minutes. It answers the open question of
whether the mined pairs or SimCSE produced the gain, and whether the retrieval
improvement is partly the model learning to imitate TF-IDF. STS comes free;
**retrieval for the variants needs one more re-pool and labelling round.**

**B — Phase 5, theme discovery.** Encode all reviews with `sbert-domain`, then
UMAP + HDBSCAN, and cluster three times (GloVe / plain BERT / sbert-domain) for
the comparison table. Note from Phase 1: expect large generic-praise clusters as
a direct consequence of the 2+ word threshold, and report them rather than hide
them.

Also outstanding:
- The honest README paragraph, material in `results/phase3_notes.md` and
  `results/phase4_notes.md`.
- 14 unjudged pooled candidates on *"my Instamart order had a problem"*.
- 24 of the 50 evaluation queries were never labelled.
- `CLAUDE.md` and `PROJECT_PLAN.md` still say Colab for training; actual runs
  use Kaggle. Ask before editing — they are portfolio documents.

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
```
