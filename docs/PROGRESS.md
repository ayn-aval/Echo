# Progress

## Current status

**Phases 0, 1, 2 and 3 complete — ablation included.**

| phase | state |
|---|---|
| 0 — Setup | done (Streamlit hello-world skipped; not needed until Phase 7) |
| 1 — Data collection | done — 100,000 Swiggy reviews in Postgres |
| 2 — Baselines + eval harness | done — STS and review-retrieval baselines measured |
| 3 — Reproduce SBERT | **done. 72.17 vs paper's 74.21; ablation run** |
| 4 — Domain adaptation | not started. Target: beat 45.77 Precision@10 |

**Immediate next step:** Phase 4 — domain adaptation. `sentence-transformers` is
allowed from here on. The README write-up is still undrafted; material for it is
in `results/phase3_notes.md`.

---

## Phase 1 — Data collection

100,000 Swiggy reviews (`in.swiggy.android`), 18 Jan – 26 Aug 2026, 220 days,
zero duplicates, 46 MB. Scraped entirely from the unfiltered stream — the five
per-rating backup streams were never needed.

- **99.9% carry a Swiggy reply.** This is the Phase 4 weak-supervision signal and
  it is close to universal, stronger than the brief assumed.
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

**Phase 4 — domain adaptation.** Concrete target: **beat 45.77 Precision@10**
(TF-IDF's 65.00 is the number that would make the trained model genuinely useful).
`sentence-transformers` is allowed from Phase 4 onward; Phase 3's raw-PyTorch rule
does not extend forward.

1. **Weak-supervision pairs from the Swiggy data.** The signal is confirmed: 99.9%
   of reviews carry a Swiggy reply, and the replies are templated by complaint
   category. Propose two or three pairing strategies (review-with-its-own-reply;
   two reviews sharing a near-identical reply; others), show real example pairs
   from each, and recommend which gives the cleanest positives — then the user
   picks.
2. **Continue fine-tuning the Phase 3 encoder** with `MultipleNegativesRankingLoss`.
   Starting point: `models/sbert-distilroberta-300k/encoder` (gitignored).
3. **Re-pool before evaluating.** `python -m eval.build_pool --augment` — this is
   mandatory for any new model (lesson 1 above), or the comparison is invalid.
4. **Three-way table:** baseline vs Phase 3 SBERT vs domain-adapted SBERT, on
   Recall@10 and MRR. Re-run STS as well and explain the tradeoff if generic
   performance degraded — a drop there is an expected cost of specialisation, and
   is reported, not hidden.

Still outstanding from Phase 3: the honest README paragraph, material in
`results/phase3_notes.md`.

## Commands

```bash
source venv/bin/activate
python -m eval.run_sts          # three baselines, seven STS datasets
python -m eval.run_retrieval    # Recall@10 / Precision@10 / MRR
python -m eval.compare_paper    # side-by-side against the paper's Table 1
python -m eval.diagnostics      # structural checks + scaling curve
python -m eval.build_pool --augment   # re-pool after training a new model
streamlit run app/label.py      # relevance labelling
```
