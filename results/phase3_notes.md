# Phase 3 — Sentence-BERT reproduction

`distilroberta-base`, 300,000 stratified SNLI + MultiNLI pairs, one epoch, batch 16,
Adam at 2e-5, linear warmup over the first 10% of steps, mean pooling, classifier on
`(u, v, |u−v|)`. Trained on a Colab T4 with a siamese loop written by hand in raw
PyTorch — `sentence-transformers` is not imported anywhere in this phase.

## Table 1 comparison

| | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg |
|---|---|---|---|---|---|---|---|---|
| Avg. GloVe (ours) | 57.25 | 53.79 | 52.00 | 59.70 | 49.83 | 47.37 | 57.21 | 53.88 |
| Avg. GloVe (paper) | 55.14 | 70.66 | 59.73 | 68.25 | 63.66 | 58.02 | 53.76 | 61.32 |
| Avg. BERT (ours) | 30.87 | 59.89 | 47.73 | 60.29 | 63.73 | 47.29 | 58.65 | 52.64 |
| Avg. BERT (paper) | 38.78 | 57.98 | 57.98 | 63.15 | 61.06 | 46.35 | 58.40 | 54.81 |
| BERT CLS (ours) | 21.54 | 32.11 | 21.28 | 37.89 | 44.24 | 20.30 | 42.74 | 31.44 |
| BERT CLS (paper) | 20.16 | 30.01 | 20.09 | 36.88 | 38.08 | 16.50 | 42.63 | 29.19 |
| **SBERT distilroberta 300k (ours)** | 67.20 | **73.71** | 69.11 | 75.58 | 72.35 | 75.60 | 71.63 | **72.17** |
| SRoBERTa-NLI-base (paper) | 71.54 | 72.49 | 70.80 | 78.74 | 73.69 | 77.77 | 74.46 | 74.21 |
| SBERT-NLI-base (paper) | 70.97 | 76.53 | 73.19 | 79.09 | 74.30 | 77.03 | 72.91 | 74.89 |

**72.17 against the paper's 74.21 for its closest model — a 2.04 point gap on 32% of
the training data.** We exceed the paper on STS13 (+1.22).

## Was the gap a bug or the smaller subset?

Checked before concluding, all reproducible via `python -m eval.diagnostics`:

| check | result |
|---|---|
| Step-1 loss vs theoretical ln(3) = 1.0986 | 1.0901 |
| Loop can learn (192 pairs, 40 epochs) | loss 0.0011, accuracy 1.00 |
| One shared encoder, not two | params total == encoder + head exactly |
| Pooling ignores padding (mean and max) | pass |
| Trained weights actually saved | 101/103 tensors changed; matches checkpoint |
| Resume restores step and optimizer state | pass |
| NLI labels 0/1/2, balanced, no −1 survivors | 785 SNLI rows filtered |

**The scaling curve is the decisive evidence** (`results/phase3_debug_scaling.csv`,
MiniLM-L6 debug model, STS avg over STS-B/12/16):

| training pairs | STS avg | vs untrained |
|---|---|---|
| 0 (untrained control) | 49.58 | — |
| 50k @ 2e-5 | 44.83 | −4.75 |
| 50k @ 5e-5 | 49.14 | −0.44 |
| 150k @ 2e-5 | 50.53 | +0.95 |

Performance still climbs with more data at every point measured, which is what a
data-limited result looks like. A flat curve well below the paper would have meant a
bug regardless of scale.

Two findings from that curve worth keeping:

1. **Partial NLI training is worse than none.** At 50k pairs the model scored *below*
   an untrained control of the same architecture. Fine-tuning first disturbs the
   geometry that masked-language-model pretraining produced, and only rebuilds
   something better once there is enough signal. A reproduction that stopped at 50k
   would have concluded, wrongly, that the method does not work.
2. **The paper's 2e-5 is tuned for `bert-base`.** On a model a quarter that size it
   was too low; 5e-5 recovered 4.3 of the 4.75 lost points. `distilroberta-base` is
   82M against `bert-base`'s 110M, so 2e-5 was right for the real run.

## An honesty note about our GloVe baseline

**Our GloVe row is 7.4 points below the paper's (53.88 vs 61.32), and that flatters
our result.** We report the trained model beating our GloVe by +18.29; the paper's
own margin over its GloVe is +12.89. The extra ~5 points are our baseline being
weak, not our model being strong.

The cause is the aggregation choice documented in `baselines_notes.md`: we compute a
single Spearman over all pairs, while SentEval — which the paper used — scores each
sub-dataset separately and averages. Measured directly on STS-B with GloVe: 44.11
pooled versus 50.07 per-subset, a ~6 point difference that accounts for most of the
gap. Our BERT rows are much closer to the paper's, which is why the harness itself
is not suspect.

Two alternative explanations were tested and rejected: including punctuation tokens
in the GloVe lookup makes results *worse* (−1.6 to −8.5 across four datasets), and
the vocabulary construction is not at fault (STS14 scores 52.00 whether the
vocabulary is built from four datasets or seven).

**When quoting an improvement over GloVe, use the paper's margin as the sanity
check, not ours.**

## Reproducing

```bash
python -m eval.diagnostics        # structural checks + scaling curve
python -m eval.run_sts            # the three baselines, seven datasets
python -m eval.compare_paper      # the table above
```
Training: `notebooks/phase3_train_sbert.ipynb` on a Colab T4.

## The retrieval evaluation is not yet valid for this model — and why

Running the Phase 2 retrieval eval on the trained model produced Recall@10 of 7.56
and Precision@10 of 11.54 — worse than the untrained BERT it started from, and
absurd next to its 72.17 on STS. That number is an **artifact of how the evaluation
set was built**, not a result, and has been removed from `baselines.csv` rather than
reported.

| model | % of its top-10 that was ever judged | of those judged, % relevant |
|---|---|---|
| tfidf | 100.0% | 65.0% |
| glove-avg | 100.0% | 58.8% |
| bert-mean | 100.0% | 43.5% |
| sbert-distilroberta-300k | **13.8%** | **83.3%** |

The pool was built in Phase 2 from TF-IDF, averaged GloVe and mean-pooled BERT, each
contributing its top 10 per query. Every result those three return was therefore
judged. The trained model did not exist then, so **86% of what it retrieves was
never shown to a human and is scored as irrelevant by default.**

Of the 14% that *was* judged, 83.3% were relevant — the highest precision of any
model tested, against TF-IDF's 65.0%.

Inspection confirms it. For the query *"refund never arrived after the order was
cancelled"*, its top results include *"they cancel order and not given any refund"*
and *"trash app they cancelled my order and gave no refund"* — correct, and mostly
unjudged, because they share almost no vocabulary with the query and so were never
surfaced by the three systems that built the pool.

**This is the pooling bias predicted in the Phase 2 plan, arriving far more severely
than expected**, and the reason is the point of the whole project: the trained model
finds reviews that match by *meaning* rather than by *words*, which is exactly the
region of the corpus a lexical pool never covers.

The standard remedy is to re-pool. `python -m eval.build_pool --augment` adds the
trained model's candidates without discarding existing judgements: 224 new
candidates across the 26 labelled queries, about ten minutes of labelling. Until
those are judged, no retrieval number for this model should be quoted.

## Re-scored after re-pooling — and TF-IDF still wins

224 candidates from the trained model were added to the pool and judged (89 of
them relevant, a 40% hit rate). Coverage is now equalised, so the comparison is
finally like-for-like:

| model | top-10 judged | Recall@10 | Precision@10 | MRR |
|---|---|---|---|---|
| **tfidf** | 100.0% | **38.70** | **65.00** | **85.71** |
| glove-avg | 100.0% | 30.41 | 58.85 | 78.22 |
| sbert-distilroberta-300k | 96.2% | 23.46 | 45.77 | 72.25 |
| bert-mean | 100.0% | 22.69 | 43.46 | 74.10 |

The trained model went from 7.56 to 23.46 Recall@10 once its results were actually
judged — but **it still loses to keyword search, and only just beats the untrained
BERT it started from.**

The earlier 83.3% figure was measured on the 14% of its results that happened to
overlap the lexical pool. That was a biased sample: results sharing vocabulary with
the query are more likely relevant, so it flattered the model. With near-full
coverage the honest number is 47.6% precision on judged results.

Note which metric moved. **Precision@10 is unchanged for all three baselines**
(65.00 / 58.85 / 43.46 before and after) because it depends only on which of the
top ten are relevant. **Recall@10 fell for everyone** (TF-IDF 46.01 → 38.70) because
89 more relevant reviews entered the denominator. Precision is the metric to quote
when a test collection is still growing; recall is not comparable across pool
revisions.

## What this means, and why it is the right result to get here

**Scoring 72.17 on STS and 45.77 Precision@10 on app reviews is not a contradiction
— it is the finding.** Generic sentence-similarity ability does not transfer to this
corpus.

The training data is SNLI and MultiNLI: clean, grammatical, mostly image captions
and formal prose. The corpus is Swiggy reviews — three words on average, frequently
misspelled, often romanised Hindi. The queries are well-formed English sentences
while the documents are not, so the model is asked to bridge a register gap it never
saw in training.

`PROJECT_PLAN.md` opens Phase 4 with exactly this sentence: *"Generic STS
performance is not the same as performance on app reviews."* This measurement is the
evidence for that claim, and it establishes the number Phase 4 has to beat —
**45.77 Precision@10, against TF-IDF's 65.00.**

A reproduction that only reported the STS score would have looked better and taught
less.
