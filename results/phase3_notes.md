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
