# Phase 4 — domain adaptation

Continued fine-tuning the Phase 3 encoder on Swiggy review pairs with
`MultipleNegativesRankingLoss`. `sentence-transformers` is permitted from this
phase onward; Phase 3's raw-PyTorch rule does not extend forward.

**Headline: review-retrieval Precision@10 rose 45.77 → 61.15, and generic STS
rose 72.17 → 74.54 at the same time.** The phase target was to beat 45.77. The
gap to TF-IDF narrowed from 19.23 points to 3.85, but TF-IDF still wins.

---

## The project plan's premise was wrong, and had to be measured

`PROJECT_PLAN.md` opens Phase 4 assuming Swiggy's replies are "templated by
complaint category", which would make two reviews sharing a reply a free positive
pair. They are not. They are templated by **star rating**, and carry nothing else.

Reproducible with `python -m eval.reply_signal` -> `results/phase4_reply_signal.csv`:

| test | result |
|---|---|
| replies naming a specific complaint topic | 0.7% |
| frequent replies (50+ uses) that always go to one rating | 73 of 91 |
| median within-reply rating std | 0.00 |
| 1-star: predict template from review text | 8.12% vs 8.63% majority (**−0.51**) |
| 5-star: predict template from review text | 11.18% vs 10.07% majority (+1.11) |

The third row is decisive. Hold the star rating fixed and a TF-IDF classifier
still cannot guess which of 23 one-star templates was sent. The templates are
rotated at random within a rating band, so "two reviews with the same reply"
means precisely "two reviews with the same star rating" — a signal already
present in a column.

Real pairs the rejected strategies would have produced:

```
shared reply (2,353 reviews): "Hello, we would like to know more about it…"
   A [1★] Payments through UPI ID are discontinued
   B [1★] late too long to deliver
```

A UPI outage paired with a late delivery. Training on that teaches the model
that all complaints mean the same thing, which is the opposite of what theme
discovery needs.

## What was used instead

`python -m src.training.mine_pairs` -> `data/phase4_pairs.jsonl`, 53,061 pairs.

**mined — 7,197 pairs.** Two different reviews that TF-IDF *and* the Phase 3
encoder both rank in each other's top-10. Two systems with unrelated failure
modes agreeing is evidence; the Phase 3 encoder agreeing with itself is an echo,
so the TF-IDF constraint is what makes the set worth anything. 8,111 trivial
case-and-emoji variants were discarded. Examples:

```
sbert 0.90 · tfidf 0.32   A  they took extra charge for express delivery but then they delay the time
                          B  I will pay extra charges for express delivery but the order was too late
sbert 0.83 · tfidf 0.62   A  thanks discount dene ke liye 🥲
                          B  thank you offer Dene ke liye
```

**Precision is roughly 80%, not clean.** The failure mode is a shared syntactic
frame — `"not giving discount"` paired with `"not giving cod option"`. It could
not be filtered without cutting genuine pairs. MNRL tolerates noisy positives,
but 80% is the honest figure.

Dropping the TF-IDF constraint yields ~85,000 pairs and reintroduces exactly the
failure of the reply-based strategies: `"app is down during protest"` paired with
`"service are pathetic now days"` — both negative, different themes.

**simcse — 45,864 self-pairs.** Each review with itself; dropout supplies two
different views. No false positive is possible by construction.

**Leakage check.** 38 of 7,197 mined pairs (0.53%) join two reviews judged
relevant to the same query. No judgement is read while mining. Separately, the
overlap between the 28,663 distinct STS sentences and the 44,978 distinct
training texts is **zero**.

## Training

`src/training/train_domain.py`, one epoch, batch 64, Adam 2e-5, 10% warmup,
mean pooling carried forward from Phase 3. 829 steps, ~20 minutes on a Kaggle T4.

Verified end-to-end on MPS before spending a GPU session, which caught three
things that would have failed on Kaggle: `sentence-transformers` 6 routes `fit()`
through the HuggingFace Trainer and needs `accelerate`; the `losses`/`models`
import paths are deprecated; and the Phase 3 anti-`sentence-transformers` guard
had to be confirmed inert under the notebook's import order.

Verified after training rather than assumed:

- 102 of 103 tensors changed, with the largest movement in layer 5 — the
  expected fine-tuning pattern.
- Kaggle's transformers writes LayerNorm as `gamma`/`beta` where the local
  version uses `weight`/`bias`. Had the remapping not happened, those layers
  would have loaded randomly initialised and the encoder would have been
  silently broken while still producing plausible vectors. Checked against the
  raw safetensors file.
- The encoder loads through `src/embeddings/sbert.py` unchanged, so there is no
  separate code path between training and evaluation.

## Review retrieval

`python -m eval.run_retrieval`. 26 queries, 1,054 judgements, 546 relevant.
Re-pooled first with `python -m eval.build_pool --augment`, then 119 new
candidates were judged — mandatory, per the Phase 3 lesson.

| model | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| **tfidf** | **34.88** | **65.00** | **85.71** |
| glove-avg | 26.90 | 58.85 | 78.22 |
| bert-mean | 19.83 | 43.46 | 74.10 |
| sbert-distilroberta-300k (Phase 3) | 20.57 | 45.77 | 72.25 |
| **sbert-domain (Phase 4)** | **27.86** | **61.15** | **83.81** |

Recall@10 has a ceiling of 56.28 on this collection — with a median of 16
relevant reviews per query you cannot fit them all into a top-10.

**Two checks that make the table trustworthy.**

1. Every previously-measured model returned *identical* Precision@10 and MRR to
   Phase 3 — 65.00 / 58.85 / 43.46 / 45.77 and 85.71 / 78.22 / 74.10 / 72.25, to
   the decimal. Only Recall moved, and only because 67 more relevant reviews
   entered the denominator. Re-pooling did not disturb the baselines.
2. Pooling coverage is symmetric. Share of each model's top-10 that has actually
   been judged: tfidf, glove and bert 100%; **both** trained models 96.15%. The
   two trained models are equally covered, so their comparison is fair, and both
   are slightly understated against TF-IDF rather than flattered. The shortfall
   is 14 unjudged candidates on one query, *"my Instamart order had a problem"*,
   10 of which have been outstanding since Phase 3.

## Generic STS did not degrade — it improved

`python -m eval.run_sts_trained` -> `results/sts_trained.csv`. Both models scored
on the same machine with the same harness, so this is not a Kaggle number set
against a local one.

| dataset | Phase 3 | Phase 4 | Δ |
|---|---|---|---|
| STS12 | 67.20 | 64.92 | −2.28 |
| STS13 | 73.71 | 79.09 | **+5.38** |
| STS14 | 69.11 | 71.48 | +2.37 |
| STS15 | 75.58 | 78.41 | +2.83 |
| STS16 | 72.35 | 75.79 | +3.44 |
| STS-B | 75.60 | 77.65 | +2.05 |
| SICK-R | 71.63 | 74.46 | +2.83 |
| **Avg** | **72.17** | **74.54** | **+2.37** |

This was predicted to fall and did not. Six of seven datasets improved. The
explanation is that contrastive training improves sentence embeddings largely
independently of the corpus it runs on: it spreads out the vector space, which
is the same anisotropy problem that made raw `bert-mean` score 52.64 in Phase 2.
Training on Swiggy reviews fixed a defect that had nothing to do with Swiggy.

Phase 3 reproduced its Colab numbers exactly, to two decimals on all seven
datasets, so the +2.37 is not an artifact of comparing across machines.

**Framing this honestly.** 74.54 exceeds the paper's 74.21 for SRoBERTa-NLI-base.
That is **not** "beating the paper": their number comes from NLI training alone,
ours adds a second stage they never ran. It is a different recipe, not a better
result on the same one. What can be said is that Phase 3's 2.04-point shortfall,
incurred on 32% of the paper's training data, was closed by domain adaptation.

## Limitations

1. **TF-IDF still wins**, 65.00 to 61.15 on Precision@10. The honest headline is
   that domain adaptation closed 80% of the gap while also improving generic STS,
   not that the trained model is the best retriever on this corpus.
2. **Part of the retrieval gain may be circular.** The mined pairs required
   TF-IDF to agree, so the model may have partly learned to imitate TF-IDF —
   which is the system that wins this benchmark. Its score profile moving toward
   TF-IDF's is consistent with that. Phase 4b is designed to settle it.
3. **The two pair sources are not yet separated.** The combined run cannot say
   whether the mined pairs or SimCSE produced the gain, and they have very
   different standing: SimCSE cannot encode any domain relation, so any gain it
   produces is corpus-independent.
4. **Hinglish is still not bridged.** `"khana thanda tha"` against `"the food was
   cold"` scores 0.066, versus 0.049 for a genuinely unrelated pair — barely
   distinguishable. Phase 3 was no better (0.286 vs 0.307). This was a
   hoped-for gain and it is not there.
5. **80% mined-pair precision** is a real noise floor on the training signal.
6. 24 of the 50 evaluation queries remain unlabelled, and 14 pooled candidates on
   one query are unjudged.

## Phase 4b — the ablation, set up but not yet run

`notebooks/phase4b_ablation_kaggle.ipynb`. Two runs, identical to the Phase 4
configuration except for `--sources`, so any difference is attributable to the
pair source rather than to a changed hyperparameter:

```bash
python -m src.training.train_domain --sources mined   ...   # 7,197 pairs, 112 steps
python -m src.training.train_domain --sources simcse  ...   # 45,864 pairs, 716 steps
```

STS is answered immediately, since it needs no labelled pool. **Retrieval is not
free**: each new model must be re-pooled with `eval/build_pool.py --augment` and
its new candidates judged before its number means anything. That is one more
labelling round covering both variants.
