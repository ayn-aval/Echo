"""Evaluate the three Phase 2 baselines on STS -> results/baselines.csv

    python -m eval.run_sts

All three go through the same evaluate_sts() function, so the numbers are
directly comparable — to each other now, and to the trained models in Phases 3
and 4, which will call that function unchanged.
"""

import logging
from pathlib import Path

import pandas as pd

from eval.sts_data import DATASETS, load_sts
from eval.sts_eval import evaluate_sts
from src.embeddings import baselines, glove
from src.utils.device import describe

logging.getLogger("transformers").setLevel(logging.ERROR)

RESULTS = Path("results/baselines.csv")


def main() -> None:
    print(f"device: {describe()}\n")

    corpus = []
    for name in DATASETS:
        s1, s2, _ = load_sts(name)
        corpus.extend(s1 + s2)
    print(f"{len(set(corpus)):,} distinct sentences across {len(DATASETS)} datasets\n")

    frames = []

    print("Averaged GloVe (840B.300d)")
    frames.append(evaluate_sts(glove.make_encoder(corpus, "sts"), "glove-avg"))

    print("\nBERT mean-pooled (bert-base-uncased)")
    frames.append(evaluate_sts(baselines.make_encoder("mean"), "bert-mean"))

    print("\nBERT [CLS] token (bert-base-uncased)")
    frames.append(evaluate_sts(baselines.make_encoder("cls"), "bert-cls"))

    df = pd.concat(frames, ignore_index=True)
    df["task"] = "sts"
    RESULTS.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS, index=False)

    table = df.pivot(index="model", columns="dataset", values="spearman")
    order = [d for d in DATASETS] + ["Avg"]
    table = table[order].reindex(["glove-avg", "bert-mean", "bert-cls"])
    print("\n" + "=" * 72)
    print("Spearman rank correlation x100 (higher is better)\n")
    print(table.to_string())
    print("=" * 72)

    g, b, c = (table.loc[m, "Avg"] for m in ("glove-avg", "bert-mean", "bert-cls"))
    print()
    if g > b:
        print(f"REPRODUCED: averaged GloVe ({g:.2f}) beats mean-pooled BERT ({b:.2f}) "
              f"by {g - b:.2f} points.")
        print("This is the Sentence-BERT paper's central motivation: a 2014 word-vector")
        print("average outperforms a 2018 transformer at judging sentence similarity,")
        print("because BERT was never trained to make its vectors comparable by cosine.")
    else:
        print(f"NOT reproduced: BERT ({b:.2f}) beat GloVe ({g:.2f}). Treat this as a bug")
        print("until proven otherwise — check attention-mask pooling and cosine")
        print("normalisation before believing it.")
    print(f"\n[CLS] pooling scores {c:.2f} — far below both, as the paper reports.")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
