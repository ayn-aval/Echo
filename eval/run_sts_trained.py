"""STS scores for every trained encoder on disk -> results/sts_trained.csv

    python -m eval.run_sts_trained

eval/run_sts.py covers the three Phase 2 baselines. This is its counterpart for
the models we trained ourselves, so the Phase 3 and Phase 4 numbers in the
write-up come from a script rather than from a notebook cell nobody can re-run.

It calls the same evaluate_sts() the baselines use, on the same machine, so the
Phase 3 vs Phase 4 comparison is not a Colab number set against an MPS one.
"""

import logging
from pathlib import Path

import pandas as pd

from eval.sts_eval import evaluate_sts
from src.embeddings import sbert
from src.utils.device import describe

logging.getLogger("transformers").setLevel(logging.ERROR)

RESULTS = Path("results/sts_trained.csv")


def main() -> None:
    print(f"device: {describe()}\n")
    trained = sbert.available()
    if not trained:
        print("No trained encoder on disk. Expected one of:")
        for name, (path, _) in sbert.TRAINED.items():
            print(f"  {name:28} {path}")
        return

    frames = []
    for name, (path, _) in trained.items():
        print(f"{name}")
        frames.append(evaluate_sts(sbert.make_encoder(path, quiet=True), name))

    df = pd.concat(frames, ignore_index=True)
    df["task"] = "sts"
    RESULTS.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS, index=False)

    wide = df.pivot(index="model", columns="dataset", values="spearman")
    order = [c for c in ["STS12", "STS13", "STS14", "STS15", "STS16", "STS-B",
                         "SICK-R", "Avg"] if c in wide.columns]
    print(f"\n{wide[order].to_string()}")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
