"""Reproduce the paper's Table 6 ablation.

    python -m eval.ablation --pairs 100000

Two questions, both from Reimers & Gurevych section 6:

  1. Which pooling strategy?  MEAN 80.78, CLS 79.80, MAX 79.07 in the paper.
  2. What should the classifier see?  The paper's striking result is that
     |u-v| alone (69.78) beats u and v together (66.04), that adding it to
     (u,v) jumps the score to 80.78, and that adding u*v on top *hurts* (80.44).

Nine runs — three poolings and seven concatenations, sharing one configuration.
Each is trained fresh, then measured with the Phase 2 harness on STS-B.

Deliberately run at reduced scale. Nine full-size runs is not affordable on a free
Colab session, and the ablation is about the *ordering* of configurations rather
than their absolute values. The subset size is reported in the output, not hidden.

Safe to re-run: configurations already present in results/ablation.csv are
skipped, so a Colab disconnect costs one run rather than all nine.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from eval.sts_eval import evaluate_sts
from src.embeddings import sbert
from src.training.model import VARIANTS

RESULTS = Path("results/ablation.csv")
MAIN_VARIANT = "u,v,|u-v|"

# The paper's own numbers, for the side-by-side. Table 6, NLI-trained.
PAPER = {"pooling:mean": 80.78, "pooling:max": 79.07, "pooling:cls": 79.80,
         "concat:u,v": 66.04, "concat:|u-v|": 69.78, "concat:u*v": 70.54,
         "concat:|u-v|,u*v": 78.37, "concat:u,v,u*v": 77.44,
         "concat:u,v,|u-v|": 80.78, "concat:u,v,|u-v|,u*v": 80.44}


def configs():
    """Nine runs. mean + (u,v,|u-v|) is the shared reference point of both halves."""
    seen = set()
    for pooling in ("mean", "max", "cls"):
        yield f"pooling:{pooling}", pooling, MAIN_VARIANT
        seen.add((pooling, MAIN_VARIANT))
    for variant in VARIANTS:
        if ("mean", variant) not in seen:
            yield f"concat:{variant}", "mean", variant


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--pairs", type=int, default=100_000)
    ap.add_argument("--out-root", default="models/ablation")
    args = ap.parse_args()

    done = {}
    if RESULTS.exists():
        done = dict(zip(pd.read_csv(RESULTS).config, pd.read_csv(RESULTS).spearman))
        print(f"{len(done)} configurations already done — skipping those\n")

    rows = [{"config": k, "spearman": v} for k, v in done.items()]
    for name, pooling, variant in configs():
        if name in done:
            continue
        out = Path(args.out_root) / name.replace(":", "_").replace("|", "abs").replace("*", "x")
        print(f"\n=== {name}  (pooling={pooling}, variant={variant}) ===")
        cmd = [sys.executable, "-m", "src.training.train", "--model", args.model,
               "--pairs", str(args.pairs), "--pooling", pooling,
               "--variant", variant, "--out", str(out),
               "--checkpoint-every", "2000", "--resume"]
        if subprocess.run(cmd).returncode != 0:
            print(f"  {name} FAILED — continuing with the rest")
            continue

        score = evaluate_sts(sbert.make_encoder(out / "encoder", quiet=True),
                             name, ["STS-B"]).iloc[0].spearman
        rows.append({"config": name, "spearman": score})
        pd.DataFrame(rows).to_csv(RESULTS, index=False)   # save after every run

    df = pd.DataFrame(rows).set_index("config")
    df["paper"] = pd.Series(PAPER)
    df["delta"] = (df.spearman - df.paper).round(2)
    print(f"\n{'=' * 62}\nTable 6 ablation — {args.model}, {args.pairs:,} pairs, STS-B\n")
    print(df.to_string())

    # (u,v,|u-v|) is not its own row -- it is the `pooling:mean` run, the shared
    # reference point of both halves of the table. Read the reference value there.
    ref = ("concat:u,v", "pooling:mean", "concat:u,v,|u-v|,u*v")
    if set(ref) <= set(df.index):
        base, plus, plusprod = (df.loc[k, "spearman"] for k in ref)
        print(f"\nCLAIM 1 — |u-v| is the critical component:")
        print(f"  (u,v) {base:.2f} -> (u,v,|u-v|) {plus:.2f}   {plus - base:+.2f}"
              f"   {'HOLDS' if plus > base else 'DOES NOT HOLD'}"
              f"   (paper: +14.74)")
        print(f"CLAIM 2 — adding u*v hurts:")
        print(f"  (u,v,|u-v|) {plus:.2f} -> +u*v {plusprod:.2f}   {plusprod - plus:+.2f}"
              f"   {'HOLDS' if plusprod < plus else 'DOES NOT HOLD'}"
              f"   (paper: -0.34)")
        print("\nIf a claim does not hold, the likely reasons in order: the paper's own\n"
              "margin is small (-0.34 for claim 2) and may not survive a smaller subset;\n"
              "one seed per configuration; and batch-16 noise.")


if __name__ == "__main__":
    main()
