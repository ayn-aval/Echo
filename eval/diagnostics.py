"""Checks that distinguish an implementation bug from a data-volume limitation.

    python -m eval.diagnostics

Written because "my numbers are below the paper's" has two very different causes
and only one of them is worth apologising for. Every check here either passes or
tells you which of the two you are looking at.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.training.model import VARIANTS, SiameseNLI, pool

logging.getLogger("transformers").setLevel(logging.ERROR)
RESULTS = Path("results")
DEBUG_MODEL = "nreimers/MiniLM-L6-H384-uncased"


def check_weight_sharing(model):
    """Two encoders instead of one would be completely silent at runtime."""
    enc = sum(p.numel() for p in model.encoder.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    total = sum(p.numel() for p in model.parameters())
    return total == enc + head, f"{total:,} == {enc:,} + {head:,}"


def check_pooling_masks():
    """Padding must not affect the pooled vector. Getting this wrong never raises."""
    h = torch.randn(1, 2, 8)
    junk = torch.cat([h, torch.randn(1, 2, 8) * 50], dim=1)
    mask = torch.tensor([[1, 1, 0, 0]])
    ok = []
    for how in ("mean", "max"):
        a = pool(h, torch.ones(1, 2), how)
        b = pool(junk, mask, how)
        ok.append(torch.allclose(a, b, atol=1e-6))
    return all(ok), f"mean={ok[0]} max={ok[1]}"


def check_variant_widths(model):
    u, v = torch.randn(4, 8), torch.randn(4, 8)
    widths = {}
    for name, parts in VARIANTS.items():
        model.parts = parts
        widths[name] = model.combine(u, v).shape[-1] // 8
    expected = {n: len(p) for n, p in VARIANTS.items()}
    return widths == expected, f"{len(widths)} variants, all widths correct"


def check_labels():
    """A surviving -1 label silently becomes class 2 in PyTorch indexing."""
    from src.training.nli_data import load_nli
    ds = load_nli(3000)
    labels = set(ds["label"])
    balance = pd.Series(ds["label"]).value_counts(normalize=True)
    ok = labels == {0, 1, 2} and balance.max() < 0.4
    return ok, f"labels {sorted(labels)}, max class share {balance.max():.1%}"


def main() -> None:
    print("Structural checks — these fail loudly if the architecture is wrong\n")
    model = SiameseNLI(DEBUG_MODEL)
    for name, fn in [("weight sharing (one encoder, not two)", lambda: check_weight_sharing(model)),
                     ("pooling ignores padding", check_pooling_masks),
                     ("concatenation variant widths", lambda: check_variant_widths(model)),
                     ("NLI labels are 0/1/2 and balanced", check_labels)]:
        ok, detail = fn()
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:42} {detail}")

    curve = RESULTS / "phase3_debug_scaling.csv"
    if curve.exists():
        print(f"\nScaling curve ({curve}) — the decisive bug-vs-data evidence\n")
        df = pd.read_csv(curve)
        for _, r in df.iterrows():
            bar = "#" * int(r.sts_avg / 2)
            print(f"  {r.run:22} {r.sts_avg:6.2f}  {bar}")
        base = df[df.pairs == 0].sts_avg.iloc[0]
        trained = df[df.pairs > 0]
        print(f"\n  untrained control: {base:.2f}")
        print(f"  best trained:      {trained.sts_avg.max():.2f} "
              f"({trained.loc[trained.sts_avg.idxmax(), 'run']})")
        print("\n  A curve still climbing with more data means the subset size is the\n"
              "  honest explanation. A flat curve well below the paper would mean\n"
              "  something is broken regardless of scale.")


if __name__ == "__main__":
    main()
