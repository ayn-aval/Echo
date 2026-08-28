"""SNLI + MultiNLI, prepared for siamese training.

Natural Language Inference gives us sentence pairs labelled with how they relate:
entailment (the second follows from the first), contradiction (they conflict), or
neutral. The Sentence-BERT paper's insight is that this is enough supervision to
teach an encoder that *meaning* similarity should show up as *vector* closeness —
we never train on similarity scores directly.

    python -m src.training.nli_data --pairs 6000
"""

import argparse
from collections import Counter

from datasets import concatenate_datasets, load_dataset

LABEL_NAMES = ["entailment", "neutral", "contradiction"]
CACHE = "data/nli"
COLUMNS = ["premise", "hypothesis", "label"]


def _prepare(repo: str, split: str, source: str):
    ds = load_dataset(repo, split=split, cache_dir=CACHE)
    ds = ds.select_columns(COLUMNS)

    # SNLI and MultiNLI both ship rows where the five annotators reached no
    # majority. They are marked -1, which is NOT a fourth class — feeding them to
    # a 3-way softmax corrupts training silently, since -1 indexes the last class
    # in PyTorch. Dropping them is the single most important line in this file.
    before = len(ds)
    ds = ds.filter(lambda r: r["label"] in (0, 1, 2), num_proc=1)
    dropped = before - len(ds)

    ds = ds.add_column("source", [source] * len(ds))
    print(f"  {source:9} {len(ds):7,} usable   ({dropped:,} dropped with label -1)")
    return ds


def load_nli(n_pairs=None, seed=42):
    """Balanced subset of SNLI + MultiNLI, as a datasets.Dataset."""
    combined = concatenate_datasets([
        _prepare("stanfordnlp/snli", "train", "snli"),
        _prepare("nyu-mll/multi_nli", "train", "mnli"),
    ]).shuffle(seed=seed)

    if n_pairs is None or n_pairs >= len(combined):
        return combined

    # Stratify: equal numbers of each label, so the loss starts at a predictable
    # ln(3) and batch accuracy is readable against a 33% floor.
    per_label = n_pairs // 3
    keep = []
    for label in range(3):
        idx = [i for i, l in enumerate(combined["label"]) if l == label][:per_label]
        keep.extend(idx)
    return combined.select(sorted(keep)).shuffle(seed=seed)


def describe(ds) -> None:
    labels = Counter(ds["label"])
    sources = Counter(ds["source"])
    print(f"\n  {len(ds):,} pairs")
    for i, name in enumerate(LABEL_NAMES):
        print(f"    {name:14} {labels[i]:7,}  ({labels[i] / len(ds):.1%})")
    for name, count in sources.items():
        print(f"    from {name:9} {count:7,}  ({count / len(ds):.1%})")
    print(f"    label values present: {sorted(labels)}  (must be [0, 1, 2])")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=None)
    args = ap.parse_args()

    ds = load_nli(args.pairs)
    describe(ds)
    print("\n  sample pairs:")
    for row in ds.select(range(3)):
        print(f"    [{LABEL_NAMES[row['label']]:13}] {row['premise'][:58]}")
        print(f"    {'':16} -> {row['hypothesis'][:58]}")


if __name__ == "__main__":
    main()
