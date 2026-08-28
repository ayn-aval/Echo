"""Loading the STS benchmark datasets.

STS (Semantic Textual Similarity) is a set of sentence pairs that human annotators
scored 0-5 for how close in meaning they are. It is the standard way to measure a
sentence embedding: encode both sentences, take the cosine similarity, and see how
well that ordering matches the humans' ordering.

We use the test splits of STS12-16 and STS-B, the same seven-dataset set the
Sentence-BERT paper reports in its Table 1.
"""

from functools import lru_cache

from datasets import load_dataset

# mteb's mirrors of the original SemEval datasets, already cleaned and aligned.
DATASETS = {
    "STS12": ("mteb/sts12-sts", "test"),
    "STS13": ("mteb/sts13-sts", "test"),
    "STS14": ("mteb/sts14-sts", "test"),
    "STS15": ("mteb/sts15-sts", "test"),
    "STS16": ("mteb/sts16-sts", "test"),
    "STS-B": ("mteb/stsbenchmark-sts", "test"),
}


@lru_cache(maxsize=None)
def load_sts(name: str):
    """Return (sentences1, sentences2, gold_scores) for one dataset."""
    repo, split = DATASETS[name]
    ds = load_dataset(repo, split=split, cache_dir="data/sts")
    return (list(ds["sentence1"]), list(ds["sentence2"]),
            [float(s) for s in ds["score"]])


def load_all():
    return {name: load_sts(name) for name in DATASETS}


if __name__ == "__main__":
    total = 0
    for name in DATASETS:
        s1, s2, gold = load_sts(name)
        total += len(s1)
        print(f"  {name:6} {len(s1):5,} pairs   gold {min(gold):.1f}-{max(gold):.1f}"
              f"   e.g. {s1[0][:44]!r} / {s2[0][:44]!r} = {gold[0]}")
    print(f"  {'total':6} {total:5,} pairs")
