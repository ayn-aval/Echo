"""The evaluation function every phase of this project reuses.

It takes one thing: a callable that turns a list of strings into a matrix of
vectors. It knows nothing about BERT, GloVe, PyTorch or sentence-transformers.
That is deliberate — Phases 3, 4 and 5 call this exact function unchanged, so
their numbers are comparable to these baselines rather than merely similar-looking.

Spearman rank correlation is the metric: it asks whether the model *orders* the
pairs the way humans did, not whether it reproduces their exact numbers. A model
that scores every pair 0.2 too high still gets a perfect Spearman.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eval.sts_data import DATASETS, load_sts


def cosine_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. Normalise first, or long sentences win by length."""
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return (a * b).sum(axis=1)


def evaluate_sts(encode, name: str = "model", datasets=None) -> pd.DataFrame:
    """Score an encoder on the STS datasets.

    encode: Callable[[list[str]], np.ndarray] returning shape (n_sentences, dim).
    Returns a DataFrame of model, dataset, n_pairs, spearman (x100, as the paper
    reports it), with an 'Avg' row appended.
    """
    rows = []
    for ds in datasets or list(DATASETS):
        s1, s2, gold = load_sts(ds)

        # STS reuses the same sentences across many pairs, so encode each
        # distinct string once and look it up. Free speed, identical results.
        distinct = list(dict.fromkeys(s1 + s2))
        vectors = np.asarray(encode(distinct), dtype=np.float32)
        if vectors.shape[0] != len(distinct):
            raise ValueError(
                f"encode() returned {vectors.shape[0]} vectors for "
                f"{len(distinct)} sentences")
        where = {s: i for i, s in enumerate(distinct)}

        sims = cosine_pairs(vectors[[where[s] for s in s1]],
                            vectors[[where[s] for s in s2]])
        rho = spearmanr(sims, gold).statistic
        rows.append({"model": name, "dataset": ds, "n_pairs": len(gold),
                     "spearman": round(float(rho) * 100, 2)})
        print(f"    {ds:6} {rho * 100:6.2f}")

    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"model": name, "dataset": "Avg",
                       "n_pairs": int(df.n_pairs.sum()),
                       "spearman": round(df.spearman.mean(), 2)}
    return df
