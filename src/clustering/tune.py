"""Sweep HDBSCAN parameters and show what each setting actually produces.

    python -m src.clustering.tune --model sbert-domain
    python -m src.clustering.tune --model sbert-domain --show 20

Tuning by silhouette alone is how you end up with a mathematically tidy
clustering full of themes nobody would name. So every setting prints its numbers
*and* its biggest clusters as real terms and real reviews, and the choice is made
by reading them.

    min_cluster_size  the smallest group allowed to count as a theme. Raise it
                      and you get fewer, broader themes and more noise; lower it
                      and you get many narrow ones, including some that are just
                      one phrase repeated.

    min_samples       how conservative HDBSCAN is about density. It sets how many
                      neighbours a point needs before its neighbourhood counts as
                      dense. Raise it and clusters shrink to their firm cores
                      while everything on the fringe becomes noise; lower it and
                      clusters swell and start bleeding into each other.

They interact: min_cluster_size decides which candidate groups survive, and
min_samples decides how much of the surrounding fringe those groups keep. The
default min_samples = min_cluster_size is aggressive about noise, which is why
the sweep tries lower values too.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.clustering.cluster import (cluster, load_vectors, reduce, silhouette,
                                    summarise)

CORPUS = Path("data/vectors/corpus.parquet")
GRID_MCS = [30, 60, 120, 250]
GRID_MS = [5, 15, None]     # None -> HDBSCAN uses min_cluster_size


def top_terms(texts, labels, cluster_id, n=6):
    """Terms distinctive to this cluster, against the rest of the corpus."""
    inside = [t for t, l in zip(texts, labels) if l == cluster_id]
    if not inside:
        return ""
    tf = TfidfVectorizer(max_features=4000, stop_words="english", min_df=2)
    try:
        tf.fit(texts)
        scores = np.asarray(tf.transform([" ".join(inside)]).todense()).ravel()
    except ValueError:
        return ""
    names = tf.get_feature_names_out()
    return ", ".join(names[i] for i in np.argsort(-scores)[:n])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="sbert-domain")
    ap.add_argument("--show", type=int, default=4,
                    help="how many of the biggest clusters to print per setting")
    ap.add_argument("--only", nargs=2, type=int, metavar=("MCS", "MS"),
                    help="inspect one setting in depth instead of sweeping")
    args = ap.parse_args()

    texts = pd.read_parquet(CORPUS).content.tolist()
    vectors = load_vectors(args.model)
    reduced = reduce(args.model)
    print(f"\n{len(texts):,} texts · {args.model}\n")

    settings = ([tuple(args.only)] if args.only
                else [(m, s) for m in GRID_MCS for s in GRID_MS])

    rows, computed = [], {}
    for mcs, ms in settings:
        labels, _ = cluster(reduced, mcs, ms)
        computed[(mcs, ms)] = labels          # reuse below; HDBSCAN is not free
        stats = summarise(labels)
        stats.update(model=args.model, min_cluster_size=mcs,
                     min_samples=ms, silhouette=silhouette(vectors, labels))
        rows.append(stats)
        print(f"  mcs={mcs:<4} ms={str(ms):<5} clusters={stats['n_clusters']:<5} "
              f"noise={stats['noise_pct']:>5.1f}%  largest={stats['largest']:<6} "
              f"median={stats['median_size']:<5} silhouette={stats['silhouette']}")

    df = pd.DataFrame(rows)[["min_cluster_size", "min_samples", "n_clusters",
                             "noise_pct", "largest", "median_size", "silhouette"]]
    print(f"\n{df.to_string(index=False)}")

    # The part that matters: what the clusters actually contain.
    for mcs, ms in settings:
        labels = computed[(mcs, ms)]
        order = pd.Series(labels[labels >= 0]).value_counts().index[:args.show]
        print(f"\n{'=' * 78}\nmcs={mcs} ms={ms} — {len(order)} biggest clusters shown")
        for cid in order:
            members = [t for t, l in zip(texts, labels) if l == cid]
            print(f"\n  cluster {cid}  ({len(members):,} texts)")
            print(f"    terms: {top_terms(texts, labels, cid)}")
            for t in members[:3]:
                print(f"      · {' '.join(str(t).split())[:82]}")


if __name__ == "__main__":
    main()
