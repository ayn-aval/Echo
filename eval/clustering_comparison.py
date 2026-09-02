"""Run the identical clustering pipeline on all three embeddings and compare.

    python -m eval.clustering_comparison
    python -m eval.clustering_comparison --persist   # also write themes for all three

This is the comparison that justifies the project: does the model we trained
produce better themes than averaged GloVe or off-the-shelf BERT? Everything
downstream of the vectors is held identical — same UMAP settings, same HDBSCAN
settings, same naming — so the only thing varying is the embedding.

Reading the output honestly:

  noise %        comparable across models, but MUST be read with largest_pct.
                 Lower noise looks like better coverage, and it is exactly what a
                 model achieves by dumping the corpus into one giant bucket.
                 GloVe posts the best noise figure here (15.0%) while holding
                 58.6% of all reviews in a single "theme" that mixes 5-star
                 praise with refund complaints. Quoting noise alone would rank
                 the worst model first.

  largest_pct    share of the corpus in the single biggest cluster. Low is good:
                 it means the pipeline actually separated topics.

  silhouette     NOT strictly comparable across models. It is computed inside
                 each model's own embedding space, and those spaces have
                 different geometry and dimensionality (GloVe is 300d, the others
                 768d). Treat it as a within-model diagnostic. Quoting it as
                 proof that one model beats another would be wrong.

  hand-audit     the measure that actually settles it, because a human reads the
                 review and the theme it was put in and says yes or no.
                 `streamlit run app/audit.py`, then --audit here to score it.

Writes results/clustering_comparison.csv.
"""

import argparse
import itertools
import math
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

from src.clustering.cluster import cluster, load_vectors, reduce, silhouette, summarise
from src.clustering.name_themes import build, persist
from src.db.connection import connection

MODELS = ["glove-avg", "bert-mean", "sbert-domain"]
RESULTS = Path("results/clustering_comparison.csv")


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion, valid at small n.

    Reported because 34 judgements per model is a small sample and the raw
    percentages look more precise than they are.
    """
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round((centre - half) * 100, 1), round((centre + half) * 100, 1)


def audit_detail():
    """Confidence intervals, pairwise significance, and the mega-cluster split.

    The last one matters: a model whose biggest cluster swallows most of the
    corpus has most of its audit sample drawn from that one cluster, so its
    headline accuracy is really a verdict on that cluster.
    """
    with connection() as conn:
        df = pd.read_sql("""
            SELECT a.model, a.theme_id, a.belongs, t.label
              FROM theme_audit a
              JOIN themes t ON t.model = a.model AND t.theme_id = a.theme_id""", conn)
        biggest = pd.read_sql("""
            SELECT model, theme_id AS biggest FROM (
              SELECT model, theme_id, row_number() OVER (
                     PARTITION BY model ORDER BY n_rows DESC) rn
                FROM themes) x WHERE rn = 1""", conn)
    if df.empty:
        return
    g = df.groupby("model").belongs.agg(["sum", "count"])
    print("\naudit accuracy with 95% confidence intervals")
    for model, r in g.iterrows():
        k, n = int(r["sum"]), int(r["count"])
        lo, hi = wilson(k, n)
        print(f"  {model:14} {k}/{n} = {k / n * 100:5.1f}%   95% CI [{lo}, {hi}]")

    print("\npairwise (Fisher exact) — at this sample size not every gap is real")
    for a, b in itertools.combinations(g.index, 2):
        table = [[int(g.loc[a, "sum"]), int(g.loc[a, "count"] - g.loc[a, "sum"])],
                 [int(g.loc[b, "sum"]), int(g.loc[b, "count"] - g.loc[b, "sum"])]]
        pval = fisher_exact(table)[1]
        print(f"  {a:14} vs {b:14} p={pval:.4f}  "
              f"{'SIGNIFICANT' if pval < 0.05 else 'not significant'}")

    d = df.merge(biggest, on="model")
    d["from_biggest"] = d.theme_id == d.biggest
    print("\nshare of each audit sample drawn from that model's biggest cluster")
    for model, grp in d.groupby("model"):
        n_big = int(grp.from_biggest.sum())
        inside = grp[grp.from_biggest].belongs.mean()
        outside = grp[~grp.from_biggest].belongs.mean()
        print(f"  {model:14} {n_big:2}/{len(grp)} rows ({n_big / len(grp) * 100:4.1f}%)"
              f"  accuracy inside "
              f"{'n/a' if pd.isna(inside) else f'{inside * 100:5.1f}%'}"
              f"  outside {outside * 100:5.1f}%")


def audit_scores():
    """Per-model accuracy from the blind hand-audit, if any has been done."""
    with connection() as conn:
        df = pd.read_sql("SELECT model, belongs FROM theme_audit", conn)
    if df.empty:
        return {}
    g = df.groupby("model").belongs.agg(["mean", "count"])
    return {m: (round(r["mean"] * 100, 2), int(r["count"])) for m, r in g.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-cluster-size", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--persist", action="store_true",
                    help="also write each model's themes to Postgres, which the "
                         "hand-audit needs")
    args = ap.parse_args()

    audits = audit_scores()
    rows = []
    for model in MODELS:
        print(f"\n{model}")
        vectors = load_vectors(model)
        labels, _ = cluster(reduce(model), args.min_cluster_size, args.min_samples)
        stats = summarise(labels)
        # Share of actual review rows, not distinct texts — the number a person
        # asking "how much of my data got a theme" means.
        acc, n = audits.get(model, (None, 0))
        rows.append({"model": model, "dim": vectors.shape[1], **stats,
                     "silhouette": silhouette(vectors, labels),
                     "audit_accuracy": acc, "audit_n": n})
        print(f"  clusters={stats['n_clusters']}  noise={stats['noise_pct']}%  "
              f"largest={stats['largest']}")

        if args.persist:
            themes, assign, _ = build(model, args.min_cluster_size, args.min_samples)
            persist(themes, assign, model)
            print(f"  persisted {len(themes)} themes")

    df = pd.DataFrame(rows)
    RESULTS.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nwrote {RESULTS}")
    if audits:
        audit_detail()
    if not audits:
        print("\naudit_accuracy is empty — run `streamlit run app/audit.py`, then "
              "re-run this to fill it in. Silhouette alone does not settle which "
              "model is better; see this module's docstring for why.")


if __name__ == "__main__":
    main()
