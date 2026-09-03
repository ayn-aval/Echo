"""Verify every headline number in README.md still matches results/.

    python -m eval.check_readme

The README quotes about thirty figures. They were correct when written; nothing
stops a later re-run of an eval script from moving one and leaving the README
quietly wrong, which is worse than having no number at all — a reader has no way
to tell a stale figure from a current one.

So each claim below names where it comes from, reads the value out of the CSV the
relevant eval script wrote, and asserts that exact string appears in README.md.
Add a number to the README, add a row here.
"""

import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _csv(name):
    return pd.read_csv(RESULTS / name)


def _r(value, places=1):
    """Round the way a person writing prose would.

    Python's format() rounds 82.35 to '82.3', because 82.35 is not exactly
    representable in binary and the stored double sits just below it. A reader
    computing 28/34 writes 82.4, so the README says 82.4 and is right. Decimal
    with ROUND_HALF_UP reproduces that.
    """
    q = Decimal(1).scaleb(-places)
    return str(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def claims():
    """(label, value as it must appear in the README, source)."""
    base = _csv("baselines.csv")
    ret = base[base.task == "retrieval"]
    prec = ret[ret.dataset == "precision@10"].set_index("model").score
    rec = ret[ret.dataset == "recall@10"].set_index("model").score
    mrr = ret[ret.dataset == "mrr"].set_index("model").score
    ceiling = ret[ret.dataset == "max_possible_recall@10"].score.iloc[0]

    sts = _csv("sts_trained.csv")
    avg = sts[sts.dataset == "Avg"].set_index("model").spearman

    bench = _csv("search_benchmark.csv")
    acc = bench[bench.measure == "accuracy"].set_index("config")
    lat = bench[bench.measure == "latency"].set_index("stage")

    clus = _csv("clustering_comparison.csv").set_index("model")
    abl = _csv("ablation.csv").set_index("config").spearman
    tbl = _csv("table1_comparison.csv").set_index(_csv("table1_comparison.csv").columns[0])

    out = [
        ("STS avg, NLI-only model", f"{avg['sbert-distilroberta-300k']:.2f}", "sts_trained.csv"),
        ("STS avg, domain-adapted", f"{avg['sbert-domain']:.2f}", "sts_trained.csv"),
        ("Paper's SRoBERTa-NLI-base avg", f"{tbl.loc['SRoBERTa-NLI-base (paper)', 'Avg']:.2f}",
         "table1_comparison.csv"),
        ("Our GloVe avg", f"{tbl.loc['Avg. GloVe (ours)', 'Avg']:.2f}", "table1_comparison.csv"),
        ("Paper's GloVe avg", f"{tbl.loc['Avg. GloVe (paper)', 'Avg']:.2f}", "table1_comparison.csv"),

        ("TF-IDF Precision@10", f"{prec['tfidf']:.2f}", "baselines.csv"),
        ("TF-IDF Recall@10", f"{rec['tfidf']:.2f}", "baselines.csv"),
        ("TF-IDF MRR", f"{mrr['tfidf']:.2f}", "baselines.csv"),
        ("Phase 3 Precision@10", f"{prec['sbert-distilroberta-300k']:.2f}", "baselines.csv"),
        ("Phase 4 Precision@10", f"{prec['sbert-domain']:.2f}", "baselines.csv"),
        ("Recall@10 ceiling", f"{ceiling:.2f}", "baselines.csv"),

        ("Two-stage Precision@10", f"{acc.loc['faiss-top50+cross-encoder', 'precision@10']:.2f}",
         "search_benchmark.csv"),
        ("Two-stage Recall@10", f"{acc.loc['faiss-top50+cross-encoder', 'recall@10']:.2f}",
         "search_benchmark.csv"),
        ("Two-stage MRR", f"{acc.loc['faiss-top50+cross-encoder', 'mrr']:.2f}",
         "search_benchmark.csv"),
        ("IVF Precision@10", f"{acc.loc['faiss-ivf-nprobe10', 'precision@10']:.2f}",
         "search_benchmark.csv"),

        ("Single-stage p50 ms", f"{lat.loc['total_1stage', 'p50_ms']:.2f}", "search_benchmark.csv"),
        ("Two-stage p50 ms", f"{lat.loc['total_2stage', 'p50_ms']:.2f}", "search_benchmark.csv"),
        ("FAISS exact p50 ms", f"{lat.loc['faiss_exact', 'p50_ms']:.2f}", "search_benchmark.csv"),
        ("Cross-encode 50 p50 ms", f"{lat.loc['cross_encode_50', 'p50_ms']:.2f}",
         "search_benchmark.csv"),

        ("Themes found", f"{int(clus.loc['sbert-domain', 'n_clusters'])}",
         "clustering_comparison.csv"),
        ("Largest theme %", f"{clus.loc['sbert-domain', 'largest_pct']:.2f}",
         "clustering_comparison.csv"),
        ("GloVe largest theme %", f"{clus.loc['glove-avg', 'largest_pct']:.2f}",
         "clustering_comparison.csv"),
        ("Audit, domain-adapted", _r(clus.loc['sbert-domain', 'audit_accuracy']),
         "clustering_comparison.csv"),
        ("Audit, GloVe", _r(clus.loc['glove-avg', 'audit_accuracy']),
         "clustering_comparison.csv"),

        # The README quotes the ablation as deltas and as the two non-overlapping
        # ranges, which is the claim being made; the raw scores live in the CSV.
        ("Ablation delta from |u-v|",
         f"+{abl['pooling:mean'] - abl['concat:u,v']:.2f}", "ablation.csv"),
        ("Ablation delta from u*v",
         f"+{abl['concat:u,v,|u-v|,u*v'] - abl['pooling:mean']:.2f}", "ablation.csv"),
        ("Ablation, lowest with |u-v|", f"{abl['concat:|u-v|']:.2f}", "ablation.csv"),
        ("Ablation, highest with |u-v|", f"{abl['concat:u,v,|u-v|,u*v']:.2f}", "ablation.csv"),
        ("Ablation, lowest without", f"{abl['concat:u,v']:.2f}", "ablation.csv"),
        ("Ablation, highest without", f"{abl['concat:u,v,u*v']:.2f}", "ablation.csv"),
    ]

    summary = _csv("dataset_summary.csv").set_index(_csv("dataset_summary.csv").columns[0]).value
    out += [
        ("Reviews collected", f"{int(float(summary['reviews_total'])):,}", "dataset_summary.csv"),
        ("Distinct texts", f"{int(float(summary['distinct_texts_kept'])):,}", "dataset_summary.csv"),
        ("Days covered", f"{int(float(summary['days_covered']))}", "dataset_summary.csv"),
    ]
    return out


def main() -> None:
    readme = (ROOT / "README.md").read_text()
    # Strip the mermaid block: node labels legitimately round figures for display.
    prose = re.sub(r"```mermaid.*?```", "", readme, flags=re.S)

    checks = claims()
    bad = []
    for label, value, source in checks:
        missing = value not in prose
        if missing:
            bad.append((label, value, source))
        print(f"  {'MISS' if missing else 'ok  '} {label:32} {value:>10}   {source}")

    print(f"\n{len(checks) - len(bad)}/{len(checks)} README figures match results/")
    if bad:
        print("\nThese appear in no results file, or the README is stale:")
        for label, value, source in bad:
            print(f"  {label}: expected {value!r} from {source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
