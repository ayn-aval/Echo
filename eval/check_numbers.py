"""Trace every fixed number on the results board back to its source file.

    python -m eval.check_numbers

app/results.py holds the board's copy as literals, which is what makes the page
fast and keeps the wording in one place. The risk that buys is drift: a figure
rerun, a CSV rewritten, and the page keeps quoting last week's number with
nobody the wiser. This asserts each literal still equals what its source says.

Figures 04, 05 and 07 are not here — they query the database at run time and
cannot drift. Everything else is checked.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import results as R  # noqa: E402

RESULTS = Path("results")
FAILURES = []


def check(label: str, got, want, tol: float = 0.005) -> None:
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52} board={got!r} source={want!r}")
    if not ok:
        FAILURES.append(label)


def main() -> None:
    table = pd.read_csv(RESULTS / "table1_comparison.csv", index_col=0)
    ours = table.loc["SBERT distilroberta 300k (ours)"]
    paper = table.loc["SRoBERTa-NLI-base (paper)"]
    columns = ["STS12", "STS13", "STS14", "STS15", "STS16", "STS-B", "SICK-R"]

    print("Figure 01 · the rebuild against the paper")
    for (label, board_paper, board_ours), col in zip(R.STS, columns):
        check(f"{label} — this project", board_ours, float(ours[col]))
        check(f"{label} — the paper", board_paper, float(paper[col]))
    check("headline · rebuild average", float(R.HEADLINE[0]["value"]),
          round(float(ours["Avg"]), 1), tol=0.051)
    check("headline · paper average in the note", "74.2",
          f"{float(paper['Avg']):.1f}")
    check("footer stat · average across the seven", R.STS_FOOT[0][1],
          f"{float(ours['Avg']):.1f}")

    print("\nFigure 01 · after tuning on Swiggy reviews")
    sts = pd.read_csv(RESULTS / "sts_trained.csv")
    domain_avg = float(sts[(sts.model == "sbert-domain") & (sts.dataset == "Avg")].spearman)
    check("footer stat · tuned average", R.STS_FOOT[1][1], f"{domain_avg:.1f}")

    print("\nFigure 02 · search quality")
    base = pd.read_csv(RESULTS / "baselines.csv")
    p10 = base[(base.dataset == "precision@10") & (base.task == "retrieval")]
    p10 = dict(zip(p10.model, p10.score))
    bench = pd.read_csv(RESULTS / "search_benchmark.csv")
    two_stage = float(bench.loc[bench.config == "faiss-top50+cross-encoder",
                                "precision@10"].iloc[0])
    sources = {"Keyword matching": p10["tfidf"],
               "Meaning search, first version": p10["sbert-distilroberta-300k"],
               "Meaning search, tuned on Swiggy": p10["sbert-domain"],
               "Tuned, then re-ordered": two_stage}
    for name, score, _ in R.RETRIEVAL:
        check(name, score, round(sources[name] / 10, 1), tol=0.051)
    check("headline · relevant results", float(R.HEADLINE[1]["value"]),
          round(two_stage / 10, 1), tol=0.051)
    check("headline · keyword search in the note", "6.5",
          f"{p10['tfidf'] / 10:.1f}")
    check("Figure 02 stat · lead over keyword search", R.RETRIEVAL_STATS[0][1],
          f"{(two_stage - p10['tfidf']) / 10:.1f}")

    latency = bench[(bench.measure == "latency") & (bench.stage == "total_2stage")]
    check("Figure 02 stat · time to answer", R.RETRIEVAL_STATS[1][1],
          f"{float(latency.p50_ms.iloc[0]):.0f} ms")

    print("\nFigure 03 · same meaning, different words")
    coh = pd.read_csv(RESULTS / "coherence_example.csv")
    check("no shared topic words", int(coh.max_shared_topic_words.max()), 0)
    for (_, text, match), row in zip(R.COHERENCE, coh.itertuples()):
        check(f"review {row.position} text", text.replace("’", "'"),
              row.content.replace("’", "'"))
        if match:
            check(f"review {row.position} closeness", match,
                  f"{row.similarity_to_first * 100:.0f}%")
    check("topic name", R.COHERENCE_TOPIC["name"], coh.theme.iloc[0])

    print("\nHeadline · topics found")
    clusters = pd.read_csv(RESULTS / "clustering_comparison.csv")
    domain = clusters[clusters.model == "sbert-domain"].iloc[0]
    check("topics found", int(R.HEADLINE[2]["value"]), int(domain.n_clusters))
    check("audit · correct out of 34 in the note", "28",
          str(round(domain.audit_accuracy / 100 * domain.audit_n)))
    check("Figure 06 · reviews checked per model", R.FAILURES[3][0],
          str(int(domain.audit_n)))
    check("Figure 06 · reviews left ungrouped", R.FAILURES[0][0],
          f"{domain.noise_pct:.0f}%")

    print(f"\n{'ALL NUMBERS TRACE TO SOURCE' if not FAILURES else f'{len(FAILURES)} MISMATCH(ES): {FAILURES}'}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
