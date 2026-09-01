"""Side-by-side against the Sentence-BERT paper's Table 1.

    python -m eval.compare_paper

Our numbers come from results/baselines.csv (the three Phase 2 baselines, run by
eval/run_sts.py) and results/sts_results_colab.csv (the trained model, from the
Colab run). The paper's numbers are transcribed from Table 1 of arXiv:1908.10084.
"""

from pathlib import Path

import pandas as pd

COLS = ["STS12", "STS13", "STS14", "STS15", "STS16", "STS-B", "SICK-R", "Avg"]

PAPER = {
    "Avg. GloVe":        [55.14, 70.66, 59.73, 68.25, 63.66, 58.02, 53.76, 61.32],
    "Avg. BERT":         [38.78, 57.98, 57.98, 63.15, 61.06, 46.35, 58.40, 54.81],
    "BERT CLS":          [20.16, 30.01, 20.09, 36.88, 38.08, 16.50, 42.63, 29.19],
    "SBERT-NLI-base":    [70.97, 76.53, 73.19, 79.09, 74.30, 77.03, 72.91, 74.89],
    "SRoBERTa-NLI-base": [71.54, 72.49, 70.80, 78.74, 73.69, 77.77, 74.46, 74.21],
}

OURS = {"glove-avg": "Avg. GloVe", "bert-mean": "Avg. BERT", "bert-cls": "BERT CLS"}


def main() -> None:
    base = pd.read_csv("results/baselines.csv")
    base = base[base.get("task", "sts") == "sts"]
    rows = {}
    for key, label in OURS.items():
        s = base[base.model == key].set_index("dataset").spearman
        rows[f"{label} (ours)"] = s.reindex(COLS).values

    colab = Path("results/sts_results_colab.csv")
    if colab.exists():
        s = pd.read_csv(colab).set_index("dataset").spearman
        rows["SBERT distilroberta 300k (ours)"] = s.reindex(COLS).values

    table = pd.DataFrame(rows, index=COLS).T
    paper = pd.DataFrame(PAPER, index=COLS).T
    paper.index = [f"{i} (paper)" for i in paper.index]

    order = ["Avg. GloVe (ours)", "Avg. GloVe (paper)",
             "Avg. BERT (ours)", "Avg. BERT (paper)",
             "BERT CLS (ours)", "BERT CLS (paper)",
             "SBERT distilroberta 300k (ours)",
             "SRoBERTa-NLI-base (paper)", "SBERT-NLI-base (paper)"]
    both = pd.concat([table, paper]).reindex([o for o in order if o in
                                              set(table.index) | set(paper.index)])
    print(both.round(2).to_string())
    both.round(2).to_csv("results/table1_comparison.csv")

    ours = both.loc["SBERT distilroberta 300k (ours)"]
    print("\nOur trained model vs the paper's closest row (SRoBERTa-NLI-base):")
    delta = ours - both.loc["SRoBERTa-NLI-base (paper)"]
    for c in COLS:
        mark = "  <- we win" if delta[c] > 0 else ""
        print(f"  {c:7} {ours[c]:6.2f}  vs {both.loc['SRoBERTa-NLI-base (paper)', c]:6.2f}"
              f"   {delta[c]:+6.2f}{mark}")

    print("\nvs our own untrained baselines, on the same seven datasets:")
    for label in ["Avg. GloVe (ours)", "Avg. BERT (ours)", "BERT CLS (ours)"]:
        print(f"  beats {label:22} by {ours['Avg'] - both.loc[label, 'Avg']:+6.2f}")
    print("\nwrote results/table1_comparison.csv")


if __name__ == "__main__":
    main()
