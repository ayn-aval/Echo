"""Every fixed number and string on the results board, with its source.

Figures 04, 05 and 07 read the database live and so are not here. Everything
else is: the numbers below come from `results/` and change only when the script
named beside them is rerun.

Two rules govern the wording. Nothing is illustrative — every figure is
reproducible from `eval/`. And nothing is written in notation: no correlation
coefficients, no significance levels, no metric names. "Precision@10 of 0.7577"
is a fact about a spreadsheet; "about 8 of every 10 results are relevant" is the
same fact in a form a reader can use.
"""

# ── top bar ───────────────────────────────────────────────────────────────────
STANDFIRST = "Built from scratch in PyTorch, then run over 100,000 Swiggy app reviews"

# ── headline ──────────────────────────────────────────────────────────────────
# 1: results/table1_comparison.csv, our SBERT row against SRoBERTa-NLI-base, the
#    paper's model built on the same base. 2: results/search_benchmark.csv and
#    results/baselines.csv. 3: results/clustering_comparison.csv, sbert-domain.
HEADLINE = [
    dict(label="Rebuilt from the research paper", value="72.2", ghost=True,
         chip="paper 74.2",
         note="Scored on seven public tests of sentence meaning. The published "
              "model was trained on <b>three times</b> as much data."),
    dict(label="Relevant search results", value="7.6", ghost=False,
         chip="+1.1",
         note="Out of every 10 reviews a search returns. Ordinary keyword "
              "search returns <b>6.5</b>."),
    dict(label="Topics found on its own", value="110", ghost=True,
         chip="no list given",
         note="Nobody supplied a list. <b>57</b> are complaints, and 28 of 34 "
              "checked reviews sat in the right topic."),
]

# ── Figure 01 · rebuild against the paper ─────────────────────────────────────
# results/table1_comparison.csv. Ours: SBERT distilroberta, 300k NLI pairs,
# trained with the siamese loop in src/training/train.py. Paper: SRoBERTa-NLI-base.
STS = [
    ("2012 test set", 71.54, 67.20),
    ("2013 test set", 72.49, 73.71),
    ("2014 test set", 70.80, 69.11),
    ("2015 test set", 78.74, 75.58),
    ("2016 test set", 73.69, 72.35),
    ("Benchmark set", 77.77, 75.60),
    ("SICK sentences", 74.46, 71.63),
]
STS_SCALE = (60, 82)
STS_FOOT = [
    ("Average across the seven", "72.2", "the paper reports 74.2"),
    ("After tuning on Swiggy reviews", "74.5", "above the paper, on the same tests"),
]

# ── Figure 02 · search quality ────────────────────────────────────────────────
# results/baselines.csv (retrieval rows) and results/search_benchmark.csv, all
# scored on the same 26 hand-judged queries used since the first baseline.
RETRIEVAL = [
    ("Keyword matching", 6.5, False),
    ("Meaning search, first version", 4.6, False),
    ("Meaning search, tuned on Swiggy", 6.1, False),
    ("Tuned, then re-ordered", 7.6, True),
]
RETRIEVAL_STATS = [
    ("Ahead of keyword matching by", "1.1", "more relevant results in every 10", True),
    ("Time to answer one search", "69 ms", "on a laptop, no server", False),
]

# ── Figure 03 · same meaning, different words ─────────────────────────────────
# results/coherence_example.csv, written by `python -m eval.pick_coherence`.
# The rule: among complaints the model put in one topic that share no complaint
# word at all, the three it placed closest together.
COHERENCE = [
    ("First review", "don’t take this app fact swiggy don’t install this app", None),
    ("Second review", "very Bad app please not use this app", "68%"),
    ("Third review", "do not order anything on this app, worst experience with this app.",
     "67%"),
]
COHERENCE_TOPIC = dict(label="Topic 101", name="Telling others not to install it",
                       count="256", meta="reviews in this topic")

# ── Figure 05 · one topic, week by week ───────────────────────────────────────
# Topic ids read from theme_weekly at run time. Notes describe what the chart shows.
SERIES = [
    dict(theme_id=64, label="Worst service and support",
         note="59 reviews in the week of 20 July, against about 20 in a normal week"),
    dict(theme_id=105, label="Much later than promised",
         note="the largest topic, down from its May level"),
    dict(theme_id=103, label="Waited an hour or more",
         note="falling steadily since May"),
]

# ── Figure 06 · where it falls short ──────────────────────────────────────────
# All four from results/phase5_notes.md, recorded rather than tuned away.
FAILURES = [
    ("40%", "Reviews left ungrouped",
     "The model leaves a review out rather than force it into a topic it does "
     "not fit. Settings that grouped more merged almost everything into one."),
    ("2,855", "One topic is a language, not a subject",
     "Reviews written in Hindi-English collect together whatever they are "
     "complaining about."),
    ("1 in 5", "Reviews with nothing to act on",
     "Around 13,600 are plain praise — good, nice, excellent — carrying no "
     "detail anyone can use."),
    ("34", "Reviews checked by hand, per model",
     "Enough to show this model beats the simplest one. Not enough to separate "
     "it from the next best."),
]

# ── pipeline ──────────────────────────────────────────────────────────────────
PIPELINE = [
    ("01", "Collect", "Google Play reviews, daily"),
    ("02", "Store", "PostgreSQL, 100,000 reviews"),
    ("03", "Train", "the model, written from scratch"),
    ("04", "Simplify", "reduce each review to a compact form"),
    ("05", "Group", "gather similar reviews into topics"),
    ("06", "Name", "give every topic a readable name"),
    ("07", "Serve", "search, and week-by-week tracking"),
]
PIPELINE_HIGHLIGHT = 2      # the step this project contributes

# ── Figure 07 · search ────────────────────────────────────────────────────────
SEARCH_EXAMPLES = ["refund never came back to my account",
                   "driver was rude to me",
                   "the app is unusable on my old phone"]
SEARCH_K = 10      # Figure 02 reports on the top 10, so show the top 10

# ── footer ────────────────────────────────────────────────────────────────────
FOOTER_LEFT = (
    "A rebuild of Reimers &amp; Gurevych (2019), <i>Sentence-BERT: Sentence "
    "Embeddings using Siamese BERT-Networks</i>. The model, both training "
    "objectives and the evaluation loop were written directly in PyTorch. Every "
    "figure here is reproducible from <b>eval/</b> and saved to <b>results/</b>.")
FOOTER_RIGHT = ("100,000 public Google Play reviews · January to August 2026<br>"
                "Independent study, not affiliated with Swiggy")
