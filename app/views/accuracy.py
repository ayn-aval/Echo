"""Is it accurate? — how often the system is right, and where it is not.

Every metric is translated at the edge. The numbers are the ones in results/,
unchanged; only the labels differ, because "Precision@10" tells a manager
nothing and "8 right out of 10" tells them everything. Method vocabulary is
confined to the expander at the bottom — this is the only screen allowed any,
and eval/check_app.py enforces that on the other three.
"""

import design
import plotly.graph_objects as go
import streamlit as st
from shared import csv, sql

base = csv("baselines.csv")
ret = base[base.task == "retrieval"] if "task" in base.columns else base
prec = ret[ret.dataset == "precision@10"].set_index("model").score
bench = csv("search_benchmark.csv")
acc = bench[bench.measure == "accuracy"] if "measure" in bench.columns else bench
two = acc[acc.config.str.contains("cross-encoder", na=False)]
best = float(two["precision@10"].iloc[0]) if not two.empty else float(prec.max())
words = float(prec.get("tfidf", 0))

clus = csv("clustering_comparison.csv")
mine = clus[clus.model == "sbert-domain"] if not clus.empty else clus
audit = float(mine.audit_accuracy.iloc[0]) if not mine.empty else 82.4

design.kicker("Measured, not claimed")
st.markdown("# Should you trust the list?")
design.lede(
    "Mostly yes, for deciding <b>what to look at next</b> — and no, not as an "
    "exact count of affected customers. Here is what was measured, and by which "
    "script.")

design.rule()

STATS = [
    (f"{best / 10:.0f} in 10", "search results are on topic",
     f"Of the top 10 reviews returned for a search, about {best / 10:.0f} are "
     f"about what you asked. Plain word matching on the same searches manages "
     f"{words / 10:.1f}."),
    (f"{audit / 10:.0f} in 10", "reviews are in a sensible group",
     "Checked by hand on 102 reviews, without knowing which method produced "
     "each one. The rest usually mention two problems at once and get filed "
     "under the louder one."),
    ("Matches the paper", "on the standard benchmark",
     "The sentence model here was rebuilt from scratch and scores 74.5 on the "
     "industry sentence-similarity test, against 74.2 in the original paper."),
]
for col, (number, label, body) in zip(st.columns(3, gap="medium"), STATS):
    with col:
        design.stat(number, label)
        design.sub(body)

design.rule()


def bars(names, values, colours, height=230, xlab=""):
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colours,
        marker_line_width=0,
        text=[f"{v:.1f}" for v in values], textposition="outside",
        textfont=dict(color=design.INK, size=12),
        hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
    fig.update_xaxes(range=[0, max(values) * 1.18])
    return design.style(fig, height=height, xlab=xlab)


left, right = st.columns(2, gap="large")
with left:
    st.markdown("### Does it find the right reviews?")
    design.sub("Right answers in the top 10, over 26 searches labelled by hand.")
    rows = [("Matching words only", words / 10),
            ("Matching meaning", float(prec.get("sbert-domain", 0)) / 10),
            ("Matching meaning, checked twice", best / 10)]
    st.plotly_chart(
        bars([r[0] for r in rows][::-1], [r[1] for r in rows][::-1],
             [design.ACCENT, design.NEUTRAL_700, design.NEUTRAL_700],
             xlab="Right answers out of the top 10"),
        width="stretch", config={"displayModeBar": False})
    design.sub("Word matching beat the trained model on its own. Only checking "
               "the shortlist a second time got ahead of it — an honest result, "
               "not the one that was expected.")

with right:
    st.markdown("### Are the groups sensible?")
    design.sub("Correct out of every 10, judged by hand without knowing which "
               "method produced each.")
    if not clus.empty:
        label = {"glove-avg": "Simple method", "bert-mean": "Standard method",
                 "sbert-domain": "This system"}
        names = clus.model.map(label).fillna(clus.model).tolist()
        colours = [design.ACCENT if m == "sbert-domain" else design.NEUTRAL_700
                   for m in clus.model]
        st.plotly_chart(
            bars(names[::-1], (clus.audit_accuracy / 10).tolist()[::-1],
                 colours[::-1], xlab="Correct out of every 10"),
            width="stretch", config={"displayModeBar": False})
        design.sub("The gap over the standard method is real but small, and 102 "
                   "judgements is not enough to be sure of it.")

design.rule()

# ── the honest half ─────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")
with left:
    st.markdown("### Where it gets things wrong")
    for title, body in [
        ("Two complaints in one review",
         "Cold food <i>and</i> a rude rider — only the louder one gets counted."),
        ("Hindi and Hinglish", "About 4 in 100 reviews. The system sees that "
         "they look alike but not what they say, so it can group them by "
         "language instead of by subject."),
        ("Reviews with no detail", "About 1 in 5. “Good” and “nice” form some "
         "of the largest groups. Real, and nothing to act on."),
        ("How much was checked", "26 searches and 102 group judgements. Enough "
         "to separate methods that clearly differ, not enough to separate the "
         "close ones."),
    ]:
        design.limit(title, body)

with right:
    st.markdown("### What happens to a review")
    st.markdown(
        "1. Every review is pulled from Google Play and stored.\n"
        "2. Each is turned into a set of numbers standing for its *meaning*, so "
        "reviews that say the same thing in different words end up near each "
        "other.\n"
        "3. Reviews that sit close together become one problem, and the problem "
        "is named from the words that make it distinctive.\n"
        "4. Each problem gets a weekly count, and an alert fires when a week is "
        "far above that problem's own recent normal.")

    hidden = sql("""SELECT coalesce(display_name, label) AS name, n_rows
                      FROM themes WHERE model='sbert-domain' AND NOT actionable
                     ORDER BY n_rows DESC""")
    if not hidden.empty:
        st.markdown("### Groups kept off the other screens")
        design.sub("Real groupings that no team can act on. They still count "
                   "towards every total on this site.")
        top = int(hidden.n_rows.max()) or 1
        for h in hidden.itertuples():
            design.bar(h.name, f"{int(h.n_rows):,}", 100 * int(h.n_rows) / top)

st.write("")
with st.expander("Show the technical pipeline and the raw figures"):
    st.markdown("""
google-play-scraper → Postgres → sentence embeddings from a Sentence-BERT
reproduction written in raw PyTorch (siamese training loop by hand, then
domain-adapted on mined review pairs) → UMAP + HDBSCAN for the grouping →
c-TF-IDF for the labels → FAISS for semantic search → weekly series with
z-score spike alerts.

| measure | result |
|---|---|
| Sentence-BERT reproduction (STS avg) | 74.54 · paper 74.21 |
| Search, Precision@10 — word matching (TF-IDF) | 65.00 |
| Search, Precision@10 — trained bi-encoder | 61.15 |
| Search, Precision@10 — with cross-encoder reranking | **75.77** |
| Group audit — this model | 82.4% |
| Group audit — mean-pooled BERT | 73.5% |
| Group audit — averaged GloVe | 44.1% |

74.54 comes from a different training recipe, so it is not a better result on
the same one. The 82.4% against 73.5% gap over mean-pooled BERT is **not**
statistically significant (Fisher exact, p=0.56). Every figure is reproducible
from a script in `eval/` and written to `results/`.
""")
