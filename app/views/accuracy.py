"""How it works — what the system does and how well, in plain terms.

Every metric name is translated at the edge. The numbers are the ones in
results/, unchanged; only the labels differ, because "Precision@10" tells a
manager nothing and "8 right out of 10" tells them everything. One unit is used
throughout — out of ten — so the same quantity never appears twice in two forms.
"""

import design
import plotly.graph_objects as go
import streamlit as st
from shared import csv, sql

design.appbar("About", "How it works")

base = csv("baselines.csv")
ret = base[base.task == "retrieval"] if "task" in base.columns else base
prec = ret[ret.dataset == "precision@10"].set_index("model").score
bench = csv("search_benchmark.csv")
acc = bench[bench.measure == "accuracy"] if "measure" in bench.columns else bench
two = acc[acc.config.str.contains("cross-encoder", na=False)]
best = float(two["precision@10"].iloc[0]) if not two.empty else float(prec.max())

design.hero(
    eyebrow="Measured, not claimed",
    headline="When you search, about 8 of the top 10 reviews are the right ones",
    value=f"{best / 10:.1f}",
    unit="right out of every 10 results",
    side=(f"Simple word matching gets <b>{float(prec.get('tfidf', 0)) / 10:.1f}</b>"
          f"<br>Every number here comes from a script in <b>eval/</b>"))

st.markdown("## What it does")
STEPS = [
    ("Reads every review", "100,000 reviews, including Hindi and Hinglish."),
    ("Groups them by meaning", "“App keeps crashing” and “closes by itself” "
                               "count as one problem."),
    ("Ranks what to fix", "By how many people, how unhappy, and whether it "
                          "is growing."),
]
for col, (title, what) in zip(st.columns(3, gap="medium"), STEPS):
    with col:
        st.markdown(
            f"<div class='card' style='height:100%;'><h4>{title}</h4>"
            f"<div style='color:{design.INK_2};font-size:.88rem;line-height:1.55;'>"
            f"{what}</div></div>", unsafe_allow_html=True)


def bars(names, values, colours, suffix="", height=240, xlab=""):
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colours,
        marker_line_width=0,
        text=[f"{v:.1f}{suffix}" for v in values], textposition="outside",
        textfont=dict(color=design.INK_2, size=12),
        hovertemplate="%{y}: %{x:.1f}" + suffix + "<extra></extra>"))
    fig.update_traces(marker_cornerradius=5)
    return design.style(fig, height=height, xlab=xlab)


st.markdown("## Does it find the right reviews?")
rows = [("Matching words only", float(prec.get("tfidf", 0)) / 10, design.MUTED),
        ("Matching meaning", float(prec.get("sbert-domain", 0)) / 10, design.MUTED),
        ("Matching meaning, checked twice", best / 10, design.BLUE)]
st.plotly_chart(
    bars([r[0] for r in rows][::-1], [r[1] for r in rows][::-1],
         [r[2] for r in rows][::-1], "", 230,
         "Right answers out of the top 10"),
    width="stretch", config={"displayModeBar": False})

st.markdown("## Are the topics sensible?")
clus = csv("clustering_comparison.csv")
if not clus.empty:
    label = {"glove-avg": "Simple method", "bert-mean": "Standard method",
             "sbert-domain": "This system"}
    clus = clus.assign(name=clus.model.map(label).fillna(clus.model))
    colour = [design.BLUE if m == "sbert-domain" else design.MUTED
              for m in clus.model]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Reviews put in the right topic")
        st.plotly_chart(bars(clus.name[::-1], clus.audit_accuracy[::-1] / 10,
                             colour[::-1], "", 210,
                             "Correct out of every 10, checked by hand"),
                        width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("### Reviews dumped in one giant group")
        st.plotly_chart(bars(clus.name[::-1], clus.largest_pct[::-1] / 10,
                             colour[::-1], "", 210,
                             "Out of every 10 reviews — lower is better"),
                        width="stretch", config={"displayModeBar": False})
    design.note("102 reviews were checked by hand without knowing which method "
                "produced them.")

st.markdown("## What it gets wrong")
hidden = sql("""SELECT coalesce(display_name, label) AS name, n_rows
                  FROM themes WHERE model='sbert-domain' AND NOT actionable
                 ORDER BY n_rows DESC""")
LIMITS = [
    ("Hindi and Hinglish", "4 in 100 reviews",
     "It sees that they look alike but not what they say, so it can group them "
     "by language instead of subject."),
    ("Reviews with no detail", "1 in 5 reviews",
     "“Good” and “nice” form some of the largest topics. Real, but nothing to "
     "act on."),
    ("How much was checked", "26 searches, 102 topics",
     "Enough to separate methods that clearly differ, not enough to separate "
     "the close ones."),
]
for col, (title, scale, what) in zip(st.columns(3, gap="medium"), LIMITS):
    with col:
        st.markdown(
            f"<div class='card' style='height:100%;'><h4>{title}</h4>"
            f"<div style='color:{design.CRITICAL};font-weight:660;"
            f"font-size:.85rem;margin-bottom:7px;'>{scale}</div>"
            f"<div style='color:{design.INK_2};font-size:.88rem;line-height:1.55;'>"
            f"{what}</div></div>", unsafe_allow_html=True)

if not hidden.empty:
    st.markdown("### Topics kept off the other screens")
    worst = int(hidden.n_rows.max())
    design.rank_rows([(r.name, r.n_rows / worst, f"{r.n_rows:,}", "reviews",
                       design.MUTED) for r in hidden.itertuples()])
    design.note("These are real groupings the system found, and no team can act "
                "on them. They still count towards every total.")

with st.expander("Technical detail"):
    st.markdown("""
| measure | result |
|---|---|
| Sentence-BERT reproduction (STS avg) | 74.54 · paper 74.21 |
| Search, Precision@10 — word matching | 65.00 |
| Search, Precision@10 — trained model | 61.15 |
| Search, Precision@10 — with reranking | **75.77** |
| Topic audit — this model | 82.4% |
| Topic audit — averaged GloVe | 44.1% |

74.54 is a different training recipe, not a better result on the same one.
The 82.4% vs 73.5% gap over mean-pooled BERT is **not** significant (p=0.56).
Full write-ups in `results/phase3_notes.md` onward.
""")
