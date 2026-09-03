"""How it works — what the system does and how well, in plain terms.

Every metric name is translated at the edge. The underlying numbers are the ones
in results/, unchanged; only the labels differ, because "Precision@10" tells a
manager nothing and "8 of the top 10 results were right" tells them everything.
"""

import design
import plotly.graph_objects as go
import streamlit as st
from shared import csv

design.appbar("About", "How it works")

st.markdown("## What it does")
c1, c2, c3 = st.columns(3, gap="medium")
STEPS = [
    ("Reads every review", "100,000 reviews, including ones written in Hindi "
                           "and Hinglish."),
    ("Groups them by meaning", "“App keeps crashing” and “closes by itself” "
                               "count as the same problem."),
    ("Ranks what to fix", "By how many people, how unhappy they were, and "
                          "whether it is growing."),
]
for col, (title, what) in zip((c1, c2, c3), STEPS):
    with col:
        st.markdown(
            f"<div class='card' style='height:100%;'><h4>{title}</h4>"
            f"<div style='color:{design.INK_2};font-size:.88rem;line-height:1.55;'>"
            f"{what}</div></div>", unsafe_allow_html=True)


def bars(names, values, colours, suffix="%", height=250, xlab=""):
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colours,
        marker_line_width=0,
        text=[f"{v:.0f}{suffix}" for v in values], textposition="outside",
        textfont=dict(color=design.INK_2, size=12),
        hovertemplate="%{y}: %{x:.1f}" + suffix + "<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    return design.style(fig, height=height, xlab=xlab)


# ── search quality ──────────────────────────────────────────────────────────
st.markdown("## Does it find the right reviews?")

base = csv("baselines.csv")
ret = base[base.task == "retrieval"] if "task" in base.columns else base
prec = ret[ret.dataset == "precision@10"].set_index("model").score
bench = csv("search_benchmark.csv")
acc = bench[bench.measure == "accuracy"] if "measure" in bench.columns else bench
two = acc[acc.config.str.contains("cross-encoder", na=False)]

rows = [("Matching words only", float(prec.get("tfidf", 0)), design.MUTED),
        ("Matching meaning", float(prec.get("sbert-domain", 0)), design.MUTED)]
if not two.empty:
    rows.append(("Matching meaning, checked twice",
                 float(two["precision@10"].iloc[0]), design.BLUE))

st.plotly_chart(
    bars([r[0] for r in rows][::-1], [r[1] for r in rows][::-1],
         [r[2] for r in rows][::-1], "%", 220,
         "Out of every 10 results, how many were right"),
    width="stretch", config={"displayModeBar": False})

design.tiles([
    ("Results that are right", "8 in 10", "when searching by meaning"),
    ("Simple word search", "6.5 in 10", "the method this had to beat"),
    ("Time to search", "under a second", "across 45,864 reviews"),
])

# ── grouping quality ────────────────────────────────────────────────────────
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
        st.plotly_chart(bars(clus.name[::-1], clus.audit_accuracy[::-1],
                             colour[::-1], "%", 210,
                             "Checked by hand, method hidden"),
                        width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("### Reviews stuck in one giant group")
        st.plotly_chart(bars(clus.name[::-1], clus.largest_pct[::-1],
                             colour[::-1], "%", 210, "Lower is better"),
                        width="stretch", config={"displayModeBar": False})

design.note("102 reviews were checked by hand, without knowing which method "
            "produced them.")

# ── limits ──────────────────────────────────────────────────────────────────
st.markdown("## What it gets wrong")

LIMITS = [
    ("Hindi and Hinglish", "4% of reviews",
     "It spots that they look alike but not what they say, so it can group "
     "them by language instead of subject."),
    ("Reviews with no detail", "1 in 5 reviews",
     "“Good” and “nice” form some of the largest topics. Real, but nothing to "
     "act on."),
    ("How much was checked", "26 searches, 102 topics",
     "Enough to separate the methods that clearly differ, not enough to "
     "separate the close ones."),
]
cols = st.columns(3, gap="medium")
for col, (title, scale, what) in zip(cols, LIMITS):
    with col:
        st.markdown(
            f"<div class='card' style='height:100%;'>"
            f"<h4>{title}</h4>"
            f"<div style='color:{design.CRITICAL};font-weight:640;"
            f"font-size:.85rem;margin-bottom:6px;'>{scale}</div>"
            f"<div style='color:{design.INK_2};font-size:.88rem;line-height:1.55;'>"
            f"{what}</div></div>", unsafe_allow_html=True)

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
