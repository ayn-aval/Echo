"""Results — does this actually work, and how do we know."""

import design
import plotly.graph_objects as go
from shared import csv, st

design.appbar("About", "How it works",
              "What this system does, how accurate it is, and where it falls short.")


def bars(names, values, colours, suffix="", height=250, xlab=""):
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colours,
        marker_line_width=0,
        text=[f"{v:.1f}{suffix}" for v in values], textposition="outside",
        textfont=dict(color=design.INK_2, size=12),
        hovertemplate="%{y}: %{x:.1f}" + suffix + "<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    return design.style(fig, height=height, xlab=xlab)


# ── Test 1: search ──────────────────────────────────────────────────────────
st.markdown("## Test one — can it find the right reviews?")
design.note("26 searches were written by hand, and every review they returned "
            "was judged relevant or not by a person. The score is how many of "
            "the top ten results were genuinely relevant.")

bench = csv("search_benchmark.csv")
base = csv("baselines.csv")
ret = base[base.task == "retrieval"] if "task" in base.columns else base
prec = ret[ret.dataset == "precision@10"].set_index("model").score

rows = [
    ("Word matching (the simple approach)", float(prec.get("tfidf", 0)), design.MUTED),
    ("Word meanings, averaged", float(prec.get("glove-avg", 0)), design.MUTED),
    ("Off-the-shelf language model", float(prec.get("bert-mean", 0)), design.MUTED),
    ("This project, trained model", float(prec.get("sbert-domain", 0)), design.ORANGE),
]
acc = bench[bench.measure == "accuracy"] if "measure" in bench.columns else bench
two = acc[acc.config.str.contains("cross-encoder", na=False)]
if not two.empty:
    rows.append(("This project, careful search",
                 float(two["precision@10"].iloc[0]), design.BLUE))

names = [r[0] for r in rows][::-1]
st.plotly_chart(bars(names, [r[1] for r in rows][::-1],
                     [r[2] for r in rows][::-1], "%", 300,
                     "Relevant results in the top ten (%)"),
                width="stretch", config={"displayModeBar": False})

design.tiles([
    ("Best result", "75.8%", "of the top ten results were relevant"),
    ("Simple word matching", "65.0%", "the method this had to beat"),
    ("Speed", "8 ms", "for an instant search over 45,864 reviews"),
])

with st.expander("Why word matching was hard to beat, and how it was finally beaten"):
    st.markdown("""
Plain word matching scored 65%, and for five stages of this project nothing beat
it. That is worth saying plainly: on short app reviews, matching words is a
genuinely strong method, and a great deal of work went into passing it.

Two things closed the gap. Training the model on Swiggy reviews specifically took
it from 45.8% to 61.2%. Adding a second, slower pass that re-reads the top fifty
results took it to 75.8%.

**One number here was almost reported wrongly.** The first measurement said the
second pass made things *worse*. It had not — it was surfacing good reviews that
nobody had ever judged, and unjudged results count as wrong. Once those 133
reviews were judged, the real answer appeared. The measurement was fixed rather
than the result being explained away.
""")

# ── Test 2: topics ──────────────────────────────────────────────────────────
st.markdown("## Test two — are the topics any good?")
design.note("102 reviews were shown next to the topic they had been put in, and "
            "judged one by one without knowing which method produced them.")

clus = csv("clustering_comparison.csv")
if not clus.empty:
    label = {"glove-avg": "Word meanings, averaged",
             "bert-mean": "Off-the-shelf language model",
             "sbert-domain": "This project, trained model"}
    clus = clus.assign(name=clus.model.map(label).fillna(clus.model))
    colour = [design.BLUE if m == "sbert-domain" else design.MUTED
              for m in clus.model]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Reviews placed in the right topic")
        st.plotly_chart(bars(clus.name[::-1], clus.audit_accuracy[::-1],
                             colour[::-1], "%", 230, "Judged correct (%)"),
                        width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("### Largest single topic")
        st.plotly_chart(bars(clus.name[::-1], clus.largest_pct[::-1],
                             colour[::-1], "%", 230,
                             "Share of all reviews in one topic (%)"),
                        width="stretch", config={"displayModeBar": False})
        design.note("Lower is better. A method that drops most reviews into one "
                    "giant group has not really sorted anything.")

with st.expander("Two things this test does not prove"):
    st.markdown("""
**The lead over the off-the-shelf model is not proven.** 82.4% against 73.5%
looks decisive, but with only 34 judgements per method a gap that size can appear
by chance — the statistical test gives a 56% probability it is noise. What is not
in doubt is the shape of the result: this model produced 110 usable topics with
the largest holding 6% of reviews, against 42 topics with the largest holding
41%. A topic containing 41% of your reviews cannot be acted on.

**One measure here would rank the worst method first.** The simplest method leaves
the fewest reviews ungrouped, which sounds good, and achieves it by putting 59% of
everything into a single group mixing five-star praise with refund complaints.
That is why the second chart is shown beside the first.
""")

# ── Test 3: the research reproduction ───────────────────────────────────────
st.markdown("## Test three — does the underlying method reproduce?")
design.note("The model is a reproduction of a published research paper, "
            "Sentence-BERT (2019). Scoring it on the same public benchmarks the "
            "paper used shows whether the reproduction worked.")

t1 = csv("table1_comparison.csv").rename(columns={"Unnamed: 0": "method"})
trained = csv("sts_trained.csv")

design.tiles([
    ("The paper's score", "74.21", "reported by the original authors"),
    ("This reproduction", "74.54", "on a third of the training data"),
    ("Before adapting to reviews", "72.17", "the plain reproduction"),
])

if not t1.empty:
    st.dataframe(t1, hide_index=True, width="stretch")
    design.note("Higher is better; 100 would mean perfect agreement with human "
                "judgement. Each column is a different public benchmark.")

with st.expander("Why 74.54 is not 'beating the paper'"):
    st.markdown("""
The reproduction scored 74.54 where the paper reports 74.21, but those are not
the same experiment. The paper trained one way; this added a second training
stage on Swiggy reviews that the paper never ran. It is a different recipe, not a
better result on the same one.

The plain reproduction scored **72.17**, about two points below the paper, using
roughly a third of the training data. That gap is the honest comparison.

There is also a known flaw in the comparison baseline: one of the simpler methods
scores 7.4 points lower here than in the paper, because of a difference in how the
average is calculated. Any improvement quoted against that baseline is therefore
flattering, and the paper's own margin is the safer reference.
""")

st.markdown("## What this system cannot do")
st.markdown(
    f"<div class='card' style='color:{design.INK_2};line-height:1.65;'>"
    "<strong style='color:#0b0b0b;'>Hindi and Hinglish are not understood.</strong> "
    "Around 4% of reviews are written in romanised Hindi. The system recognises "
    "that they look alike but not what they say, so it groups them by language "
    "rather than by subject, and can return the opposite meaning in search.<br><br>"
    "<strong style='color:#0b0b0b;'>A fifth of reviews say nothing specific.</strong> "
    "Reviews reading only \"good\" or \"nice\" form some of the largest topics. "
    "They are shown rather than hidden, because their volume is itself a finding.<br><br>"
    "<strong style='color:#0b0b0b;'>Every score rests on 26 searches and 102 "
    "judgements</strong>, made by one person. That is enough to separate the "
    "methods that clearly differ and not enough to separate the close ones."
    "</div>", unsafe_allow_html=True)
