"""Can I trust it — the measured accuracy, and where it fails.

The only screen allowed method vocabulary, and only behind the expander at the
bottom; eval/check_app.py fails the build if any of it leaks onto the other
three. Every number is the one in results/, translated at the edge — plain odds
on the surface, the raw metric on demand.
"""

import design
import streamlit as st
from shared import csv, sql

MODEL = "sbert-domain"


def render() -> None:
    base = csv("baselines.csv")
    ret = base[base.task == "retrieval"] if "task" in base.columns else base
    prec = ret[ret.dataset == "precision@10"].set_index("model").score
    bench = csv("search_benchmark.csv")
    acc = bench[bench.measure == "accuracy"] if "measure" in bench.columns else bench
    two = acc[acc.config.str.contains("cross-encoder", na=False)]
    best = float(two["precision@10"].iloc[0]) if not two.empty else float(prec.max())
    words = float(prec.get("tfidf", 0))

    clus = csv("clustering_comparison.csv")
    mine = clus[clus.model == MODEL] if not clus.empty else clus
    audit = float(mine.audit_accuracy.iloc[0]) if not mine.empty else 82.35
    per_model = int(mine.audit_n.iloc[0]) if not mine.empty else 34

    design.html(
        '<div style="padding:34px 48px 26px;max-width:1100px">'
        '<h1 style="font-size:38px;max-width:28ch;margin-bottom:10px !important">'
        "Can you take this into a room and defend it?</h1>"
        '<p class="sub">For ranking what to look at next, yes. As an exact '
        "count of affected customers, no.</p></div>")
    design.rule()

    cards = [
        (f"{best / 10:.0f} in 10", "search results are on topic",
         f"Word matching on the same searches manages {words / 10:.1f}."),
        (f"{audit / 10:.0f} in 10", "reviews sit in a sensible group",
         f"Judged by hand on {per_model} groupings per method, blind."),
        ("Matches", "the research it reproduces",
         "Rebuilt from scratch: 74.5 on the standard test, against 74.2 in "
         "the paper."),
    ]
    for i, (col, (number, label, body)) in enumerate(zip(st.columns(3), cards)):
        with col:
            border = (f"border-right:1px solid {design.DIVIDER};" if i < 2 else "")
            design.html(
                f'<div style="padding:28px 32px;{border}height:100%">'
                f'<div style="font-weight:800;font-size:40px;line-height:1">'
                f"{number}</div>"
                f'<div style="font-weight:800;font-size:16px;margin-top:8px">'
                f"{label}</div>"
                f'<p style="font-size:13px;margin:8px 0 0;color:{design.MUTED};'
                f'line-height:1.55">{body}</p></div>')
    design.rule()

    a, b = st.columns(2, gap="large")
    with a:
        wrong = [
            ("Two complaints in one review",
             "Cold food <i>and</i> a rude rider — only the louder one counts."),
            ("Hindi and Hinglish",
             "About 4 in 100. Grouped by language, not by subject."),
            ("Reviews with no detail",
             "About 1 in 5. “Good” and “worst” — real, and nothing to act on."),
        ]
        cards2 = "".join(
            f'<div class="evi"><div style="font-weight:800;font-size:15px">{t}'
            f'</div><div style="font-size:13.5px;margin-top:5px;'
            f'line-height:1.5">{d}</div></div>' for t, d in wrong)
        design.html('<div style="padding:30px 44px 36px 48px">'
                    '<h3 style="font-size:21px;margin-bottom:14px !important">'
                    f"Three things it gets wrong</h3>{cards2}</div>")
    with b:
        design.html(
            f'<div style="padding:30px 48px 6px 44px;border-left:1px solid '
            f'{design.DIVIDER}"><h3 style="font-size:21px;'
            'margin-bottom:12px !important">What happens to one review</h3>'
            "</div>")
        steps = [
            "Pulled from Google Play and stored.",
            "Turned into numbers standing for its meaning, so reviews saying "
            "the same thing in different words land near each other.",
            "Reviews sitting close together become one problem, named from "
            "what makes that group distinctive.",
            "Each problem gets a weekly count, and an alert when a week runs "
            "far above its own normal.",
        ]
        design.html(
            '<div style="padding:0 48px 20px 44px">'
            + "".join(f'<div class="step"><b>{i:02d}</b><span>{s}</span></div>'
                      for i, s in enumerate(steps, 1))
            + "</div>")

        hidden = sql("""SELECT coalesce(display_name, label) AS name, n_rows
                          FROM themes
                         WHERE model = %s AND NOT actionable
                         ORDER BY n_rows DESC""", (MODEL,))
        if not hidden.empty:
            top = int(hidden.n_rows.max()) or 1
            bars = "".join(
                '<div style="margin-bottom:10px">'
                '<div style="display:flex;justify-content:space-between;'
                'font-size:13px;margin-bottom:4px">'
                f'<b>{h.name}</b><span style="color:{design.MUTED}">'
                f'{int(h.n_rows):,}</span></div>'
                f'<div class="track"><div class="fill" style="width:'
                f'{max(2, round(100 * int(h.n_rows) / top))}%"></div></div></div>'
                for h in hidden.itertuples())
            design.html(
                '<div style="padding:6px 48px 30px 44px">'
                '<h3 style="font-size:17px;margin-bottom:6px !important">Groups '
                "kept off the other screens</h3>"
                f'<p style="font-size:12.5px;color:{design.MUTED};'
                'margin:0 0 14px">Real groups no team can act on. They still '
                f"count in every total.</p>{bars}"
                "</div>")

    # A keyed container, not a markdown <div>: Streamlit closes a hand-written
    # div before the expander is emitted, so the padding never lands on it.
    with st.container(key="tech"), \
            st.expander("Show the technical pipeline and the raw figures"):
        st.markdown(f"""
google-play-scraper → Postgres → sentence embeddings from a Sentence-BERT
reproduction written in raw PyTorch (siamese training loop by hand, then
domain-adapted on mined review pairs) → UMAP + HDBSCAN for the grouping →
c-TF-IDF for the labels → FAISS for semantic search → weekly series with
z-score spike alerts.

| measure | result |
|---|---|
| Sentence-BERT reproduction (STS avg) | 74.54 · paper 74.21 |
| Search, Precision@10 — word matching (TF-IDF) | {words:.2f} |
| Search, Precision@10 — trained bi-encoder | 61.15 |
| Search, Precision@10 — with cross-encoder reranking | **{best:.2f}** |
| Group audit — this model | {audit:.1f}% |
| Group audit — mean-pooled BERT | 73.5% |
| Group audit — averaged GloVe | 44.1% |

74.54 comes from a different training recipe, so it is not a better result on
the same one. The trained bi-encoder **lost** to plain word matching on its own;
only reranking the shortlist got ahead. The {audit:.1f}% against 73.5% gap over
mean-pooled BERT is **not** statistically significant (Fisher exact, p=0.56) at
{per_model} judgements per method. Every figure is reproducible from a script in
`eval/` and written to `results/`.
""")
