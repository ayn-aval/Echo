"""Themes — what people are talking about, and how many said it.

Theme figures use the 64,280 reviews with enough text to cluster, not all
100,000. That is the honest denominator: a one-word "good" cannot belong to a
theme, and pretending otherwise would inflate every count.

Two kinds of theme are shown rather than filtered out, both by explicit decision
recorded in docs/PROGRESS.md:

  * the generic-praise themes ("good", "nice / good / service") — the predicted
    consequence of keeping 2+ word reviews rather than 4+, and arguably a finding
    in themselves: a fifth of the corpus carries no actionable content.
  * theme 39, which groups Hinglish reviews by language rather than by topic —
    the Phase 4 finding that the model never bridged Hinglish to English, showing
    up in the product.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared
from shared import MODEL, sql, st

import plotly.express as px

shared.page("Themes", "🗂️", "110 themes discovered without being told what to look for.")

themes = sql("""
    SELECT t.theme_id, t.label, t.top_terms, t.n_rows, t.n_texts,
           t.avg_rating, r.content AS example
      FROM themes t
      LEFT JOIN reviews r ON r.review_id = t.example_review_id
     WHERE t.model = %s
     ORDER BY t.n_rows DESC""", (MODEL,))

if themes.empty:
    st.error("No themes yet. Run `python -m src.clustering.name_themes`.")
    st.stop()

assigned = sql("""SELECT count(*) FILTER (WHERE theme_id >= 0) AS in_theme,
                         count(*) AS total
                    FROM review_themes WHERE model = %s""", (MODEL,)).iloc[0]

a, b, c = st.columns(3)
a.metric("Themes", f"{len(themes):,}")
b.metric("Reviews with a theme", f"{assigned.in_theme:,}",
         delta=f"{assigned.in_theme / assigned.total * 100:.1f}% of clusterable reviews",
         delta_color="off")
c.metric("Largest theme", f"{themes.n_rows.max():,}",
         delta=f"{themes.n_rows.max() / assigned.total * 100:.1f}% of the corpus",
         delta_color="off")

shared.corpus_note()
st.caption("The remainder is *noise* — HDBSCAN is allowed to say a review belongs "
           "to no theme, which is what stops thousands of one-line reviews being "
           "forced into a cluster and poisoning it.")
st.divider()

top_n = st.slider("Themes to chart", 5, 40, 15)
chart = themes.head(top_n).iloc[::-1]
fig = px.bar(chart, x="n_rows", y="label", orientation="h",
             color="avg_rating", color_continuous_scale="RdYlGn",
             range_color=[1, 5],
             labels={"n_rows": "Reviews", "label": "", "avg_rating": "Avg ★"},
             hover_data={"theme_id": True})
fig.update_layout(height=28 * top_n + 120, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")
st.caption("Colour is the average star rating: red themes are complaints, green "
           "are praise. The complaint themes separate cleanly by rating without "
           "anything being told to look for them.")

st.divider()
st.subheader("Every theme")
st.dataframe(
    themes[["theme_id", "label", "n_rows", "avg_rating", "top_terms", "example"]],
    hide_index=True, width="stretch", height=320,
    column_config={
        "theme_id": st.column_config.NumberColumn("#", width="small"),
        "label": "Theme",
        "n_rows": st.column_config.NumberColumn("Reviews", format="%d"),
        "avg_rating": st.column_config.NumberColumn("Avg ★", format="%.2f"),
        "top_terms": "Distinctive terms",
        "example": "Most representative review",
    })

st.divider()
st.subheader("Drill into a theme")

labels = {int(r.theme_id): f"[{r.theme_id}] {r.label}  ·  {r.n_rows:,} reviews  ·  {r.avg_rating}★"
          for r in themes.itertuples()}
chosen = st.selectbox("Theme", list(labels), format_func=lambda i: labels[i])
row = themes[themes.theme_id == chosen].iloc[0]

m1, m2, m3 = st.columns(3)
m1.metric("Reviews", f"{row.n_rows:,}")
m2.metric("Distinct texts", f"{row.n_texts:,}",
          help="One text can be written by many people; 'good' appears 15,576 times.")
m3.metric("Average rating", f"{row.avg_rating} ★")
st.caption(f"**Distinctive terms:** {row.top_terms}")

order = st.radio("Show", ["Most representative first", "Most recent first",
                          "Lowest rated first"], horizontal=True)
sort = {"Most representative first": "rt.strength DESC",
        "Most recent first": "r.reviewed_at DESC",
        "Lowest rated first": "r.score ASC, rt.strength DESC"}[order]

reviews = sql(f"""
    SELECT r.score, r.reviewed_at::date AS day, r.review_version AS version,
           round(rt.strength::numeric, 2) AS strength, r.content
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
     WHERE rt.model = %s AND rt.theme_id = %s
     ORDER BY {sort} LIMIT 200""", (MODEL, int(chosen)))

st.dataframe(reviews, hide_index=True, width="stretch", height=380,
             column_config={
                 "score": st.column_config.NumberColumn("★", width="small"),
                 "day": "Date", "version": "App version",
                 "strength": st.column_config.ProgressColumn(
                     "Fit", min_value=0.0, max_value=1.0,
                     help="How firmly HDBSCAN placed this review in the theme."),
                 "content": "Review",
             })
st.caption(f"Showing up to 200 of {row.n_rows:,} reviews in this theme.")
