"""Topics — what customers talk about, grouped by meaning.

Every topic carries a name written from its contents (src/clustering/theme_names.py).
Topics nobody can act on — reviews grouped by the language they are written in,
and reviews that are angry without saying why — are excluded here and shown on
How it works instead.
"""

import design
import plotly.graph_objects as go
from shared import MODEL, sql, st

design.appbar("Understand", "Topics")

F = st.session_state["filters"]

themes = sql(f"""
    SELECT t.theme_id, coalesce(t.display_name, t.label) AS label,
           coalesce(t.category, 'Other') AS area,
           t.top_terms, t.n_rows, t.avg_rating,
           -- The review nearest the cluster centre is the most typical, but for
           -- a praise topic that is often literally "good" — true, and useless
           -- to read. Among the 200 most typical, show the one with the most to
           -- say, so the example illustrates the topic instead of just proving
           -- it exists.
           (SELECT y.content FROM (
              SELECT x.content, x.word_count
                FROM review_themes rt
                JOIN reviews x ON x.app = rt.app AND x.review_id = rt.review_id
               WHERE rt.model = t.model AND rt.theme_id = t.theme_id
               ORDER BY rt.strength DESC LIMIT 200) y
            ORDER BY y.word_count DESC LIMIT 1) AS example
      FROM themes t
     WHERE t.model = %s AND t.actionable {F.area_clause('t')}
     ORDER BY t.n_rows DESC""", (MODEL,))

if themes.empty:
    st.error("No topics yet. Run: python -m src.clustering.name_themes")
    st.stop()

complaints = themes[themes.avg_rating <= 2.5]
praise = themes[themes.avg_rating >= 4.0]
biggest = complaints.nlargest(1, "n_rows").iloc[0]

design.hero(
    eyebrow="Grouped automatically from what people wrote",
    headline=f"The single biggest complaint is “{biggest.label.lower()}”",
    value=f"{biggest.n_rows:,}",
    unit="customers said it",
    side=(f"<b>{len(themes)}</b> subjects found in all<br>"
          f"<b>{len(complaints)}</b> are complaints, totalling "
          f"<b>{complaints.n_rows.sum():,}</b> reviews"))

view = st.radio("Show", ["Complaints", "Praise", "Everything"],
                horizontal=True, label_visibility="collapsed")
subset = {"Complaints": complaints, "Praise": praise,
          "Everything": themes}[view].nlargest(12, "n_rows")

st.markdown(f"## Most talked about — {view.lower()}")

bars = subset.iloc[::-1]
colours = [design.NEGATIVE if r <= 2.5 else design.NEUTRAL if r < 4
           else design.POSITIVE for r in bars.avg_rating]
fig = go.Figure(go.Bar(
    x=bars.n_rows, y=bars.label, orientation="h",
    marker_color=colours, marker_line_width=0,
    text=[f"{n:,}" for n in bars.n_rows], textposition="outside",
    textfont=dict(color=design.INK_2, size=12),
    customdata=bars.avg_rating,
    hovertemplate="%{y}<br>%{x:,} reviews<br>%{customdata:.1f} stars"
                  "<extra></extra>"))
fig.update_traces(marker_cornerradius=5)
st.plotly_chart(design.style(fig, height=40 * len(bars) + 90, xlab="Reviews"),
                width="stretch", config={"displayModeBar": False})
if view == "Everything":
    design.note("Red: rated 2 stars or less. Blue: rated 4 stars or more.")

st.markdown("## Look inside a topic")

labels = {int(r.theme_id): f"{r.label}  —  {r.n_rows:,} reviews, {r.avg_rating:.1f} stars"
          for r in themes.itertuples()}
# Default to the largest complaint rather than the largest topic overall. The
# largest topic is one-word praise — accurate, and a poor thing to land on.
ids = list(labels)
default_id = int(biggest.theme_id) if not complaints.empty else ids[0]
chosen = st.selectbox("Choose a topic", ids, index=ids.index(default_id),
                      format_func=lambda i: labels[i], label_visibility="collapsed")
row = themes[themes.theme_id == chosen].iloc[0]

mood = ("Unhappy" if row.avg_rating <= 2.5
        else "Mixed" if row.avg_rating < 4 else "Happy")
pill = "pill-neg" if row.avg_rating <= 2.5 else "pill-pos"
st.markdown(
    f"<div class='card'><span class='pill {pill}'>{mood}</span>"
    f"<span class='area-tag'>{row.area}</span>"
    f"<div style='margin-top:14px;font-size:1.05rem;color:{design.INK};'>"
    f"<strong>{row.n_rows:,} reviews</strong> &nbsp;·&nbsp; "
    f"{row.avg_rating:.1f} stars on average</div>"
    f"<div class='issue quote' style='margin-top:14px;border:0;'>"
    f"“{str(row.example)[:400]}”</div>"
    f"<div style='margin-top:9px;color:{design.MUTED};font-size:.76rem;'>"
    f"A typical review from this topic &nbsp;·&nbsp; "
    f"words that set it apart: {row.top_terms}</div></div>",
    unsafe_allow_html=True)

order = st.radio("Sort reviews by", ["Most typical", "Newest", "Lowest rated"],
                 horizontal=True, label_visibility="collapsed")
sort = {"Most typical": "rt.strength DESC", "Newest": "r.reviewed_at DESC",
        "Lowest rated": "r.score ASC, rt.strength DESC"}[order]

reviews = sql(f"""
    SELECT r.score AS stars, r.reviewed_at::date AS date, r.content AS review
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
     WHERE rt.model = %s AND rt.theme_id = %s
     ORDER BY {sort} LIMIT 200""", (MODEL, int(chosen)))

st.dataframe(reviews, hide_index=True, width="stretch", height=360,
             column_config={
                 "stars": st.column_config.NumberColumn("Stars", width="small"),
                 "date": st.column_config.DateColumn("Date", width="small"),
                 "review": "Review"})
design.note(f"Showing {len(reviews)} of {row.n_rows:,} reviews in this topic.")

with st.expander("About these topics"):
    st.markdown(
        "- Topics are found from the reviews themselves — nobody wrote the list "
        "in advance.\n"
        "- About a third of reviews are too vague to place in any topic.\n"
        "- Four topics are left out here because no team can act on them: two "
        "group reviews by the language they are written in, and two hold "
        "reviews that are angry without saying why. They are on **How it works**.")
