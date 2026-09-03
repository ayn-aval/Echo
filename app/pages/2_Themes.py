"""Themes — what customers talk about, grouped by meaning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import design
import plotly.graph_objects as go
from shared import MODEL, sql, st

design.setup("Topics")
design.header(
    "What customers talk about",
    "Reviews are grouped by what they mean, not the words they use. A complaint "
    "about the app crashing and one about it closing by itself land in the same "
    "group, even though they share no words.")

themes = sql("""
    SELECT t.theme_id, coalesce(t.display_name, t.label) AS label,
           t.top_terms, t.n_rows, t.avg_rating,
           -- The review nearest the cluster centre is the most typical, but for
           -- a praise topic that is often literally "good" — true, and useless
           -- to read. Among the 50 most typical, show the one with the most to
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
     WHERE t.model = %s ORDER BY t.n_rows DESC""", (MODEL,))

if themes.empty:
    st.error("No topics yet. Run: python -m src.clustering.name_themes")
    st.stop()

complaints = themes[themes.avg_rating <= 2.5]
praise = themes[themes.avg_rating >= 4.0]

design.tiles([
    ("Topics found", f"{len(themes):,}", "discovered automatically"),
    ("Complaint topics", f"{len(complaints):,}",
     f"{complaints.n_rows.sum():,} reviews averaging 2 stars or less"),
    ("Biggest complaint", f"{complaints.n_rows.max():,}",
     f"reviews about {complaints.nlargest(1, 'n_rows').label.iloc[0]}"),
])

view = st.radio("Show", ["Complaints", "Praise", "Everything"],
                horizontal=True, label_visibility="collapsed")
subset = {"Complaints": complaints, "Praise": praise,
          "Everything": themes}[view].nlargest(12, "n_rows")

st.markdown(f"## Top {view.lower()} by number of reviews")

bars = subset.iloc[::-1]
colours = [design.NEGATIVE if r <= 2.5 else design.NEUTRAL if r < 4
           else design.POSITIVE for r in bars.avg_rating]
fig = go.Figure(go.Bar(
    x=bars.n_rows, y=bars.label, orientation="h",
    marker_color=colours, marker_line_width=0,
    text=[f"{n:,}" for n in bars.n_rows], textposition="outside",
    textfont=dict(color=design.INK_2, size=12),
    customdata=bars.avg_rating,
    hovertemplate="%{y}<br>%{x:,} reviews<br>%{customdata:.1f} stars average"
                  "<extra></extra>"))
fig.update_traces(marker_cornerradius=4)
st.plotly_chart(design.style(fig, height=40 * len(bars) + 90),
                width="stretch", config={"displayModeBar": False})
design.note("Red bars are topics where customers rated 2 stars or less. "
            "Blue bars are topics where they rated 4 stars or more.")

st.markdown("## Look inside a topic")

labels = {int(r.theme_id): f"{r.label}  ({r.n_rows:,} reviews, {r.avg_rating:.1f} stars)"
          for r in themes.itertuples()}
# Default to the largest complaint rather than the largest topic overall. The
# largest topic is one-word praise — accurate, and a poor thing to land on.
ids = list(labels)
default_id = int(complaints.nlargest(1, "n_rows").theme_id.iloc[0]) \
    if not complaints.empty else ids[0]
chosen = st.selectbox("Choose a topic", ids, index=ids.index(default_id),
                      format_func=lambda i: labels[i], label_visibility="collapsed")
row = themes[themes.theme_id == chosen].iloc[0]

mood = ("Unhappy" if row.avg_rating <= 2.5
        else "Mixed" if row.avg_rating < 4 else "Happy")
pill = "pill-neg" if row.avg_rating <= 2.5 else "pill-pos"
st.markdown(
    f"<div class='card'><span class='pill {pill}'>{mood}</span>"
    f"<div style='margin-top:12px;font-size:1.05rem;color:{design.INK};'>"
    f"<strong>{row.n_rows:,} reviews</strong> &nbsp;·&nbsp; "
    f"{row.avg_rating:.1f} stars on average</div>"
    f"<div style='margin-top:10px;color:{design.INK_2};font-size:.92rem;'>"
    f"Words that set this topic apart: {row.top_terms}</div>"
    f"<div style='margin-top:14px;padding-left:14px;"
    f"border-left:3px solid {design.AXIS};color:{design.INK_2};"
    f"font-size:.95rem;line-height:1.55;'>"
    f"“{str(row.example)[:400]}”</div>"
    f"<div style='margin-top:8px;color:{design.MUTED};font-size:.78rem;'>"
    f"The review closest to the centre of this topic</div></div>",
    unsafe_allow_html=True)

order = st.radio("Sort reviews by",
                 ["Most typical", "Newest", "Lowest rated"],
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

with st.expander("How topics are found, and what they miss"):
    st.markdown(f"""
Every review is turned into a list of numbers that represents its meaning, then
reviews with similar numbers are grouped together. Nobody wrote the topic list in
advance — the {len(themes)} topics were discovered from the reviews themselves.

**Two things worth knowing.**

Around a third of reviews are not placed in any topic. That is deliberate: a
review saying only "good app" is not about anything in particular, and forcing it
into a topic would corrupt that topic.

Several of the largest topics are simply praise — "good", "nice", "excellent".
That is honest rather than useful: a fifth of all reviews carry no specific
feedback at all. One large topic groups reviews written in Hindi and Hinglish
together regardless of subject, because the model recognises the language but not
what is being said in it. That is a real limitation of this system.
""")
