"""Explore — what customers talk about, and what any one topic looks like.

This replaces three screens. Understanding a single topic used to mean picking it
from a 110-item dropdown on Topics, then finding it again in a different
multiselect on Trends to see whether it was growing. Here you click its bar once
and get its history and its reviews in place.

Search sits at the top of the same screen because it answers the same question —
"what are people saying about X" — for an X that is not one of the topics.
"""

import design
import plotly.graph_objects as go
import shared
from shared import MODEL, sql, st

design.appbar("Explore", "What customers say")

F = st.session_state["filters"]
TOPIC = "topic"

query = st.text_input("Search", placeholder="Search reviews by meaning — "
                      "try: my refund never arrived", label_visibility="collapsed")

# ── search, when there is a query ───────────────────────────────────────────
if query:
    shared.get_search()
    from src.search.query import search
    hits = search(query, k=25)
    best = hits.iloc[0]
    design.hero(
        eyebrow=f"Closest match for “{query}”",
        headline=f"“{' '.join(str(best.content).split())[:150]}”",
        value=f"{len(hits)}",
        unit="matching reviews",
        side="Reviews that <b>mean</b> the same thing,<br>even in different words.")
    st.dataframe(
        hits[["rank", "score", "content"]], hide_index=True, width="stretch",
        height=460,
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "score": st.column_config.ProgressColumn("How close", min_value=0.0,
                                                     max_value=1.0),
            "content": "Review"})
    st.stop()

# ── otherwise, the topics ───────────────────────────────────────────────────
themes = sql(f"""
    SELECT t.theme_id, coalesce(t.display_name, t.label) AS label,
           coalesce(t.category, 'Other') AS area, t.n_rows, t.avg_rating
      FROM themes t
     WHERE t.model = %s AND t.actionable {F.area_clause('t')}
     ORDER BY t.n_rows DESC""", (MODEL,))

if themes.empty:
    st.error("No topics yet. Run: python -m src.clustering.name_themes")
    st.stop()

complaints = themes[themes.avg_rating <= 2.5]
chosen_id = st.session_state.get(TOPIC)
row = themes[themes.theme_id == chosen_id]
row = row.iloc[0] if not row.empty else None

if row is None:
    biggest = complaints.iloc[0] if not complaints.empty else themes.iloc[0]
    design.hero(
        eyebrow="Found automatically from what people wrote",
        headline=f"The biggest complaint is “{biggest.label.lower()}”",
        value=f"{biggest.n_rows:,}",
        unit="customers said it",
        side=(f"<b>{len(themes)}</b> subjects in all<br>"
              f"<b>{len(complaints)}</b> of them complaints"))
else:
    design.hero(
        eyebrow=row.area,
        headline=row.label,
        value=f"{row.n_rows:,}",
        unit="customers said it",
        side=f"Rated <b>{row.avg_rating:.1f}</b> out of 5")

    week = sql("""SELECT week_start, reviews FROM theme_weekly
                   WHERE model=%s AND theme_id=%s ORDER BY week_start""",
               (MODEL, int(row.theme_id)))
    if len(week) > 2:
        fig = go.Figure(go.Scatter(
            x=week.week_start, y=week.reviews, mode="lines",
            line=dict(color=design.CRITICAL if row.avg_rating <= 2.5
                      else design.BLUE, width=2.2),
            fill="tozeroy", fillcolor="rgba(208,59,59,0.07)",
            hovertemplate="Week of %{x|%d %b}: %{y} reviews<extra></extra>"))
        st.plotly_chart(design.style(fig, height=240, ylab="Reviews that week"),
                        width="stretch", config={"displayModeBar": False})

    # Most-typical alone returns the shortest reviews — nine rows of "worst
    # service" that prove the topic exists and say nothing about it. A hard word
    # floor is worse: it left 29 of this theme's 713. So take the 300 most typical
    # and show the ones with the most to say, which keeps the list both
    # representative and worth reading whatever the theme's size.
    reviews = sql("""
        SELECT stars, date, review FROM (
            SELECT r.score AS stars, r.reviewed_at::date AS date,
                   r.content AS review, r.word_count
              FROM review_themes rt
              JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
             WHERE rt.model = %s AND rt.theme_id = %s
             ORDER BY rt.strength DESC LIMIT 300) typical
         ORDER BY word_count DESC LIMIT 100""", (MODEL, int(row.theme_id)))
    st.dataframe(reviews, hide_index=True, width="stretch", height=340,
                 column_config={
                     "stars": st.column_config.NumberColumn("Stars", width="small"),
                     "date": st.column_config.DateColumn("Date", width="small"),
                     "review": "Review"})

    if st.button("Back to all topics"):
        st.session_state[TOPIC] = None
        st.rerun()

st.markdown("## Most talked about")
design.note("Click a bar to open it.")

view = st.radio("Show", ["Complaints", "Praise", "Everything"],
                horizontal=True, label_visibility="collapsed")
subset = {"Complaints": complaints,
          "Praise": themes[themes.avg_rating >= 4.0],
          "Everything": themes}[view].nlargest(12, "n_rows")

clicked = design.click_bars(
    [(r.label, int(r.n_rows),
      design.NEGATIVE if r.avg_rating <= 2.5 else
      design.NEUTRAL if r.avg_rating < 4 else design.POSITIVE)
     for r in subset.itertuples()],
    key=f"topic_bars_{view}",
    selected=(row.label if row is not None else None), xlab="Reviews")

if clicked:
    match = themes[themes.label == clicked]
    if not match.empty and int(match.theme_id.iloc[0]) != chosen_id:
        st.session_state[TOPIC] = int(match.theme_id.iloc[0])
        st.rerun()
