"""Explore complaints — search in plain English, or browse every problem.

Search and topics live on one screen because they answer the same question,
"what are people saying about X", for an X that either is or is not one of the
discovered problems. Clicking a problem opens it in place: its week-by-week
history and its actual reviews, without a dropdown anywhere.
"""

import html

import design
import plotly.graph_objects as go
import shared
import streamlit as st
from shared import MODEL, sql

K = 25              # how many reviews a search returns
TOPIC = "topic"


def stars(n):
    """"1 star", not "1 stars"."""
    n = int(n)
    return f"{n} star" if n == 1 else f"{n} stars"


def clean(text, limit=None):
    out = " ".join(str(text).split())
    if limit and len(out) > limit:
        out = out[:limit].rstrip() + "…"
    return html.escape(out)


EXAMPLES = ["refund never came back to my account",
            "driver was rude to me",
            "the app is unusable on my old phone"]

# The example buttons need to be able to fill the box, and Streamlit forbids
# assigning to a key that a widget already owns — st.session_state.query cannot
# be modified after the widget with key "query" is instantiated. So the seed
# lives under its own key and is passed as `value`, and the input is given no
# key of its own: Streamlit then derives the widget's identity from its
# arguments, so a new seed produces a new widget carrying the new text, while
# anything typed survives reruns where the seed has not changed.
st.session_state.setdefault("seed", "")

st.markdown("# Ask about anything customers mention")
design.sub("Type it how a customer would say it. You do not have to guess their "
           "exact words — “driver was rude” also finds “rider shouted at me”.")

query = st.text_input("Search", value=st.session_state["seed"],
                      placeholder="e.g. refund never came back to my account",
                      label_visibility="collapsed")

design.sub("Try:")
for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    with col:
        if st.button(example, key=f"ex_{example}"):
            st.session_state["seed"] = example
            st.rerun()

# ── search ──────────────────────────────────────────────────────────────────
if query:
    shared.get_search()
    from src.search.query import search

    hits = search(query, k=K)
    # The headline must not restate K. len(hits) is always 25 because that is
    # what was asked for, and summing n_rows is nearly the same — matched texts
    # are almost always distinct. How many of the closest reviews came from
    # unhappy customers is a real answer to "is this a sore point".
    meta = sql("""SELECT review_id, score AS stars, reviewed_at::date AS date
                    FROM reviews WHERE app = 'swiggy' AND review_id = ANY(%s)""",
               (list(hits.review_id),))
    topic = sql("""SELECT rt.review_id,
                          coalesce(t.display_name, t.label) AS topic
                     FROM review_themes rt
                     JOIN themes t ON t.model = rt.model
                                  AND t.theme_id = rt.theme_id
                    WHERE rt.model = %s AND rt.review_id = ANY(%s)""",
                (MODEL, list(hits.review_id)))
    hits = hits.merge(meta, on="review_id", how="left") \
               .merge(topic, on="review_id", how="left")
    unhappy = int((hits.stars <= 2).sum())

    st.markdown(f"## {unhappy} of the {len(hits)} closest reviews are unhappy "
                "customers"
                f'<span style="font-size:13px;font-weight:400;'
                f'color:{design.MUTED};margin-left:10px">'
                f"averaging {hits.stars.mean():.1f} stars</span>",
                unsafe_allow_html=True)
    design.sub("Closest in meaning first. Nothing was matched on keywords, so "
               "reviews saying this in other words are here too.")

    for r in hits.itertuples():
        tag = (f'<span class="tag">{clean(r.topic)}</span>'
               if isinstance(r.topic, str) else "")
        st.markdown(
            '<div class="review"><div class="review-meta">'
            f'<b style="font-weight:800;color:{design.INK}">{stars(r.stars)}'
            f"</b>{tag}"
            f'<span style="margin-left:auto">{r.date:%d %b %Y}</span></div>'
            f'<div style="font-size:15px;line-height:1.5">{clean(r.content)}'
            "</div></div>", unsafe_allow_html=True)
    st.stop()

# ── otherwise, the problems ─────────────────────────────────────────────────
F = st.session_state["filters"]
themes = sql(f"""
    SELECT t.theme_id, coalesce(t.display_name, t.label) AS label,
           coalesce(t.category, 'Other') AS area, t.n_rows, t.avg_rating
      FROM themes t
     WHERE t.model = %s AND t.actionable {F.area_clause('t')}
     ORDER BY t.n_rows DESC""", (MODEL,))

if themes.empty:
    st.error("No problems yet. Run: python -m src.clustering.name_themes")
    st.stop()

complaints = themes[themes.avg_rating <= 2.5]
chosen = st.session_state.get(TOPIC)
row = themes[themes.theme_id == chosen]
row = row.iloc[0] if not row.empty else None

# ── one problem, opened ─────────────────────────────────────────────────────
if row is not None:
    design.rule()
    design.kicker(row.area)
    st.markdown(f"## {clean(row.label)}")
    design.sub(f"{row.n_rows:,} reviews, averaging {row.avg_rating:.1f} stars.")

    week = sql("""SELECT week_start, reviews FROM theme_weekly
                   WHERE model=%s AND theme_id=%s ORDER BY week_start""",
               (MODEL, int(row.theme_id)))
    if len(week) > 2:
        st.markdown("### How often this comes up, week by week")
        st.plotly_chart(
            design.line(week.week_start, week.reviews,
                        "Week of %{x|%d %b}: %{y} reviews<extra></extra>",
                        height=220, ylab="Reviews that week", fill=True),
            width="stretch", config={"displayModeBar": False})

    st.markdown("### What customers actually wrote")
    # Most-typical alone returns the shortest reviews — nine rows of "worst
    # service" that prove the topic exists and say nothing about it. A hard word
    # floor is worse: it left 29 of this theme's 713. So take the 300 most
    # typical and show the ones with the most to say.
    reviews = sql("""
        SELECT stars, date, review FROM (
            SELECT r.score AS stars, r.reviewed_at::date AS date,
                   r.content AS review, r.word_count
              FROM review_themes rt
              JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
             WHERE rt.model = %s AND rt.theme_id = %s
             ORDER BY rt.strength DESC LIMIT 300) typical
         ORDER BY word_count DESC LIMIT 40""", (MODEL, int(row.theme_id)))
    for r in reviews.itertuples():
        st.markdown(
            '<div class="review"><div class="review-meta">'
            f'<b style="font-weight:800;color:{design.INK}">{stars(r.stars)}</b>'
            f'<span style="margin-left:auto">{r.date:%d %b %Y}</span></div>'
            f'<div style="font-size:15px;line-height:1.5">{clean(r.review)}'
            "</div></div>", unsafe_allow_html=True)

    if st.button("← Back to all problems"):
        st.session_state[TOPIC] = None
        st.rerun()
    st.stop()

# ── the list of problems ────────────────────────────────────────────────────
design.rule()
st.markdown(f"## Or browse the {len(themes)} problems")
design.sub("Each was found automatically by grouping reviews that mean the same "
           "thing — nobody wrote this list. Bars show the last 8 weeks.")

view = st.radio("Show", ["Complaints", "Praise", "Everything"], horizontal=True,
                label_visibility="collapsed")
subset = {"Complaints": complaints,
          "Praise": themes[themes.avg_rating >= 4.0],
          "Everything": themes}[view].nlargest(14, "n_rows")

history = sql("""
    SELECT theme_id, week_start, reviews FROM theme_weekly
     WHERE model = %s AND theme_id = ANY(%s)
     ORDER BY theme_id, week_start""",
    (MODEL, [int(i) for i in subset.theme_id]))

for r in subset.itertuples():
    # Trimmed in pandas rather than SQL. Dating back from max(week_start) gave a
    # ragged count — a theme missing a week returned fewer bars, and one with a
    # partial week returned more — so the caption said eight while the chart
    # showed thirteen. Taking the last eight rows per theme cannot drift.
    weeks = history[history.theme_id == r.theme_id].reviews.tolist()[-8:]
    line, opener = st.columns([8, 2], vertical_alignment="center")
    with line:
        design.topic_row(clean(r.label), weeks or [1], f"{int(r.n_rows):,}")
    with opener:
        if st.button("Open", key=f"t_{r.theme_id}"):
            st.session_state[TOPIC] = int(r.theme_id)
            st.rerun()
