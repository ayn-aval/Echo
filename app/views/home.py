"""Today — the screen an operator opens first: what needs attention, and why."""

import design
import insights
import plotly.graph_objects as go
import streamlit as st
from shared import sql

h = insights.headline_health()

design.appbar(
    "Monitor", "Today",
    "The complaints most worth acting on right now, ranked by how many customers "
    "raised them, how unhappy they were, and whether the problem is growing.",
    right=f"Latest review <b>{h['latest']:%d %b %Y}</b>")

delta = h["rating_delta"]
direction = ("b-good", f"up {delta:+.2f}") if delta > 0.02 else \
            ("b-crit", f"down {delta:+.2f}") if delta < -0.02 else \
            ("b-flat", "unchanged")

design.tiles([
    ("Rating, last 28 days", f"{h['rating_28d']:.2f}",
     f"out of 5 · {direction[1]} on the 28 days before"),
    ("Unhappy customers", f"{h['unhappy'] / h['reviews'] * 100:.0f}%",
     f"{h['unhappy']:,} people rated 1 or 2 stars"),
    ("Reviews analysed", f"{h['reviews']:,}", "every review, grouped by subject"),
])

st.markdown("## Needs attention")
design.note("Ranked by number of customers affected, how low they rated the app, "
            "and whether the problem is growing. Open any item to read the "
            "reviews behind it.")

top = insights.priorities(limit=5)
if top.empty:
    st.info("No complaint topics found yet.")
else:
    LABEL = {"crit": ("Urgent", "crit", design.CRITICAL),
             "warn": ("Watch", "warn", design.WARNING),
             "flat": ("Steady", "flat", design.MUTED)}
    for i, r in enumerate(top.itertuples(), 1):
        word, cls, accent = LABEL[r.urgency]
        quote = " ".join(str(r.content).split())[:190]
        st.markdown(
            f"<div class='issue' style='--accent:{accent};'>"
            f"<div class='rank'>{i}</div><div class='body'>"
            f"<div class='name'>{r.name} &nbsp; {design.badge(word, cls)}</div>"
            f"<div class='meta'>{r.why}</div>"
            f"<div class='quote'>“{quote}…”</div>"
            f"</div></div>", unsafe_allow_html=True)

    st.page_link("views/issues.py", label="See all topics and read the reviews")

st.markdown("## Complaints over the last three months")

weekly = sql("""
    SELECT date_trunc('week', r.reviewed_at)::date AS week,
           count(*) FILTER (WHERE r.score <= 2) AS unhappy,
           count(*) AS total
      FROM reviews r
     WHERE r.app = 'swiggy'
       AND r.reviewed_at >= (SELECT max(reviewed_at) FROM reviews
                              WHERE app='swiggy') - interval '90 days'
     GROUP BY 1 ORDER BY 1""")
weekly["pct"] = weekly.unhappy / weekly.total * 100

fig = go.Figure(go.Scatter(
    x=weekly.week, y=weekly.pct, mode="lines+markers",
    line=dict(color=design.CRITICAL, width=2),
    marker=dict(size=7, color=design.CRITICAL),
    hovertemplate="Week of %{x|%d %b}<br>%{y:.0f}% rated 1 or 2 stars"
                  "<extra></extra>"))
st.plotly_chart(design.style(fig, height=260, ylab="% rating 1 or 2 stars"),
                width="stretch", config={"displayModeBar": False})
design.note("Share of reviews each week rating the app 1 or 2 stars. Rising means "
            "a larger proportion of customers are unhappy, regardless of how many "
            "reviewed.")

with st.expander("How this ranking is calculated"):
    st.markdown(f"""
Each complaint topic is scored on three things, and the score is shown so it can
be argued with:

**How many customers** raised it in the last {insights.WINDOW_WEEKS} weeks, as a
share of all complaints in that period. Share rather than a raw count, so a busy
month does not promote everything at once.

**How unhappy they were**, taken from the average star rating of the reviews in
that topic. A topic averaging 1 star weighs roughly twice one averaging 2.

**Whether it is growing**, comparing that share against the
{insights.WINDOW_WEEKS} weeks before. Growth only ever raises a score — a large,
long-standing complaint still needs fixing when it is flat.

**Two kinds of topic are deliberately excluded** from this list. Some reviews are
angry without saying why, and one topic groups reviews by the language they are
written in rather than their subject. Both are real and both are shown under
*What customers raise*, but neither is something a team can act on, so they are
kept out of a list that reads as a recommendation.
""")
