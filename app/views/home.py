"""Overview — the finding, what to do about it, and the health behind it.

Clicking a bar in "Unhappy customers by area" filters this whole screen to that
area: the headline restates and the priority list re-ranks. That click is what
replaces the business-area dropdown that used to sit in the sidebar on every
screen.
"""

import design
import filters
import insights
import plotly.graph_objects as go
import streamlit as st
from shared import ALL_REVIEWS, sql

F = st.session_state["filters"]
h = insights.headline_health()

design.appbar("What to fix", "Where you are losing customers",
              right=f"To <b>{h['latest']:%d %b %Y}</b> &nbsp;·&nbsp; {F.label}")

# ── every business area, now and in the period before ───────────────────────
areas = sql(f"""
    SELECT t.category AS area,
           count(*) AS reviews,
           round(avg(r.score)::numeric, 2) AS rating,
           count(*) FILTER (WHERE r.score <= 2) AS unhappy
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
      JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
     WHERE rt.model = 'sbert-domain' AND rt.theme_id >= 0 AND t.actionable
       AND t.category NOT IN ('General praise', 'Other')
       {F.where('r')}
     GROUP BY 1 ORDER BY count(*) FILTER (WHERE r.score <= 2) DESC""")

if areas.empty:
    st.info("No reviews in this period.")
    st.stop()

picked = filters.current_area()
focus = areas[areas.area == picked] if picked else areas
top = (focus.iloc[0] if not focus.empty else areas.iloc[0])

design.hero(
    eyebrow=f"{F.period} · {'selected area' if picked else 'biggest problem area'}",
    headline=f"{top.area} is where you are losing people",
    value=f"{top.unhappy:,}",
    unit="unhappy customers",
    side=(f"Rated <b>{top.rating:.1f}</b> out of 5<br>"
          f"across <b>{top.reviews:,}</b> reviews"))

# ── anything that spiked ────────────────────────────────────────────────────
alerts = sql(f"""
    SELECT a.week_start, a.reviews, a.baseline_mean, a.kind,
           coalesce(t.display_name, t.label) AS name
      FROM theme_alerts a
      JOIN themes t ON t.model = a.model AND t.theme_id = a.theme_id
     WHERE a.model = 'sbert-domain' AND t.actionable {F.area_clause('t')}
       AND a.week_start >= (SELECT max(week_start) FROM theme_alerts) - 28
     ORDER BY a.z DESC NULLS LAST LIMIT 3""")
if not alerts.empty:
    st.markdown("## Problems that spiked in the last 4 weeks")
    design.note("A topic that drew far more reviews in one week than it "
                "normally does. Three rising together is usually one incident, "
                "not three.")
    for r in alerts.itertuples():
        times = (r.reviews / float(r.baseline_mean)) if r.baseline_mean else 0
        detail = (f"<b>{r.reviews}</b> reviews that week against a usual "
                  f"<b>{r.baseline_mean:.0f}</b>" if r.kind != "new"
                  else f"<b>{r.reviews}</b> reviews — this topic was not there before")
        st.markdown(
            f"<div class='issue' style='--accent:{design.CRITICAL};'>"
            f"<div class='rank' style='font-size:.88rem;font-weight:680;"
            f"min-width:52px;'>{r.week_start:%d %b}</div><div class='body'>"
            f"<div class='name'>{r.name}"
            f"{design.badge(f'{times:.1f}x normal' if r.kind != 'new' else 'New', 'crit')}"
            f"</div><div class='meta'>{detail}</div></div></div>",
            unsafe_allow_html=True)

# ── what to do ──────────────────────────────────────────────────────────────
st.markdown("## Fix these first")
design.note("Ranked by how many customers raised it, how unhappy they were, "
            "and whether it is growing.")

top_issues = insights.priorities(limit=4, areas=F.areas, days=F.days)
if top_issues.empty:
    st.info("Nothing meets the threshold here.")
else:
    LABEL = {"crit": ("Urgent", design.CRITICAL), "warn": ("Watch", design.WARNING),
             "flat": ("Steady", design.MUTED)}
    for i, r in enumerate(top_issues.itertuples(), 1):
        word, accent = LABEL[r.urgency]
        st.markdown(
            f"<div class='issue' style='--accent:{accent};'>"
            f"<div class='rank'>{i}</div><div class='body'>"
            f"<div class='name'>{r.name}{design.badge(word, r.urgency)}"
            f"<span class='area-tag'>{r.area}</span></div>"
            f"<div class='meta'>{r.why}</div>"
            f"<div class='quote'>“{' '.join(str(r.content).split())[:170]}…”</div>"
            f"</div></div>", unsafe_allow_html=True)
    st.page_link("views/issues.py", label="See every topic")

# ── the clickable breakdown ─────────────────────────────────────────────────
st.markdown("## Which part of the business is worst")
design.note("Customers who gave 1 or 2 stars, counted by the part of the "
            "business their review is about. Click a bar to focus this whole "
            "screen on that area.")

clicked = design.click_bars(
    [(r.area, int(r.unhappy),
      design.BLUE if (not picked or r.area == picked) else design.MUTED)
     for r in areas.itertuples()],
    key="area_bars", selected=picked, xlab="Unhappy customers")

if clicked and clicked != picked:
    filters.set_area(clicked)
    st.rerun()
if picked:
    if st.button(f"Showing {picked} only — show all areas"):
        filters.set_area(None)
        st.rerun()

# ── the health behind it, for anyone who wants it ───────────────────────────
st.markdown("## How customers rate the app")
design.note(f"All {h['reviews']:,} reviews. These two are deliberately not "
            "affected by the time period in the menu.")

left, right = st.columns([2, 3], gap="large")
with left:
    st.markdown("### Stars people gave")
    stars = sql(f"""SELECT score, count(*) AS reviews FROM reviews
                     WHERE {ALL_REVIEWS} GROUP BY score ORDER BY score""")
    colours = [design.NEGATIVE if v <= 2 else design.NEUTRAL if v == 3
               else design.POSITIVE for v in stars.score]
    fig = go.Figure(go.Bar(
        x=[f"{v} star" if v == 1 else f"{v} stars" for v in stars.score],
        y=stars.reviews, marker_color=colours, marker_line_width=0, width=0.6,
        text=[f"{p:.0f}%" for p in stars.reviews / stars.reviews.sum() * 100],
        textposition="outside", textfont=dict(color=design.INK_2, size=12),
        hovertemplate="%{x}<br>%{y:,} reviews<extra></extra>"))
    fig.update_traces(marker_cornerradius=5)
    st.plotly_chart(design.style(fig, height=300, ylab="Reviews"),
                    width="stretch", config={"displayModeBar": False})

with right:
    st.markdown("### Average rating over time")
    daily = sql(f"""SELECT reviewed_at::date AS day, avg(score) AS rating
                      FROM reviews WHERE {ALL_REVIEWS} GROUP BY 1 ORDER BY 1""")
    daily["smooth"] = daily.rating.rolling(14, min_periods=1).mean()
    fig = go.Figure(go.Scatter(
        x=daily.day, y=daily.smooth, mode="lines",
        line=dict(color=design.BLUE, width=2.2),
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} stars<extra></extra>"))
    fig.add_hline(y=3, line_width=1, line_color=design.AXIS)
    st.plotly_chart(design.style(fig, height=300, ylab="Average stars"),
                    width="stretch", config={"displayModeBar": False})
