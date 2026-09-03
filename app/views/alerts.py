"""Alerts — weeks where a topic suddenly got much busier than normal.

Everything a reader sees is a multiple of that topic's usual week ("3x its usual
week"), never a z-score. The z-score decides the flag and stays in the database
for auditing; a multiple is the same fact in language anyone can act on.
"""

import design
import plotly.graph_objects as go
import streamlit as st
from shared import sql

F = st.session_state["filters"]

design.appbar("Monitor", "Alerts")

alerts = sql(f"""
    SELECT a.week_start, a.theme_id, a.kind, a.reviews, a.baseline_mean,
           a.baseline_sd, a.z, a.share_z, a.baseline_weeks,
           coalesce(t.display_name, t.label) AS name,
           coalesce(t.category, 'Other') AS area
      FROM theme_alerts a
      JOIN themes t ON t.model = a.model AND t.theme_id = a.theme_id
     WHERE a.model = 'sbert-domain' AND t.actionable {F.area_clause('t')}
     ORDER BY a.week_start DESC, a.z DESC NULLS LAST""")

if alerts.empty:
    st.info("Nothing unusual in this selection.")
    st.stop()

by_week = alerts.groupby("week_start").size().sort_values(ascending=False)
busiest, busiest_n = by_week.index[0], int(by_week.iloc[0])

if busiest_n >= 3:
    headline = f"Something went wrong in the week of {busiest.day} {busiest:%B}"
    side = (f"<b>{busiest_n}</b> topics spiked in the same week, which usually "
            f"means one incident rather than {busiest_n}")
else:
    headline = "A few topics ran hotter than usual"
    latest = alerts.week_start.max()
    side = f"Most recent was the week of <b>{latest.day} {latest:%B}</b>"

design.hero(
    eyebrow="Unusual weeks",
    headline=headline,
    value=f"{len(alerts)}",
    unit="spikes found",
    side=side)

st.markdown("## What spiked, and when")

for r in alerts.itertuples():
    if r.kind == "new":
        headline_txt = f"<b>{r.reviews}</b> reviews — this topic was not there before"
        accent, word = design.CRITICAL, "New"
    else:
        times = r.reviews / float(r.baseline_mean) if r.baseline_mean else 0
        headline_txt = (f"<b>{r.reviews}</b> reviews that week, against a usual "
                        f"<b>{r.baseline_mean:.0f}</b>")
        accent = design.CRITICAL if times >= 2 else design.WARNING
        word = f"{times:.1f}x normal"

    caveat = ("" if r.kind == "new" or (r.share_z is not None and r.share_z >= 2)
              else "<div class='meta' style='color:#8f5613;'>"
                   "That week had more reviews overall</div>")

    st.markdown(
        f"<div class='issue' style='--accent:{accent};'>"
        f"<div class='rank' style='font-size:.88rem;font-weight:680;"
        f"min-width:52px;'>{r.week_start:%d %b}</div><div class='body'>"
        f"<div class='name'>{r.name}"
        f"{design.badge(word, 'crit' if r.kind == 'new' else 'warn')}"
        f"<span class='area-tag'>{r.area}</span></div>"
        f"<div class='meta'>{headline_txt}</div>{caveat}"
        f"</div></div>", unsafe_allow_html=True)

st.markdown("## See a topic's history")

pick = st.selectbox("Topic", alerts.name.unique(), label_visibility="collapsed")
row = alerts[alerts.name == pick].iloc[0]
history = sql("""SELECT week_start, reviews FROM theme_weekly
                  WHERE model='sbert-domain' AND theme_id=%s
                  ORDER BY week_start""", (int(row.theme_id),))

fig = go.Figure()
if row.baseline_mean:
    mean, sd = float(row.baseline_mean), float(row.baseline_sd)
    fig.add_hrect(y0=max(mean - sd, 0), y1=mean + sd, line_width=0,
                  fillcolor="rgba(42,120,214,0.11)",
                  annotation_text="A normal week", annotation_position="top left",
                  annotation_font=dict(color=design.BLUE, size=11))
fig.add_trace(go.Scatter(
    x=history.week_start, y=history.reviews, mode="lines",
    line=dict(color=design.INK_2, width=1.9),
    hovertemplate="Week of %{x|%d %b}: %{y} reviews<extra></extra>"))
fig.add_trace(go.Scatter(
    x=[row.week_start], y=[row.reviews], mode="markers+text",
    marker=dict(size=13, color=design.CRITICAL,
                line=dict(color="#ffffff", width=2)),
    text=["Spike"], textposition="top center",
    textfont=dict(color=design.CRITICAL, size=11),
    hovertemplate="%{y} reviews<extra></extra>"))
st.plotly_chart(design.style(fig, height=300, ylab="Reviews that week"),
                width="stretch", config={"displayModeBar": False})

design.note("A topic is flagged when a week lands far outside its own normal "
            "range. Roughly 1 in 5 of these will be a false alarm.")
