"""Alerts — weeks where a topic broke out of its own normal range."""

import design
import plotly.graph_objects as go
import streamlit as st
from shared import sql

F = st.session_state["filters"]

design.appbar("Monitor", "Alerts")

alerts = sql(f"""
    SELECT a.week_start, a.theme_id, a.kind, a.reviews, a.baseline_mean,
           a.baseline_sd, a.z, a.share_z, a.threshold, a.baseline_weeks,
           coalesce(t.display_name, t.label) AS name,
           coalesce(t.category, 'Other') AS area
      FROM theme_alerts a
      JOIN themes t ON t.model = a.model AND t.theme_id = a.theme_id
     WHERE a.model = 'sbert-domain' {F.area_clause('t')}
     ORDER BY a.week_start DESC, a.z DESC NULLS LAST""")

if alerts.empty:
    st.info("No alerts in this selection. Run `python -m src.analytics.alerts` "
            "if none have been generated yet.")
    st.stop()

# A week with several topics firing at once is usually one incident, not several
# problems. Worth seeing as a week rather than as a list of rows.
by_week = alerts.groupby("week_start").size().sort_values(ascending=False)
busiest, busiest_n = by_week.index[0], int(by_week.iloc[0])

design.tiles([
    ("Alerts", f"{len(alerts)}", f"at z ≥ {alerts.threshold.iloc[0]:.1f}"),
    ("Weeks affected", f"{alerts.week_start.nunique()}", "of 23 weeks tested"),
    ("Busiest week", f"{busiest:%d %b}",
     f"{busiest_n} topics fired together"
     + (" — likely one incident" if busiest_n >= 3 else "")),
])

if busiest_n >= 3:
    st.markdown(
        f"<div class='card' style='border-left:3px solid {design.CRITICAL};'>"
        f"<h4>Week of {busiest:%d %B %Y}</h4>"
        f"<div style='color:{design.INK_2};font-size:.9rem;line-height:1.55;'>"
        f"{busiest_n} topics broke out of their normal range in the same week. "
        f"When several fire together it is usually one underlying event rather "
        f"than {busiest_n} separate problems.</div></div>",
        unsafe_allow_html=True)

st.markdown("## Every alert")

for r in alerts.itertuples():
    if r.kind == "new":
        headline = (f"{r.reviews} reviews, none at all in the previous "
                    f"{r.baseline_weeks} weeks")
        accent, word = design.CRITICAL, "New"
    else:
        multiple = r.reviews / float(r.baseline_mean) if r.baseline_mean else 0
        headline = (f"{r.reviews} reviews against a normal "
                    f"{r.baseline_mean:.0f} ± {r.baseline_sd:.0f} — "
                    f"{multiple:.1f}× its usual week")
        accent = design.CRITICAL if r.z >= 6 else design.WARNING
        word = "Spike"

    volume = ("" if r.kind == "new" or (r.share_z is not None and r.share_z >= 2)
              else "<div class='meta' style='color:#8f5613;'>Share of all reviews "
                   "did not move — this week simply had more reviews overall."
                   "</div>")

    st.markdown(
        f"<div class='issue' style='--accent:{accent};'>"
        f"<div class='rank' style='font-size:.95rem;font-weight:640;'>"
        f"{r.week_start:%d %b}</div><div class='body'>"
        f"<div class='name'>{r.name}{design.badge(word, 'crit' if word == 'New' else 'warn')}"
        f"<span class='area-tag'>{r.area}</span></div>"
        f"<div class='meta'>{headline}"
        + (f" &nbsp;·&nbsp; z = {r.z:.1f}" if r.z is not None else "") + "</div>"
        + volume + "</div></div>", unsafe_allow_html=True)

st.markdown("## How a topic is flagged")

pick = st.selectbox("Topic", alerts.name.unique(), label_visibility="collapsed")
row = alerts[alerts.name == pick].iloc[0]
series = sql("""SELECT week_start, reviews FROM theme_weekly
                 WHERE model='sbert-domain' AND theme_id=%s
                 ORDER BY week_start""", (int(row.theme_id),))

fig = go.Figure()
if row.baseline_mean:
    mean, sd = float(row.baseline_mean), float(row.baseline_sd)
    fig.add_hrect(y0=mean - sd, y1=mean + sd, line_width=0,
                  fillcolor="rgba(42,120,214,0.10)")
    fig.add_hline(y=mean, line_width=1, line_dash="solid",
                  line_color=design.BLUE)
    fig.add_hline(y=mean + row.threshold * sd, line_width=1,
                  line_color=design.CRITICAL)
fig.add_trace(go.Scatter(
    x=series.week_start, y=series.reviews, mode="lines+markers",
    line=dict(color=design.INK_2, width=1.8),
    marker=dict(size=6, color=design.INK_2),
    hovertemplate="Week of %{x|%d %b}<br>%{y} reviews<extra></extra>"))
fig.add_trace(go.Scatter(
    x=[row.week_start], y=[row.reviews], mode="markers",
    marker=dict(size=13, color=design.CRITICAL,
                line=dict(color="#ffffff", width=2)),
    hovertemplate="Flagged: %{y} reviews<extra></extra>"))
st.plotly_chart(design.style(fig, height=300, ylab="Reviews per week"),
                width="stretch", config={"displayModeBar": False})
design.note("Blue band is one standard deviation around the topic's normal week. "
            "Red line is the alerting threshold. The large dot is the flagged week.")

with st.expander("What this rule cannot catch"):
    st.markdown("""
**Slow burn.** A topic growing 10% a week pulls its own average up with it and
never crosses the line. This catches jumps; the **What changed** screen catches
trends.

**Repeat incidents.** A problem lasting two weeks raises the baseline afterwards,
making the same problem harder to flag next time.

**Busy weeks.** A festival or an outage lifts every topic at once. The share
column is the check — if a topic's share of all reviews did not move, the week
was simply busier.

**How much of this list is noise.** 110 topics are tested every week. At
z ≥ 3 roughly 0.15 alerts per week are expected from chance alone, against the
rate actually observed. At z ≥ 2 the expected false-alarm count exceeds the
number of alerts found, which is why the threshold is 3.
""")
