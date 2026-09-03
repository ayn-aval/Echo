"""Trends — what is getting worse, and when it started."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import design
import pandas as pd
import plotly.graph_objects as go
from shared import MODEL, sql, st

design.setup("Trends")
design.header(
    "What changed",
    "Compare two periods to see which complaints grew and which faded. Set the "
    "dividing line to the week a new app version shipped to see what it broke.")

series = sql("""
    SELECT date_trunc('week', r.reviewed_at)::date AS week,
           rt.theme_id, coalesce(t.display_name, t.label) AS label,
           t.avg_rating, r.review_version AS version,
           count(*) AS reviews
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
      JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
     WHERE rt.model = %s AND rt.theme_id >= 0
     GROUP BY 1, 2, 3, 4, 5""", (MODEL,))

if series.empty:
    st.error("No topics yet. Run: python -m src.clustering.name_themes")
    st.stop()

versions = sql("""SELECT review_version AS v, count(*) AS n FROM reviews
                   WHERE app='swiggy' AND review_version IS NOT NULL
                   GROUP BY 1 HAVING count(*) >= 1000 ORDER BY 2 DESC""")

weeks = sorted(series.week.unique())
c1, c2, c3 = st.columns([3, 2, 3], gap="medium")
with c1:
    start, end = st.select_slider("Period covered", options=weeks,
                                  value=(weeks[0], weeks[-1]))
with c2:
    only_complaints = st.selectbox("Show", ["Complaints only", "All topics"])
with c3:
    picked = st.multiselect("App version", versions.v.tolist(),
                            placeholder="All versions")

view = series[(series.week >= start) & (series.week <= end)]
if picked:
    view = view[view.version.isin(picked)]
if only_complaints == "Complaints only":
    view = view[view.avg_rating <= 2.5]
if view.empty:
    st.warning("No reviews match these filters. Widen the period or clear the "
               "version filter.")
    st.stop()

inner = [w for w in weeks if start < w < end]
split = st.select_slider("Dividing line", options=inner or weeks,
                         value=(inner or weeks)[len(inner or weeks) // 2])

before, after = view[view.week < split], view[view.week >= split]
if before.empty or after.empty:
    st.info("Move the dividing line inside the period to compare.")
    st.stop()


def share(df):
    g = df.groupby(["theme_id", "label"], as_index=False).reviews.sum()
    g["pct"] = g.reviews / g.reviews.sum() * 100
    return g


m = share(before).merge(share(after), on=["theme_id", "label"], how="outer",
                        suffixes=("_before", "_after")).fillna(0)
m["change"] = m.pct_after - m.pct_before
m = m[(m.reviews_before + m.reviews_after) >= 30]
worse, better = m.nlargest(6, "change"), m.nsmallest(6, "change")

top = worse.iloc[0]
growth = ((top.reviews_after - top.reviews_before) / max(top.reviews_before, 1)) * 100
design.tiles([
    ("Fastest growing complaint", top.label,
     f"{top.reviews_before:,.0f} to {top.reviews_after:,.0f} reviews, "
     f"up {growth:.0f}%"),
    ("Topics growing", f"{(m.change > 0).sum()}", "a bigger share than before"),
    ("Topics shrinking", f"{(m.change < 0).sum()}", "a smaller share than before"),
])

st.markdown(f"## Before and after {split:%d %B %Y}")
design.note("Ranked by how much of the conversation each topic takes up, not by "
            "raw counts — more people review in some weeks than others, so a "
            "topic can gain reviews without becoming a bigger problem.")

lc, rc = st.columns(2, gap="large")
for col, data, title, colour in ((lc, worse, "Growing", design.NEGATIVE),
                                 (rc, better, "Fading", design.POSITIVE)):
    with col:
        st.markdown(f"### {title}")
        d = data.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=d.change.abs(), y=d.label, orientation="h",
            marker_color=colour, marker_line_width=0,
            customdata=d[["reviews_before", "reviews_after"]],
            hovertemplate="%{y}<br>%{customdata[0]:,.0f} to %{customdata[1]:,.0f}"
                          " reviews<extra></extra>"))
        fig.update_traces(marker_cornerradius=4)
        st.plotly_chart(design.style(fig, height=34 * len(d) + 80,
                                     xlab="Change in share of reviews"),
                        width="stretch", config={"displayModeBar": False})

st.markdown("## Track a topic week by week")

options = (view.groupby(["theme_id", "label"], as_index=False).reviews.sum()
               .nlargest(25, "reviews").label.tolist())
default = [t for t in worse.label.head(2) if t in options] or options[:2]
chosen = st.multiselect("Topics", options, default=default,
                        label_visibility="collapsed")

if chosen:
    plot = (view[view.label.isin(chosen)]
            .groupby(["week", "label"], as_index=False).reviews.sum())
    fig = go.Figure()
    for i, name in enumerate(chosen):
        d = plot[plot.label == name]
        fig.add_trace(go.Scatter(
            x=d.week, y=d.reviews, name=name, mode="lines",
            line=dict(color=design.SERIES[i % len(design.SERIES)], width=2),
            hovertemplate="%{x|%d %b}<br>%{y:,} reviews<extra>" + name + "</extra>"))
    fig.add_vline(x=split, line_width=1, line_color=design.AXIS)
    st.plotly_chart(design.style(fig, height=380, legend=len(chosen) > 1,
                                 ylab="Reviews that week"),
                    width="stretch", config={"displayModeBar": False})
    design.note("The vertical line marks the dividing line set above.")
else:
    st.info("Choose at least one topic to plot.")

with st.expander("How to read this page"):
    st.markdown("""
**Why share and not counts.** If review volume doubles one month, every topic
gains reviews. Share answers the question you actually care about — is this a
bigger part of what people are unhappy about than it used to be?

**Why some versions are missing.** The version filter lists only versions with at
least 1,000 reviews. Most of the 285 versions in the data have a single review
from someone running a very old build, and 10,925 reviews record no version at
all.

**Small topics are excluded.** A topic needs at least 30 reviews across both
periods to appear, because a jump from 2 reviews to 6 is noise, not a trend.
""")
