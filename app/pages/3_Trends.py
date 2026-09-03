"""Trends — what got worse, and when.

Built around one question a product manager actually asks: *what got worse after
the last release?* A chart of lines does not answer that; a ranked list of which
themes grew does. So the movers table is the centrepiece and the chart supports it.

Two measurement decisions worth stating on the page rather than burying:

  * Movers are ranked by **share of reviews**, not raw count. Overall volume
    varies week to week, so a theme can gain reviews purely because more people
    reviewed. Share answers "is this a bigger part of what people complain
    about", which is the question being asked.
  * The app-version filter offers only versions with at least 1,000 reviews — 24
    of the 285 in the data, covering 90.7% of reviews. Most versions have a
    single review because someone is still running a two-year-old build, and a
    dropdown of 285 is unusable. 10,925 reviews carry no version at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared
from shared import MODEL, sql, st

import pandas as pd
import plotly.express as px

shared.page("Trends", "📈", "Theme volume over time — and what changed.")

series = sql("""
    SELECT date_trunc('week', r.reviewed_at)::date AS week,
           rt.theme_id, t.label, r.review_version AS version,
           count(*) AS reviews
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
      JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
     WHERE rt.model = %s AND rt.theme_id >= 0
     GROUP BY 1, 2, 3, 4""", (MODEL,))

if series.empty:
    st.error("No themes yet. Run `python -m src.clustering.name_themes`.")
    st.stop()

big_versions = sql("""
    SELECT review_version AS version, count(*) AS n
      FROM reviews WHERE app='swiggy' AND review_version IS NOT NULL
     GROUP BY 1 HAVING count(*) >= 1000 ORDER BY 2 DESC""")

f1, f2 = st.columns([2, 3])
with f1:
    weeks = sorted(series.week.unique())
    start, end = st.select_slider("Date range", options=weeks,
                                  value=(weeks[0], weeks[-1]))
with f2:
    picked_versions = st.multiselect(
        "App versions", big_versions.version.tolist(),
        help=f"{len(big_versions)} versions with 1,000+ reviews, covering "
             f"{big_versions.n.sum():,} of 89,075 versioned reviews. "
             "Leave empty for all versions.")

view = series[(series.week >= start) & (series.week <= end)]
if picked_versions:
    view = view[view.version.isin(picked_versions)]
if view.empty:
    st.warning("No reviews match these filters.")
    st.stop()

total_by_week = view.groupby("week", as_index=False).reviews.sum()

st.divider()
st.subheader("What changed")

mid = weeks[len(weeks) // 2]
split = st.select_slider(
    "Compare reviews before and after", options=weeks,
    value=mid if start <= mid <= end else weeks[len(weeks) // 2],
    help="Set this to the week a release shipped to see what moved after it.")

before, after = view[view.week < split], view[view.week >= split]
if before.empty or after.empty:
    st.info("Move the split point inside the date range to compare.")
    st.stop()


def share(df):
    g = df.groupby(["theme_id", "label"], as_index=False).reviews.sum()
    g["share"] = g.reviews / g.reviews.sum() * 100
    return g


b, a = share(before), share(after)
movers = b.merge(a, on=["theme_id", "label"], how="outer",
                 suffixes=("_before", "_after")).fillna(0)
movers["share_change"] = movers.share_after - movers.share_before
movers["reviews_change"] = movers.reviews_after - movers.reviews_before
movers = movers[(movers.reviews_before + movers.reviews_after) >= 30]

worse = movers.nlargest(8, "share_change")
better = movers.nsmallest(8, "share_change")

left, right = st.columns(2)
with left:
    st.markdown(f"**Grew after {split}** — a bigger share of what people say")
    st.dataframe(
        worse[["label", "reviews_before", "reviews_after", "share_change"]],
        hide_index=True, width="stretch",
        column_config={"label": "Theme", "reviews_before": "Before",
                       "reviews_after": "After",
                       "share_change": st.column_config.NumberColumn(
                           "Share change", format="%+.2f pp")})
with right:
    st.markdown(f"**Shrank after {split}**")
    st.dataframe(
        better[["label", "reviews_before", "reviews_after", "share_change"]],
        hide_index=True, width="stretch",
        column_config={"label": "Theme", "reviews_before": "Before",
                       "reviews_after": "After",
                       "share_change": st.column_config.NumberColumn(
                           "Share change", format="%+.2f pp")})

st.caption("Ranked by change in **share of reviews**, in percentage points, not "
           "raw count — overall volume moves week to week, so a theme can gain "
           "reviews without becoming more of a problem. Themes with fewer than "
           "30 reviews across both periods are excluded as too noisy to rank.")

st.divider()
st.subheader("Theme volume over time")

top_themes = (view.groupby(["theme_id", "label"], as_index=False).reviews.sum()
                  .nlargest(30, "reviews"))
default = worse.label.head(3).tolist() or top_themes.label.head(3).tolist()
chosen = st.multiselect("Themes to plot", top_themes.label.tolist(),
                        default=default)

if chosen:
    plot = (view[view.label.isin(chosen)]
            .groupby(["week", "label"], as_index=False).reviews.sum())
    as_share = st.checkbox("Show as share of all reviews that week", value=False)
    if as_share:
        plot = plot.merge(total_by_week, on="week", suffixes=("", "_total"))
        plot["reviews"] = plot.reviews / plot.reviews_total * 100
    fig = px.line(plot, x="week", y="reviews", color="label", markers=True,
                  labels={"week": "", "label": "Theme",
                          "reviews": "% of reviews" if as_share else "Reviews"})
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Pick at least one theme to plot.")

st.caption(f"Weekly buckets · {view.reviews.sum():,} reviews in the current "
           f"filter · themes from `{MODEL}`.")
