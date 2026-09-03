"""Overview — how much data there is and what it looks like.

Every figure on this page comes from all 100,000 reviews, NOT the 64,280 kept for
theme discovery. Phase 1 established that filtering out short reviews raises the
1-star share, because angry users explain themselves and happy ones type "good".
A rating distribution built on the themed subset would be wrong, and wrong in a
direction that flatters nothing — it would just be untrue.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared
from shared import ALL_REVIEWS, sql, st

import plotly.express as px

shared.page("Overview", "📊", "The dataset behind everything else.")

summary = sql(f"""
    SELECT count(*)                                   AS reviews,
           min(reviewed_at)::date                     AS first_day,
           max(reviewed_at)::date                     AS last_day,
           round(avg(score)::numeric, 2)              AS avg_rating,
           count(*) FILTER (WHERE keep_for_themes)    AS themed,
           -- counted, not subtracted: this is the number that would expose a
           -- collection gap, and Phase 1 verified there are none.
           count(DISTINCT reviewed_at::date)          AS days
      FROM reviews WHERE {ALL_REVIEWS}""").iloc[0]

days = int(summary.days)

a, b, c, d = st.columns(4)
a.metric("Reviews collected", f"{summary.reviews:,}")
b.metric("Average rating", f"{summary.avg_rating} ★")
c.metric("Days covered", f"{days}",
         help=f"{summary.first_day} to {summary.last_day}")
d.metric("With enough text to cluster", f"{summary.themed:,}",
         delta=f"{summary.themed / summary.reviews * 100:.1f}% of all reviews",
         delta_color="off")

shared.corpus_note()
st.divider()

ratings = sql(f"""SELECT score, count(*) AS reviews FROM reviews
                   WHERE {ALL_REVIEWS} GROUP BY score ORDER BY score""")
ratings["share"] = ratings.reviews / ratings.reviews.sum() * 100

volume = sql(f"""SELECT reviewed_at::date AS day, count(*) AS reviews,
                        round(avg(score)::numeric, 2) AS avg_rating
                   FROM reviews WHERE {ALL_REVIEWS}
                  GROUP BY 1 ORDER BY 1""")

left, right = st.columns([2, 3])

with left:
    st.subheader("Rating distribution")
    fig = px.bar(ratings, x="score", y="reviews", text=ratings.share.round(1),
                 labels={"score": "Star rating", "reviews": "Reviews"})
    fig.update_traces(texttemplate="%{text}%", textposition="outside",
                      marker_color="#f2704a")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(tickmode="linear"))
    st.plotly_chart(fig, width='stretch')
    one, five = ratings.share.iloc[0], ratings.share.iloc[-1]
    st.caption(f"Bimodal, as app-store ratings usually are: **{one:.1f}% one-star** "
               f"and **{five:.1f}% five-star**, with little in between. People "
               f"tend to review when delighted or angry.")

with right:
    st.subheader("Review volume over time")
    smooth = st.checkbox("7-day rolling average", value=True,
                         help="Daily counts are noisy; the rolling average makes "
                              "the trend readable.")
    series = volume.copy()
    if smooth:
        series["reviews"] = series.reviews.rolling(7, min_periods=1).mean()
    fig = px.area(series, x="day", y="reviews",
                  labels={"day": "", "reviews": "Reviews per day"})
    fig.update_traces(line_color="#f2704a", fillcolor="rgba(242,112,74,0.18)")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch')
    st.caption(f"About {volume.reviews.mean():.0f} reviews a day across {days} days, "
               "with zero missing days. The dip in late April 2026 is real, not a "
               "gap in collection — worth explaining, not smoothing away.")

st.divider()
st.subheader("Average rating over time")
trend = volume.copy()
trend["avg_rating"] = trend.avg_rating.rolling(7, min_periods=1).mean()
fig = px.line(trend, x="day", y="avg_rating", labels={"day": "", "avg_rating": "Average ★"})
fig.update_traces(line_color="#2d6a9f")
fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[1, 5])
st.plotly_chart(fig, width='stretch')
st.caption("7-day rolling average. A sustained fall here is the signal the "
           "Trends page exists to explain.")
