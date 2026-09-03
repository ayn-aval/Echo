"""Overview — what customers are saying, at a glance."""

import design
import plotly.graph_objects as go
from shared import ALL_REVIEWS, sql, st

design.appbar("Monitor", "Volume and ratings",
              "How many people are reviewing, how they rate the app, and whether that is moving.")

s = sql(f"""
    SELECT count(*)                                AS reviews,
           min(reviewed_at)::date                  AS first_day,
           max(reviewed_at)::date                  AS last_day,
           round(avg(score)::numeric, 2)           AS avg_rating,
           count(*) FILTER (WHERE score <= 2)      AS unhappy,
           count(DISTINCT reviewed_at::date)       AS days
      FROM reviews WHERE {ALL_REVIEWS}""").iloc[0]

ratings = sql(f"""SELECT score, count(*) AS reviews FROM reviews
                   WHERE {ALL_REVIEWS} GROUP BY score ORDER BY score""")
ratings["share"] = ratings.reviews / ratings.reviews.sum() * 100

volume = sql(f"""SELECT reviewed_at::date AS day, count(*) AS reviews,
                        avg(score) AS avg_rating
                   FROM reviews WHERE {ALL_REVIEWS} GROUP BY 1 ORDER BY 1""")

design.tiles([
    ("Reviews", f"{s.reviews:,}", f"{s.first_day:%d %b} to {s.last_day:%d %b %Y}"),
    ("Average rating", f"{s.avg_rating}", "out of 5 stars"),
    ("Unhappy customers", f"{s.unhappy / s.reviews * 100:.0f}%",
     f"{s.unhappy:,} people gave 1 or 2 stars"),
    ("Reviews per day", f"{s.reviews / s.days:,.0f}", f"across {s.days} days"),
])

st.markdown("## How customers rate the app")
design.note("Most people either love it or hate it. Very few sit in the middle.")

left, right = st.columns([2, 3], gap="large")

with left:
    colours = [design.NEGATIVE if v <= 2 else design.NEUTRAL if v == 3
               else design.POSITIVE for v in ratings.score]
    fig = go.Figure(go.Bar(
        x=[f"{v} star" if v == 1 else f"{v} stars" for v in ratings.score],
        y=ratings.reviews, marker_color=colours,
        marker_line_width=0, width=0.62,
        text=[f"{p:.0f}%" for p in ratings.share], textposition="outside",
        textfont=dict(color=design.INK_2, size=12),
        hovertemplate="%{x}<br>%{y:,} reviews<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    st.plotly_chart(design.style(fig, height=330, ylab="Reviews"),
                    width="stretch", config={"displayModeBar": False})

with right:
    v = volume.copy()
    v["smooth"] = v.reviews.rolling(7, min_periods=1).mean()
    fig = go.Figure(go.Scatter(
        x=v.day, y=v.smooth, mode="lines", line=dict(color=design.BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
        hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} reviews/day<extra></extra>"))
    st.plotly_chart(design.style(fig, height=330, ylab="Reviews per day"),
                    width="stretch", config={"displayModeBar": False})
    design.note("Daily review volume, smoothed over 7 days. The dip in late "
                "April is real — fewer people reviewed, not a gap in the data.")

st.markdown("## Is satisfaction improving?")

v = volume.copy()
v["smooth"] = v.avg_rating.rolling(14, min_periods=1).mean()
first, last = v.smooth.iloc[0], v.smooth.iloc[-1]
direction = "improved" if last > first else "declined"

fig = go.Figure(go.Scatter(
    x=v.day, y=v.smooth, mode="lines",
    line=dict(color=design.BLUE, width=2),
    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} stars<extra></extra>"))
fig.add_hline(y=3, line_width=1, line_color=design.AXIS)
st.plotly_chart(design.style(fig, height=280, ylab="Average rating"),
                width="stretch", config={"displayModeBar": False})

design.note(f"Average rating over time, smoothed over 14 days. It has "
            f"{direction} from {first:.2f} to {last:.2f} stars since January. "
            f"The line marks 3 stars — the neutral middle.")

with st.expander("About this data"):
    st.markdown(f"""
Reviews were collected from the Google Play Store for the Swiggy Android app,
covering **{s.first_day:%d %B %Y} to {s.last_day:%d %B %Y}** with no missing days.

The figures on this page use **all {s.reviews:,} reviews**. The Themes and Trends
pages use the **64,280 reviews with more than one word**, because a review that
says only "good" cannot be grouped by what it is about. Both numbers are reported
rather than one standing in for the other: dropping short reviews would raise the
share of unhappy customers, so a rating chart built on that smaller set would
overstate dissatisfaction.
""")
