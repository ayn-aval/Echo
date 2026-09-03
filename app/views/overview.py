"""Ratings — how customers score the app, and whether that is moving.

Deliberately covers all 100,000 reviews rather than the sidebar period. Phase 1
found that dropping short reviews raises the 1-star share, so the honest rating
figure is the one that counts every review; the app bar says so rather than
letting the reader assume the filter applied.
"""

import design
import plotly.graph_objects as go
from shared import ALL_REVIEWS, sql, st

s = sql(f"""
    SELECT count(*)                           AS reviews,
           min(reviewed_at)::date             AS first_day,
           max(reviewed_at)::date             AS last_day,
           round(avg(score)::numeric, 2)      AS avg_rating,
           count(*) FILTER (WHERE score <= 2) AS unhappy,
           count(*) FILTER (WHERE score >= 4) AS happy,
           count(DISTINCT reviewed_at::date)  AS days
      FROM reviews WHERE {ALL_REVIEWS}""").iloc[0]

design.appbar("Monitor", "Ratings",
              right=f"All <b>{s.reviews:,}</b> reviews &nbsp;·&nbsp; "
                    f"{s.first_day:%b} to {s.last_day:%b %Y}")

middle = s.reviews - s.unhappy - s.happy
design.hero(
    eyebrow="How customers score the app",
    headline="People either love it or hate it — almost nobody is in between",
    value=f"{s.avg_rating}",
    unit="average stars out of 5",
    side=(f"<b>{s.happy / s.reviews * 100:.0f}%</b> give 4 or 5 stars<br>"
          f"<b>{s.unhappy / s.reviews * 100:.0f}%</b> give 1 or 2 stars<br>"
          f"Only <b>{middle / s.reviews * 100:.0f}%</b> sit in the middle"))

ratings = sql(f"""SELECT score, count(*) AS reviews FROM reviews
                   WHERE {ALL_REVIEWS} GROUP BY score ORDER BY score""")
ratings["share"] = ratings.reviews / ratings.reviews.sum() * 100

volume = sql(f"""SELECT reviewed_at::date AS day, count(*) AS reviews,
                        avg(score) AS avg_rating
                   FROM reviews WHERE {ALL_REVIEWS} GROUP BY 1 ORDER BY 1""")

left, right = st.columns([2, 3], gap="large")

with left:
    st.markdown("## Stars given")
    colours = [design.NEGATIVE if v <= 2 else design.NEUTRAL if v == 3
               else design.POSITIVE for v in ratings.score]
    fig = go.Figure(go.Bar(
        x=[f"{v} star" if v == 1 else f"{v} stars" for v in ratings.score],
        y=ratings.reviews, marker_color=colours, marker_line_width=0, width=0.6,
        text=[f"{p:.0f}%" for p in ratings.share], textposition="outside",
        textfont=dict(color=design.INK_2, size=12),
        hovertemplate="%{x}<br>%{y:,} reviews<extra></extra>"))
    fig.update_traces(marker_cornerradius=5)
    st.plotly_chart(design.style(fig, height=330, ylab="Reviews"),
                    width="stretch", config={"displayModeBar": False})

with right:
    st.markdown("## Reviews per day")
    v = volume.copy()
    v["smooth"] = v.reviews.rolling(7, min_periods=1).mean()
    fig = go.Figure(go.Scatter(
        x=v.day, y=v.smooth, mode="lines", line=dict(color=design.BLUE, width=2.2),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
        hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} reviews<extra></extra>"))
    st.plotly_chart(design.style(fig, height=330, ylab="Reviews per day"),
                    width="stretch", config={"displayModeBar": False})
    design.note("Smoothed over 7 days. The dip in late April is real — fewer "
                "people reviewed, not missing data.")

v = volume.copy()
v["smooth"] = v.avg_rating.rolling(14, min_periods=1).mean()
first, last = v.smooth.iloc[0], v.smooth.iloc[-1]

st.markdown("## Rating over time")
design.note(f"{'Up' if last > first else 'Down'} from {first:.2f} to {last:.2f} "
            f"stars since January. Smoothed over 14 days.")

fig = go.Figure(go.Scatter(
    x=v.day, y=v.smooth, mode="lines",
    line=dict(color=design.BLUE, width=2.2),
    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} stars<extra></extra>"))
fig.add_hline(y=3, line_width=1, line_color=design.AXIS)
st.plotly_chart(design.style(fig, height=290, ylab="Average stars"),
                width="stretch", config={"displayModeBar": False})

with st.expander("About this data"):
    st.markdown(
        f"- **{s.reviews:,} reviews** from the Google Play Store, "
        f"{s.first_day:%d %B %Y} to {s.last_day:%d %B %Y}, no missing days.\n"
        f"- This screen counts every review, whatever the sidebar filter says.\n"
        f"- Topics elsewhere use the **64,280** reviews longer than one word — "
        f"a review saying only \"good\" is not about anything.")
