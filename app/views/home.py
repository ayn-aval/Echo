"""What to fix first — the ranked problem list, and anything that spiked.

The window is eight weeks, not one. A single week of this corpus puts only about
47 reviews behind the biggest problem and shows most topics falling, so a weekly
frame makes a real problem look like nothing happening. Spikes are still
detected weekly — that is what a spike is — and reported as a sentence rather
than as a statistic.

Clicking an area filters the whole screen to it. That click is what replaces the
business-area dropdown that used to sit in the sidebar on every screen.
"""

import html

import design
import filters
import insights
import plotly.graph_objects as go
import streamlit as st
from shared import ALL_REVIEWS, MODEL, sql

F = st.session_state["filters"]
picked = filters.current_area()


def clean(text, limit=None):
    """Escape review text before it goes anywhere near an HTML string."""
    out = " ".join(str(text).split())
    if limit and len(out) > limit:
        out = out[:limit].rstrip() + "…"
    return html.escape(out)


design.kicker(F.period)
st.markdown("# What are customers complaining about?")

window = sql(f"""SELECT count(*) AS reviews FROM reviews
                  WHERE {ALL_REVIEWS} {F.where('reviews')}""").reviews.iloc[0]
# Not every review lands in a problem: most are too short or too vague to place,
# and four groups are set aside as unactionable. Saying "they fall into the
# problems below" would imply all 26,000 are accounted for by eight rows.
placed = sql(f"""
    SELECT count(*) AS n
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
      JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
     WHERE rt.model = %s AND rt.theme_id >= 0 AND t.actionable
       {F.where('r')}""", (MODEL,)).n.iloc[0]
design.sub(f"{window:,} reviews came in over the {F.period.lower()}. "
           f"{placed:,} of them said something specific enough to place in a "
           "problem; the biggest are below, ranked by how many people raised "
           "each one, how badly they rated it, and whether it is growing.")

# ── the loudest single fact: the sharpest recent spike ──────────────────────
spike = sql("""
    SELECT a.week_start, a.reviews, a.baseline_mean, a.kind,
           coalesce(t.display_name, t.label) AS name
      FROM theme_alerts a
      JOIN themes t ON t.model = a.model AND t.theme_id = a.theme_id
     WHERE a.model = %s AND t.actionable
     ORDER BY a.z DESC NULLS LAST LIMIT 1""", (MODEL,))

if not spike.empty:
    s = spike.iloc[0]
    if s.kind == "new" or not s.baseline_mean:
        text = (f"“{clean(s['name'])}” appeared for the first time in the week "
                f"of {s.week_start:%d %B} — {s.reviews} reviews, against none "
                f"before.")
    else:
        times = s.reviews / float(s.baseline_mean)
        text = (f"Complaints about “{clean(s['name'])}” ran at {times:.1f} times "
                f"their normal rate in the week of {s.week_start:%d %B} — "
                f"{s.reviews} reviews, against about {s.baseline_mean:.0f} in a "
                f"normal week.")
    design.poster("Sharpest jump on record", text)

st.write("")
main, side = st.columns([7, 4], gap="large")

# ── the ranked list ─────────────────────────────────────────────────────────
with main:
    head, clear = st.columns([4, 1])
    with head:
        where = f"filtered to {picked}" if picked else "all parts of the business"
        st.markdown(f"## Ranked problems"
                    f'<span style="font-size:13px;font-weight:400;'
                    f'color:{design.MUTED};margin-left:10px">{where}</span>',
                    unsafe_allow_html=True)
    with clear:
        if picked and st.button("Clear filter", key="clear"):
            filters.set_area(None)
            st.rerun()

    rows = insights.priorities(limit=8, areas=F.areas, days=F.days)
    if rows.empty:
        st.info("Nothing meets the threshold in this slice.")
    else:
        body = ""
        for i, r in enumerate(rows.itertuples(), 1):
            if r.change_pct != r.change_pct:            # NaN: no prior window
                move, cls = "new", "down"
            else:
                cls = "up" if r.change_pct > 0 else "down"
                move = f"{'+' if r.change_pct > 0 else ''}{r.change_pct:.0f}%"
            body += (
                "<tr>"
                f'<td class="n">{i}</td>'
                f'<td><div class="name">{clean(r.name)}</div>'
                f'<div class="said">“{clean(r.content, 78)}” · {clean(r.area)}'
                f"</div></td>"
                f'<td class="num">{int(r.reviews):,}</td>'
                f'<td class="{cls}">{move}</td>'
                f'<td class="num">{r.avg_rating:.1f}</td>'
                "</tr>")
        st.markdown(
            '<table class="ranked"><thead><tr>'
            '<th style="width:30px">#</th>'
            "<th>The problem, in customers’ words</th>"
            "<th>Reviews</th><th>vs prior<br>window</th><th>Avg<br>rating</th>"
            f"</tr></thead><tbody>{body}</tbody></table>",
            unsafe_allow_html=True)
        design.sub("“vs prior window” compares this window against the same "
                   "problem over the window before it.")

# ── where the unhappiness sits, and the filter ──────────────────────────────
with side:
    st.markdown("### Where the unhappiness sits")
    design.sub("Reviews rated 1 or 2 stars, by part of the business. "
               "Click one to filter the list.")

    areas = sql(f"""
        SELECT t.category AS area,
               count(*) FILTER (WHERE r.score <= 2) AS unhappy
          FROM review_themes rt
          JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
          JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
         WHERE rt.model = %s AND rt.theme_id >= 0 AND t.actionable
           AND t.category NOT IN ('General praise', 'Other')
           {F.where('r')}
         GROUP BY 1 ORDER BY 2 DESC""", (MODEL,))

    if not areas.empty:
        top = int(areas.unhappy.max()) or 1
        for a in areas.itertuples():
            design.bar(a.area, f"{int(a.unhappy):,}",
                       100 * int(a.unhappy) / top, selected=picked == a.area)
            if st.button(f"Filter to {a.area}", key=f"area_{a.area}"):
                filters.set_area(None if picked == a.area else a.area)
                st.rerun()

# ── the health behind it ────────────────────────────────────────────────────
design.rule()
st.markdown("## How customers rate the app overall")
design.sub("All 100,000 reviews. These two are deliberately not affected by the "
           "time period in the menu.")

left, right = st.columns([2, 3], gap="large")
with left:
    st.markdown("### Stars people gave")
    stars = sql(f"""SELECT score, count(*) AS reviews FROM reviews
                     WHERE {ALL_REVIEWS} GROUP BY score ORDER BY score""")
    # Two states, not three. A 1-to-5 scale is diverging, and a diverging scale
    # needs two hues plus a neutral middle; this palette has one hue. Rather
    # than invent a second, the chart says the only thing that matters.
    colours = [design.ACCENT if v <= 2 else design.NEUTRAL_700
               for v in stars.score]
    fig = go.Figure(go.Bar(
        x=[f"{v} star" if v == 1 else f"{v} stars" for v in stars.score],
        y=stars.reviews, marker_color=colours, marker_line_width=0, width=0.62,
        text=[f"{p:.0f}%" for p in stars.reviews / stars.reviews.sum() * 100],
        textposition="outside", textfont=dict(color=design.INK, size=12),
        hovertemplate="%{x}<br>%{y:,} reviews<extra></extra>"))
    st.plotly_chart(design.style(fig, height=280, ylab="Reviews"),
                    width="stretch", config={"displayModeBar": False})
    design.sub("Red is 1 and 2 stars — the reviews everything else on this "
               "screen is about.")

with right:
    st.markdown("### Average rating over time")
    daily = sql(f"""SELECT reviewed_at::date AS day, avg(score) AS rating
                      FROM reviews WHERE {ALL_REVIEWS} GROUP BY 1 ORDER BY 1""")
    daily["smooth"] = daily.rating.rolling(14, min_periods=1).mean()
    st.plotly_chart(
        design.line(daily.day, daily.smooth,
                    "%{x|%d %b %Y}<br>%{y:.2f} stars<extra></extra>",
                    height=280, ylab="Average stars"),
        width="stretch", config={"displayModeBar": False})
    design.sub("Smoothed over 14 days, so one bad day does not read as a trend.")
