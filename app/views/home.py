"""Today — where to improve, by business area."""

import design
import insights
import plotly.graph_objects as go
import streamlit as st
from shared import sql

F = st.session_state["filters"]
h = insights.headline_health()

design.appbar("Monitor", "Where to improve",
              right=f"To <b>{h['latest']:%d %b %Y}</b> &nbsp;·&nbsp; {F.label}")

d = h["rating_delta"]
design.tiles([
    ("Rating", f"{h['rating_28d']:.2f}",
     design.trend(d, f"{d:+.2f} vs previous 28 days")),
    ("Unhappy", f"{h['unhappy'] / h['reviews'] * 100:.0f}%",
     f"{h['unhappy']:,} rated 1–2 stars"),
    ("Reviews", f"{h['reviews']:,}", "grouped by subject"),
])

# ── business areas ──────────────────────────────────────────────────────────
areas = sql(f"""
    SELECT t.category AS area,
           count(*) AS reviews,
           round(avg(r.score)::numeric, 2) AS rating,
           count(*) FILTER (WHERE r.score <= 2) AS unhappy
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
      JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
     WHERE rt.model = 'sbert-domain' AND rt.theme_id >= 0
       AND t.category NOT IN ('General praise', 'Other')
       {F.where('r')}{F.area_clause('t')}
     GROUP BY 1 ORDER BY count(*) FILTER (WHERE r.score <= 2) DESC""")

spark = sql(f"""
    SELECT t.category AS area, date_trunc('week', r.reviewed_at)::date AS week,
           count(*) FILTER (WHERE r.score <= 2) AS unhappy
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
      JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
     WHERE rt.model = 'sbert-domain' AND rt.theme_id >= 0
       {F.where('r')}{F.area_clause('t')}
     GROUP BY 1, 2 ORDER BY 2""")

st.markdown("## Complaints by area")

if areas.empty:
    st.info("No reviews in this selection. Widen the period or clear the area filter.")
else:
    cols = st.columns(min(len(areas), 3), gap="medium")
    for i, r in enumerate(areas.itertuples()):
        s = spark[spark.area == r.area]
        with cols[i % len(cols)]:
            st.markdown(
                f"<div class='area'><div class='area-top'>"
                f"<div class='area-name'>{r.area}</div>"
                f"{design.rating_chip(float(r.rating))}</div>"
                f"<div class='area-num'>{r.unhappy:,}</div>"
                f"<div class='area-sub'>unhappy of {r.reviews:,} reviews</div>"
                f"</div>", unsafe_allow_html=True)
            if len(s) > 2:
                st.plotly_chart(design.sparkline(s.week, s.unhappy),
                                width="stretch",
                                config={"displayModeBar": False},
                                key=f"spark{i}")

# ── what to fix first ───────────────────────────────────────────────────────
st.markdown("## Fix these first")

top = insights.priorities(limit=4, areas=F.areas, days=F.days)
if top.empty:
    st.info("Nothing meets the threshold in this selection.")
else:
    LABEL = {"crit": ("Urgent", design.CRITICAL), "warn": ("Watch", design.WARNING),
             "flat": ("Steady", design.MUTED)}
    for i, r in enumerate(top.itertuples(), 1):
        word, accent = LABEL[r.urgency]
        st.markdown(
            f"<div class='issue' style='--accent:{accent};'>"
            f"<div class='rank'>{i}</div><div class='body'>"
            f"<div class='name'>{r.name}"
            f"{design.badge(word, r.urgency)}"
            f"<span class='area-tag'>{r.area}</span></div>"
            f"<div class='meta'>{r.why}</div>"
            f"<div class='quote'>“{' '.join(str(r.content).split())[:170]}…”</div>"
            f"</div></div>", unsafe_allow_html=True)
    st.page_link("views/issues.py", label="Open all topics")

with st.expander("How this list is ranked"):
    st.markdown(
        "Three inputs: how many customers raised it in the period, how low they "
        "rated the app, and whether its share of complaints is growing. Growth "
        "only raises a score — a large, flat complaint still needs fixing.\n\n"
        "Topics that are angry without saying why, and the topic that groups "
        "reviews by language rather than subject, are excluded here. Both stay "
        "visible under **What customers raise**.")
