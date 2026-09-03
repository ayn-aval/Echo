"""Today — the finding, then what to do about it.

This screen used to open with three tiles of equal weight, which left the reader
to work out which of them mattered. It now opens with one sentence naming the
worst business area and one number, because that is the question the growth team
actually arrives with.
"""

import design
import insights
import streamlit as st
from shared import sql

F = st.session_state["filters"]
h = insights.headline_health()

design.appbar("Monitor", "Today",
              right=f"To <b>{h['latest']:%d %b %Y}</b> &nbsp;·&nbsp; {F.label}")

# ── how each business area is doing, now and in the period before ───────────
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
       {F.where('r')}{F.area_clause('t')}
     GROUP BY 1 ORDER BY count(*) FILTER (WHERE r.score <= 2) DESC""")

if areas.empty:
    st.info("No reviews in this selection. Widen the period or clear the area filter.")
    st.stop()

# The same measure over the period immediately before this one, so the headline
# can say whether the worst area is getting worse. Skipped on "All time",
# because there is no period before all of time.
prior = sql(f"""
    SELECT t.category AS area, count(*) FILTER (WHERE r.score <= 2) AS unhappy
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
      JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
     WHERE rt.model = 'sbert-domain' AND rt.theme_id >= 0 AND t.actionable
       AND r.reviewed_at <  (SELECT max(reviewed_at) FROM reviews WHERE app='swiggy')
                            - interval '{F.days} days'
       AND r.reviewed_at >= (SELECT max(reviewed_at) FROM reviews WHERE app='swiggy')
                            - interval '{2 * F.days if F.days else 0} days'
       {F.area_clause('t')}
     GROUP BY 1""") if F.days else None

top = areas.iloc[0]
was = None
if prior is not None and not prior.empty:
    match = prior[prior.area == top.area]
    was = int(match.unhappy.iloc[0]) if not match.empty else None

if was:
    # Said as a direction, not a signed number: fewer complaints is good news and
    # "-13%" reads to most people as something having got worse.
    pct = (top.unhappy - was) / was * 100
    movement = (f"<b>{abs(pct):.0f}% {'more' if pct > 0 else 'fewer'}</b> "
                f"than the {F.days} days before"
                if abs(pct) >= 5 else "Level with the period before")
else:
    movement = f"<b>{top.unhappy / top.reviews * 100:.0f}%</b> of what is said about it"

design.hero(
    eyebrow=f"{F.period} · biggest opportunity",
    headline=f"{top.area} is where you are losing people",
    value=f"{top.unhappy:,}",
    unit="unhappy customers",
    side=(f"{movement}<br>Rated <b>{top.rating:.1f}</b> out of 5 across "
          f"<b>{top.reviews:,}</b> reviews"))

# ── anything urgent ─────────────────────────────────────────────────────────
recent = sql(f"""
    SELECT count(*) AS n, max(a.week_start) AS week
      FROM theme_alerts a
      JOIN themes t ON t.model = a.model AND t.theme_id = a.theme_id
     WHERE a.model = 'sbert-domain' AND t.actionable
       AND a.week_start >= (SELECT max(week_start) FROM theme_alerts) - 28
       {F.area_clause('t')}""").iloc[0]
if recent.n:
    st.markdown(
        f"<div class='card' style='border-left:4px solid {design.CRITICAL};"
        f"display:flex;align-items:center;gap:14px;'>"
        f"<span style='font-weight:680;color:{design.INK};font-size:1.02rem;'>"
        f"{recent.n} topic{'s' if recent.n > 1 else ''} spiked</span>"
        f"<span style='color:{design.INK_2};font-size:.9rem;'>"
        f"in the four weeks to {recent.week:%d %B}</span></div>",
        unsafe_allow_html=True)
    st.page_link("views/alerts.py", label="See what spiked")

# ── the answer to "so what do we do" ────────────────────────────────────────
st.markdown("## Fix these first")

top_issues = insights.priorities(limit=4, areas=F.areas, days=F.days)
if top_issues.empty:
    st.info("Nothing meets the threshold in this selection.")
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

# ── the breakdown behind the headline ───────────────────────────────────────
st.markdown("## Unhappy customers by area")

# One colour for every bar. Length already carries the count; colouring the same
# bar by star rating put a second, unlabelled encoding on the same mark, so the
# biggest bar could be a milder colour than a smaller one for reasons nothing on
# screen explained. The rating stays in the row, coloured, where it reads as its
# own fact.
worst = int(areas.unhappy.max())


def stars(rating: float) -> str:
    colour = ("#a82f2f" if rating <= 2.0
              else "#8f5613" if rating < 3.5 else "#0a6f0a")
    return (f"of {{:,}} · <span style='color:{colour};font-weight:640;'>"
            f"{rating:.1f} stars</span>")


design.rank_rows([
    (r.area, r.unhappy / worst, f"{r.unhappy:,}",
     stars(float(r.rating)).format(r.reviews), design.BLUE)
    for r in areas.itertuples()])

with st.expander("How this list is ranked"):
    st.markdown(
        "- How many customers raised it\n"
        "- How low they rated the app\n"
        "- Whether it is growing\n\n"
        "Reviews that are angry without saying why are left out.")
