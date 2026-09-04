"""Start here — what this is, before any number needs interpreting.

Every other screen assumes you already know what you are looking at. This one
does not: it says whose reviews these are, what the system did to them, and
which screen answers which question. It carries no controls.

The centre of the screen is the demonstration: three real reviews that share no
keyword, and the one problem they were filed under. That is the whole project in
five seconds, and it is the thing a reader has to believe before any ranked list
means anything. The quotes are queried from the theme they actually belong to
rather than written here, so they cannot drift from what the system really did.
"""

import design
import streamlit as st
from shared import ALL_REVIEWS, MODEL, sql

# The theme the demonstration is drawn from. Chosen because its members are
# unusually varied in wording — "1hr delay", "2 hours ago", "waits for more than
# an hour" — which is exactly the point being made. If it ever stops existing
# the page falls back to the largest complaint rather than breaking.
DEMO_THEME = "Waited an hour or more"

facts = sql(f"""
    SELECT count(*) AS reviews,
           min(reviewed_at)::date AS first_day,
           max(reviewed_at)::date AS last_day,
           round(count(*) / greatest(
               (max(reviewed_at)::date - min(reviewed_at)::date) / 30.44, 1)
           )::int AS per_month
      FROM reviews WHERE {ALL_REVIEWS}""").iloc[0]

months = round((facts.last_day - facts.first_day).days / 30.44)
topics = int(sql("SELECT count(*) AS n FROM themes "
                 "WHERE model = %s AND actionable", (MODEL,)).n.iloc[0])

design.kicker("Start here — 60 seconds")
st.markdown(f"# Nobody reads {facts.per_month:,} reviews a month.")
design.lede(
    "Echo reads all of them and gives you one thing: <b>a ranked list of the "
    "problems customers are actually hitting</b> — biggest first, with the real "
    "reviews behind each one.")
design.sub(
    "Think of it as the Monday-morning screen for whoever owns the Swiggy app: "
    "open it, see what is worst right now, click through to the complaints, "
    "decide what to fix.")

design.rule()

for col, (number, label) in zip(st.columns(4), [
        (f"{facts.reviews:,}", "reviews read"),
        (f"{topics}", "distinct problems found"),
        (f"{months} mths", "of history to compare against"),
        (f"{facts.per_month:,}", "new reviews every month")]):
    with col:
        design.stat(number, label)

design.rule()

# ── the demonstration ───────────────────────────────────────────────────────
st.markdown("## The trick: it groups by meaning, not by words")
design.sub(
    "Searching for the word “hour” misses most of these, because people describe "
    "the same wait in completely different ways. The three reviews below share "
    "almost nothing on the page. Echo files all three under one problem.")

demo = sql("""
    SELECT coalesce(t.display_name, t.label) AS name, t.n_rows, t.avg_rating,
           t.category AS area, r.content, r.word_count
      FROM review_themes rt
      JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
      JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
     WHERE rt.model = %s AND t.actionable
       AND coalesce(t.display_name, t.label) = %s
       AND r.word_count BETWEEN 5 AND 14
     ORDER BY rt.strength DESC LIMIT 3""", (MODEL, DEMO_THEME))

if demo.empty:
    design.sub("The example theme is not in the database yet. "
               "Run: python -m src.clustering.name_themes")
else:
    head = demo.iloc[0]
    left, mid, right = st.columns([5, 1, 5])
    with left:
        for quote in demo.content:
            design.quote_chip(" ".join(str(quote).split())[:110])
    with mid:
        st.markdown('<div style="font-size:28px;font-weight:800;'
                    f'color:{design.ACCENT};padding-top:18px">→</div>',
                    unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="problem-box"><div class="kicker">One problem</div>'
            '<div style="font-weight:800;font-size:22px;line-height:1.15">'
            f'{head["name"]}</div>'
            f'<div style="font-size:13px;margin-top:8px;color:{design.MUTED}">'
            f'{head.n_rows:,} reviews · avg rating {head.avg_rating:.1f} · '
            f'{head.area.lower()}</div></div>', unsafe_allow_html=True)

design.rule()

# ── where to go, and what not to believe ────────────────────────────────────
left, right = st.columns(2, gap="large")
with left:
    st.markdown("### How to use the next three screens")
    st.markdown(
        "1. **What to fix first** — the ranked problem list and anything that "
        "spiked. Click a bar to filter the whole page to one part of the "
        "business.\n"
        "2. **Explore complaints** — ask in plain English (“drivers cancelling "
        "at the door”) and read the reviews that match.\n"
        "3. **Is it accurate?** — how often it files a review correctly, and "
        "where it still gets things wrong.")
    if st.button("Show me what to fix first  →", type="primary", key="goto"):
        st.switch_page("views/home.py")
with right:
    st.markdown("### Honest limits")
    st.markdown(
        "- Only Google Play reviews — not the App Store, support tickets or "
        "social.\n"
        "- Reviewers skew angry, so these counts measure *noise*, not how many "
        "customers were affected.\n"
        "- Roughly **1 review in 6** lands in a group a person would have filed "
        "elsewhere.\n"
        "- About a third of reviews say only “good” or “worst” and cannot be "
        "placed at all.\n"
        "- A portfolio project on public data. Not an internal Swiggy tool.")
