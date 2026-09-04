"""Start here — what this is, before any number needs interpreting.

Every other screen assumes you already know what you are looking at. This one
does not: it says whose reviews these are, what the system did to them, and
which screen answers which question. It is the only screen with no controls and
nothing to interpret.

The figures are queried rather than typed, so the page cannot quietly go stale
when the corpus is rebuilt — the same rule the rest of the app follows.
"""

import design
import streamlit as st
from shared import ALL_REVIEWS, MODEL, sql

design.appbar("Echo", "Customer feedback intelligence",
              right="Swiggy &nbsp;·&nbsp; Google Play")

facts = sql(f"""
    SELECT count(*) AS reviews,
           min(reviewed_at)::date AS first_day,
           max(reviewed_at)::date AS last_day,
           round(count(*) / greatest(
               (max(reviewed_at)::date - min(reviewed_at)::date) / 30.44, 1)
           )::int AS per_month
      FROM reviews WHERE {ALL_REVIEWS}""").iloc[0]

# The browsable count, not the raw one. Clustering produced 110 groups and four
# of them are set aside as unactionable (reviews grouped by language, or angry
# with no reason given), so Topics & search shows 106. Counting all 110 here
# would make the opening screen disagree with the screen it links to; "How it
# works" is where the four that were dropped are named.
topics = int(sql("SELECT count(*) AS n FROM themes "
                 "WHERE model = %s AND actionable", (MODEL,)).n.iloc[0])

design.hero(
    eyebrow="What this is",
    headline=(f"{facts.reviews:,} customer reviews, sorted into the "
              f"problems worth fixing"),
    value=f"{facts.reviews:,}",
    unit=f"reviews · {facts.first_day:%d %b} to {facts.last_day:%d %b %Y}",
    side=(f"About <b>{facts.per_month:,}</b> more arrive every month.<br>"
          f"No team has time to read them."))

# ── the three questions, which are also the three other screens ─────────────
st.markdown("## What you can answer here")

ANSWERS = [
    ("What should we fix first?",
     "A ranked list of problems, each with a real customer quote and the "
     "numbers behind its place in the list.",
     "views/home.py", "Open what to fix"),
    ("What are customers talking about?",
     f"{topics} subjects found in the reviews themselves — nobody wrote the "
     "list. Search matches meaning, so “money not returned” also finds “my "
     "refund never came”.",
     "views/issues.py", "Open topics and search"),
    ("Can I trust these numbers?",
     "Every figure on this site is produced by a script and checked against "
     "reviews read by hand. That screen also says what it gets wrong.",
     "views/accuracy.py", "Open how it works"),
]

for col, (question, answer, page, link) in zip(st.columns(3, gap="medium"),
                                               ANSWERS):
    with col:
        st.markdown(
            f"<div class='card' style='min-height:170px;'><h4>{question}</h4>"
            f"<div style='color:{design.INK_2};font-size:.88rem;line-height:1.55;'>"
            f"{answer}</div></div>", unsafe_allow_html=True)
        st.page_link(page, label=link)

# ── what the system actually does ───────────────────────────────────────────
st.markdown(f"## How it reads {facts.reviews:,} reviews")

STEPS = [
    ("Reads every review",
     f"All {facts.reviews:,} of them, written between {facts.first_day:%B %Y} "
     f"and {facts.last_day:%B %Y}, including the ones in Hindi and Hinglish."),
    ("Groups them by meaning",
     "<em>“App keeps crashing”</em> and <em>“it closes by itself”</em> count "
     "as one problem, though they share no words. That is how all "
     f"{topics} topics on this site were found."),
    ("Ranks what to fix",
     "By how many customers raised it, how unhappy they were, and whether it "
     "is growing week on week."),
]

for col, (n, (title, what)) in zip(st.columns(3, gap="medium"),
                                   enumerate(STEPS, 1)):
    with col:
        st.markdown(f"<div class='step'><div class='n'>{n}</div>"
                    f"<h4>{title}</h4><p>{what}</p></div>",
                    unsafe_allow_html=True)

design.note("Built on public Google Play reviews as a portfolio project. "
            "It is right about most reviews, not all of them — “How it works” "
            "shows how often, and where it fails.")
