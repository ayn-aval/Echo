"""Echo — the landing page.

    streamlit run app/main.py

Streamlit discovers app/pages/*.py automatically. app/label.py and app/audit.py
sit outside pages/ on purpose: they are internal annotation tools that write to
the database, and have no place in a dashboard someone is reading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import design
from shared import ALL_REVIEWS, MODEL, sql, st

design.setup("Echo")

st.markdown(
    f"<div style='color:{design.MUTED};font-size:.78rem;font-weight:600;"
    f"letter-spacing:.10em;text-transform:uppercase;margin-bottom:6px;'>Echo</div>",
    unsafe_allow_html=True)
design.header(
    "What are customers actually complaining about?",
    "100,000 Swiggy reviews from the Google Play Store, sorted automatically by "
    "what they are about — so the biggest problems, and the ones getting worse, "
    "are visible without anyone reading them all.")

s = sql(f"""SELECT count(*) AS reviews,
                   count(*) FILTER (WHERE score <= 2) AS unhappy,
                   min(reviewed_at)::date AS first_day,
                   max(reviewed_at)::date AS last_day
              FROM reviews WHERE {ALL_REVIEWS}""").iloc[0]

worst = sql("""SELECT coalesce(display_name, label) AS name, n_rows
                 FROM themes WHERE model = %s AND avg_rating <= 2.5
                ORDER BY n_rows DESC LIMIT 1""", (MODEL,))

design.tiles([
    ("Reviews analysed", f"{s.reviews:,}",
     f"{s.first_day:%B %Y} to {s.last_day:%B %Y}"),
    ("Unhappy customers", f"{s.unhappy / s.reviews * 100:.0f}%",
     f"{s.unhappy:,} people rated it 1 or 2 stars"),
    ("Biggest complaint",
     worst.name.iloc[0] if not worst.empty else "—",
     f"{worst.n_rows.iloc[0]:,} reviews" if not worst.empty else ""),
])

st.markdown("## Where to start")

PAGES = [
    ("Overview", "How many reviews there are, how people rate the app, and "
                 "whether satisfaction is improving."),
    ("Topics", "Every subject customers raise, ranked by how many people "
               "raised it. Open one to read the reviews behind it."),
    ("Trends", "Compare two periods to see which complaints are growing. "
               "Set the dividing line to a release date to see what it broke."),
    ("Search", "Describe a problem in your own words and find every review "
               "that means the same thing, however it was worded."),
    ("Results", "The evidence that this works, including where it does not."),
]

cols = st.columns(len(PAGES), gap="medium")
for col, (name, what) in zip(cols, PAGES):
    with col:
        st.markdown(
            f"<div class='card' style='height:100%;'>"
            f"<div style='font-weight:640;color:{design.INK};font-size:1rem;"
            f"margin-bottom:8px;'>{name}</div>"
            f"<div style='color:{design.INK_2};font-size:.88rem;line-height:1.55;'>"
            f"{what}</div></div>", unsafe_allow_html=True)

design.note("Choose a page from the sidebar on the left.")

st.markdown("## Why this is not a keyword search")
st.markdown(
    f"<div class='card' style='color:{design.INK_2};line-height:1.7;"
    f"font-size:.95rem;'>"
    "People describe the same problem in completely different words. "
    "<em>“App keeps crashing”</em>, <em>“closes by itself”</em> and "
    "<em>“shuts down when I open it”</em> are one problem written three ways, "
    "and counting keywords would file them as three.<br><br>"
    "Every review here is read by a model that was trained to judge whether two "
    "sentences mean the same thing, then adapted to the way people actually "
    "write app reviews — short, misspelled, and often mid-complaint. Reviews are "
    "then grouped by that meaning rather than by their words."
    "</div>", unsafe_allow_html=True)
