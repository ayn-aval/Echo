"""Echo — customer feedback intelligence.

    streamlit run app/main.py

The application shell: page config, stylesheet, brand, and navigation. Individual
screens live in app/views/ and are registered below rather than discovered from a
pages/ folder, so their titles and order are chosen here instead of being derived
from filenames.

app/label.py and app/audit.py stay outside this navigation on purpose. They are
internal annotation tools that write to the database, and are run directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import design
import streamlit as st

st.set_page_config(page_title="Echo — customer feedback intelligence",
                   page_icon=design.MARK, layout="wide",
                   initial_sidebar_state="expanded")
design.boot()

# Four screens, ungrouped. Seven needed Monitor/Understand/About headings to be
# navigable; four do not, and the headings were themselves something to read.
#
# "Start here" carries no controls and no figure that needs interpreting. It
# exists because every other screen assumes the reader already knows whose
# reviews these are and what was done to them, and a first-time visitor does not.
#
# Titles name what the screen answers rather than where it sits: "Overview" and
# "Explore" describe a location, "What to fix" describes a question. url_paths
# are unchanged so existing links keep resolving.
PAGES = [
    st.Page("views/start.py", title="Start here", url_path="start", default=True),
    st.Page("views/home.py", title="What to fix", url_path="overview"),
    st.Page("views/issues.py", title="Topics & search", url_path="explore"),
    st.Page("views/accuracy.py", title="How it works", url_path="how"),
]

page = st.navigation(PAGES, position="sidebar", expanded=True)

# One filter bar for the whole app: a reader sets their area and period once and
# every screen answers for that slice. Stored in session_state so it survives
# navigation, and nameable so it survives the browser closing.
import filters  # noqa: E402  (after boot, so the stylesheet is already applied)

st.session_state["filters"] = filters.bar()

page.run()
