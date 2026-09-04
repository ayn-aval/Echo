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

st.set_page_config(page_title="Echo — what customers are complaining about",
                   page_icon=design.MARK, layout="wide",
                   initial_sidebar_state="expanded")
design.boot()

# Four screens, numbered because they are meant to be read in order the first
# time: what this is, what to fix, the detail behind it, and whether to believe
# any of it. Titles name a question rather than a location.
PAGES = [
    st.Page("views/start.py", title="1  Start here", url_path="start",
            default=True),
    st.Page("views/home.py", title="2  What to fix first", url_path="overview"),
    st.Page("views/issues.py", title="3  Explore complaints",
            url_path="explore"),
    st.Page("views/accuracy.py", title="4  Is it accurate?", url_path="how"),
]

# The wordmark is CSS on the sidebar header (see design.py): Streamlit fixes the
# sidebar's child order, so anything written here would appear under the menu.
page = st.navigation(PAGES, position="sidebar", expanded=True)

# One filter for the whole app, set once and honoured by every screen. Kept in
# session_state so it survives navigation.
import filters  # noqa: E402  (after boot, so the stylesheet is already applied)

st.session_state["filters"] = filters.bar()

page.run()
