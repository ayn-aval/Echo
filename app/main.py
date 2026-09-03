"""Echo — customer feedback intelligence.

    streamlit run app/main.py

The application shell: page config, stylesheet, brand, and navigation. Individual
screens live in app/views/ and are registered below rather than discovered from a
pages/ folder, so their titles, grouping and order are chosen here instead of
being derived from filenames.

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

NAV = {
    "Monitor": [
        st.Page("views/home.py", title="Today", url_path="today", default=True),
        st.Page("views/overview.py", title="Volume and ratings", url_path="volume"),
    ],
    "Understand": [
        st.Page("views/issues.py", title="What customers raise", url_path="issues"),
        st.Page("views/trends.py", title="What changed", url_path="changes"),
        st.Page("views/search.py", title="Find reviews", url_path="find"),
    ],
    "About": [
        st.Page("views/accuracy.py", title="How it works", url_path="how"),
    ],
}

page = st.navigation(NAV, position="sidebar", expanded=True)

with st.sidebar:
    st.markdown(
        f"<div class='sidenote'>Swiggy · Google Play<br>"
        f"100,000 reviews to 26 August 2026</div>", unsafe_allow_html=True)

page.run()
