"""Echo — what Swiggy customers are complaining about.

    streamlit run app/main.py

The shell: page config, stylesheet, the ink header, and the tab router.

Navigation is a hand-rolled router — `st.session_state["screen"]` plus four
buttons — rather than `st.navigation`. Streamlit's own navigation only renders
in the sidebar, and this design puts the tabs in a top bar above the content,
with no sidebar at all. Screens are modules exposing `render()`, not standalone
scripts, which is why eval/check_app.py drives them through this file.

app/label.py and app/audit.py stay outside this router on purpose. They are
internal annotation tools that write to the database, and are run directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import design
import streamlit as st

st.set_page_config(page_title="Echo — what customers are complaining about",
                   page_icon=design.MARK, layout="wide",
                   initial_sidebar_state="collapsed")
design.boot()

import prose  # noqa: E402  (after boot, so the stylesheet is applied)
from views import ask, fix_list, this_week, trust  # noqa: E402

st.session_state.setdefault("screen", "week")

design.ink_header(prose.ONE_LINER, prose.HEADER_META)
screen = design.tabs(st.session_state["screen"], prose.dateline())

{"week": this_week.render,
 "fix": fix_list.render,
 "ask": ask.render,
 "trust": trust.render}[screen]()
