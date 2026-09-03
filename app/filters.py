"""Global filters, shared by every screen, plus named views to return to.

One filter bar in the sidebar rather than per-page controls: a team member sets
their area and period once and every screen answers for that slice. The state
lives in st.session_state so it survives navigation, and a named view writes it
to Postgres so it survives the browser closing.
"""

import json
from dataclasses import dataclass

import pandas as pd
import psycopg2
import streamlit as st

from shared import sql
from src.db.connection import connection

PERIODS = {"Last 4 weeks": 28, "Last 8 weeks": 56, "Last 3 months": 90,
           "Last 6 months": 180, "All time": None}
DEFAULT_PERIOD = "Last 3 months"


@dataclass(frozen=True)
class Filters:
    period: str
    areas: tuple
    days: int | None

    def where(self, alias: str = "r") -> str:
        """SQL fragment for the review table under the current period."""
        if self.days is None:
            return ""
        return (f" AND {alias}.reviewed_at >= (SELECT max(reviewed_at) FROM reviews"
                f" WHERE app='swiggy') - interval '{self.days} days'")

    def area_clause(self, alias: str = "t") -> str:
        if not self.areas:
            return ""
        joined = ", ".join(f"'{a}'" for a in self.areas)
        return f" AND {alias}.category IN ({joined})"

    @property
    def label(self) -> str:
        area = "all areas" if not self.areas else ", ".join(self.areas)
        return f"{self.period.lower()} · {area}"


@st.cache_data(ttl=600, show_spinner=False)
def all_areas() -> list:
    df = sql("""SELECT category, sum(n_rows) AS n FROM themes
                 WHERE model='sbert-domain' AND category IS NOT NULL
                 GROUP BY 1 ORDER BY 2 DESC""")
    return df.category.tolist()


def _views() -> pd.DataFrame:
    return sql("SELECT name, payload FROM saved_views ORDER BY created_at DESC")


# The public deployment connects as a read-only role, so these writes are
# refused there by design. A visitor should see a plain sentence, not a traceback.
READ_ONLY_NOTE = "Saving views is turned off on the public demo."


def _save(name: str, f: Filters) -> bool:
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO saved_views (name, payload) VALUES (%s, %s)
                           ON CONFLICT (name) DO UPDATE
                           SET payload = EXCLUDED.payload, created_at = now()""",
                        (name, json.dumps({"period": f.period,
                                           "areas": list(f.areas)})))
        return True
    except psycopg2.errors.InsufficientPrivilege:
        st.info(READ_ONLY_NOTE)
        return False


def _delete(name: str) -> bool:
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM saved_views WHERE name = %s", (name,))
        return True
    except psycopg2.errors.InsufficientPrivilege:
        st.info(READ_ONLY_NOTE)
        return False


def bar() -> Filters:
    """Render the sidebar filters and return the current selection."""
    areas = all_areas()
    st.session_state.setdefault("period", DEFAULT_PERIOD)
    st.session_state.setdefault("areas", [])

    with st.sidebar:
        st.markdown("<div class='filter-head'>Filters</div>",
                    unsafe_allow_html=True)
        # `key` alone: session_state already holds the value (seeded by the
        # setdefault calls above), and passing index/default as well makes
        # Streamlit warn that the default will be ignored.
        period = st.selectbox("Period", list(PERIODS), key="period")
        picked = st.multiselect("Business area", areas,
                                placeholder="All areas", key="areas")

        current = Filters(period, tuple(picked), PERIODS[period])

        saved = _views()
        with st.popover("Saved views", width="stretch"):
            if saved.empty:
                st.caption("No saved views yet.")
            for row in saved.itertuples():
                payload = row.payload if isinstance(row.payload, dict) \
                    else json.loads(row.payload)
                c1, c2 = st.columns([5, 1], vertical_alignment="center")
                with c1:
                    if st.button(row.name, key=f"load{row.name}",
                                 width="stretch"):
                        st.session_state["period"] = payload["period"]
                        st.session_state["areas"] = payload["areas"]
                        st.rerun()
                    st.caption(f"{payload['period'].lower()} · "
                               f"{', '.join(payload['areas']) or 'all areas'}")
                with c2:
                    if st.button("Remove", key=f"del{row.name}"):
                        if _delete(row.name):
                            st.cache_data.clear()
                            st.rerun()
            st.divider()
            name = st.text_input("Name this view",
                                 placeholder="e.g. Delivery, last 4 weeks")
            if st.button("Save current filters", type="primary",
                         width="stretch", disabled=not name):
                if _save(name.strip(), current):
                    st.cache_data.clear()
                    st.rerun()

        st.markdown(f"<div class='sidenote'>Swiggy · Google Play<br>"
                    f"Showing {current.label}</div>", unsafe_allow_html=True)

    return current
