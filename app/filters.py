"""The one global control: which period to look at.

This used to carry seven controls on every screen — period, business area, and a
saved-views popover with load, remove, name and save. Two of those are gone on
purpose.

**Business area** is now chosen by clicking a bar on Overview. A dropdown listing
areas that are already drawn on screen is a second way to do the same thing, and
the click is the better one.

**Saved views** is gone entirely. It cost five of the seven controls, and it could
never work on the deployed app anyway: that connects as a read-only Postgres role,
so every save was going to be refused.
"""

from dataclasses import dataclass

import streamlit as st

PERIODS = {"Last 4 weeks": 28, "Last 8 weeks": 56, "Last 3 months": 90,
           "Last 6 months": 180, "All time": None}
DEFAULT_PERIOD = "Last 3 months"

AREA_KEY = "area"          # set by clicking a bar on Overview; None means all


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


def current_area():
    return st.session_state.get(AREA_KEY)


def set_area(area) -> None:
    st.session_state[AREA_KEY] = area


def bar() -> Filters:
    """Render the sidebar and return the current selection."""
    st.session_state.setdefault("period", DEFAULT_PERIOD)

    with st.sidebar:
        st.markdown("<div class='filter-head'>Time period</div>",
                    unsafe_allow_html=True)
        # `key` alone: session_state already holds the value, and passing index as
        # well makes Streamlit warn that the default will be ignored.
        period = st.selectbox("Period", list(PERIODS), key="period",
                              label_visibility="collapsed")

    area = current_area()
    current = Filters(period, (area,) if area else (), PERIODS[period])

    with st.sidebar:
        st.markdown(f"<div class='sidenote'>Swiggy · Google Play<br>"
                    f"Showing {current.label}<br><br>"
                    f"Changes what <b>What to fix</b> counts. The rating charts "
                    f"always use all 100,000 reviews.</div>",
                    unsafe_allow_html=True)

    return current
