"""Setup and caching shared by every dashboard page.

Streamlit re-runs the whole script on every interaction, so anything expensive
must be cached or it happens again on each keystroke. The two caches differ in a
way that matters:

    @st.cache_data      memoises a returned *value*. Streamlit serialises it and
                        hands each caller its own copy, so a page mutating the
                        result cannot corrupt what another page sees. Right for
                        DataFrames from SQL and CSV.

    @st.cache_resource  holds one *live object*, shared by every session, never
                        copied and never serialised. Right for things that cannot
                        be pickled and should exist once per process: the
                        sentence encoder, the FAISS index, the cross-encoder.

Using cache_data on a model would try to serialise it — slow at best, an error at
worst. Using cache_resource on a DataFrame hands every page the same object, so
one page sorting it in place changes it underneath another.
"""

import sys
from pathlib import Path

# Streamlit sets sys.path[0] to the *entry script's* directory, so `src` is
# invisible from app/ and `shared` is invisible from app/pages/ when a page is
# run directly. Adding both makes every page work either way.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in (str(APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# faiss must load before scikit-learn: both link their own OpenMP runtime and on
# macOS the wrong order segfaults the process at the first faiss call (exit 139,
# no traceback). Importing it here means every page inherits the right order.
# See results/phase6_notes.md.
import faiss  # noqa: E402,F401
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.db.connection import connection  # noqa: E402

MODEL = "sbert-domain"          # the model whose themes the dashboard shows
RESULTS = PROJECT_ROOT / "results"

# Phase 1 recorded that filtering short reviews raises the 1-star share, so
# rating and volume figures must come from all 100,000 rows while themes use the
# 64,280 with enough text. Pages state both rather than implying all were themed.
ALL_REVIEWS = "app = 'swiggy'"
THEMED_REVIEWS = "app = 'swiggy' AND keep_for_themes"


@st.cache_data(ttl=600, show_spinner=False)
def sql(query: str, params: tuple | None = None) -> pd.DataFrame:
    """Run a query and cache the DataFrame, keyed on the query text and params."""
    with connection() as conn:
        return pd.read_sql(query, conn, params=params)


# ── the two windows every screen answers for ────────────────────────────────
# The briefing reports one real week; the fix list reports eight, because a
# single week of this corpus puts only ~47 reviews behind the biggest problem
# and shows most of them falling. Both are stated on screen rather than implied.
WEEKS_ON_FIX_LIST = 8


@st.cache_data(ttl=600, show_spinner=False)
def week_start():
    """Monday of the last COMPLETE week. theme_weekly already drops partial
    weeks when it is built, so its maximum is the right anchor — using
    max(reviewed_at) instead would land mid-week and undercount."""
    with connection() as conn:
        return pd.read_sql("SELECT max(week_start) AS w FROM theme_weekly "
                           "WHERE model = 'sbert-domain'", conn).w.iloc[0]


def in_week(alias: str = "r") -> str:
    """SQL fragment: reviews inside the last complete week."""
    start = week_start()
    return (f" AND {alias}.reviewed_at >= '{start}'"
            f" AND {alias}.reviewed_at < '{start}'::date + 7")


def in_window(alias: str = "r", weeks: int = WEEKS_ON_FIX_LIST) -> str:
    """SQL fragment: reviews inside the trailing window the fix list covers."""
    return (f" AND {alias}.reviewed_at >= (SELECT max(reviewed_at) FROM reviews"
            f" WHERE app='swiggy') - interval '{weeks * 7} days'")


@st.cache_data(ttl=600, show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    """Load a table from results/. Empty frame if the phase has not run yet."""
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner="Loading the search index…")
def get_search():
    """The FAISS index and sentence encoder — one copy per process.

    src.search.query already lru_caches these, so this wrapper mainly stops
    Streamlit re-entering the module on every rerun.
    """
    from src.search.query import embed_query, get_searcher
    bundle = get_searcher(MODEL, approximate=False)
    # One throwaway encode inside the cached call. The first query on MPS pays
    # kernel compilation — measured at ~1.9 s against a warm 8.6 ms — and paying
    # it here means it lands in the "Loading…" spinner instead of being displayed
    # to the user as the query time. The Phase 6 benchmark discards this same
    # warm-up rather than averaging it in.
    embed_query("warm up", bundle[2])
    return bundle


@st.cache_resource(show_spinner="Loading the reranking model…")
def get_reranker():
    from src.search.rerank import MODEL as CROSS_MODEL, get_cross_encoder
    return get_cross_encoder(CROSS_MODEL)
