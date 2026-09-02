"""Blind hand-audit of theme assignments — the measure that settles Phase 5.

    streamlit run app/audit.py

Silhouette cannot compare models fairly: each is computed inside that model's own
embedding space, and those spaces have different geometry. So the deciding
evidence is a person reading a review, reading the theme it was placed in, and
saying whether it belongs.

Three things protect the verdict, mirroring app/label.py:

  * Which model produced an assignment is NEVER shown. Knowing a row came from
    the model we trained would quietly inflate its score.
  * The sample is drawn deterministically (md5 of review_id + model), so closing
    the browser and returning gives the same 100 rows rather than a fresh draw
    that could be cherry-picked.
  * Rows from all three models are interleaved and shuffled with a fixed seed.

There is no "skip" — an explicit yes or no is required — because a default answer
would bias every model's accuracy in the same direction.
"""

import random
import sys
from pathlib import Path

# Streamlit sets sys.path[0] to this file's directory, not the project root.
# Any page under app/ needs these two lines. See app/label.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.db.connection import connection

st.set_page_config(page_title="Echo — theme audit", layout="wide")

PER_MODEL = 34          # 3 models -> 102 judgements
PAGE = 12


@st.cache_data(ttl=5)
def sample():
    """A stable pseudo-random sample per model, excluding noise."""
    with connection() as conn:
        return pd.read_sql("""
            WITH ranked AS (
              SELECT rt.app, rt.review_id, rt.model, rt.theme_id,
                     r.content, t.label,
                     row_number() OVER (
                       PARTITION BY rt.model
                       ORDER BY md5(rt.review_id || rt.model)) AS rn
                FROM review_themes rt
                JOIN themes  t ON t.model = rt.model AND t.theme_id = rt.theme_id
                JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
               WHERE rt.theme_id >= 0
            )
            SELECT ranked.*, a.belongs
              FROM ranked
              LEFT JOIN theme_audit a
                     ON a.app = ranked.app AND a.review_id = ranked.review_id
                    AND a.model = ranked.model
             WHERE rn <= %s""", conn, params=(PER_MODEL,))


def save(marks):
    """marks: {(review_id, model): (theme_id, belongs)}"""
    with connection() as conn, conn.cursor() as cur:
        for (review_id, model), (theme_id, belongs) in marks.items():
            cur.execute("""
                INSERT INTO theme_audit (app, review_id, model, theme_id, belongs)
                VALUES ('swiggy', %s, %s, %s, %s)
                ON CONFLICT (app, review_id, model)
                DO UPDATE SET belongs = EXCLUDED.belongs, judged_at = now()
            """, (review_id, model, int(theme_id), belongs))


rows = sample()
if rows.empty:
    st.error("No themes in the database yet. Run:\n\n"
             "`python -m eval.clustering_comparison --persist`")
    st.stop()

# One fixed shuffle so the three models are interleaved, never grouped.
order = list(rows.index)
random.Random(20260903).shuffle(order)
rows = rows.loc[order].reset_index(drop=True)

done = rows.belongs.notna().sum()
todo = rows[rows.belongs.isna()]

with st.sidebar:
    st.header("Progress")
    st.metric("Judged", f"{done} / {len(rows)}")
    st.progress(done / max(len(rows), 1))
    st.caption("The model behind each assignment is hidden on purpose. "
               "Judge only whether the review fits the theme shown.")
    if done:
        with connection() as conn:
            acc = pd.read_sql("SELECT model, avg(belongs::int)*100 AS accuracy, "
                              "count(*) AS n FROM theme_audit GROUP BY model", conn)
        st.caption("Revealed once judged")
        st.dataframe(acc.round(1), hide_index=True, use_container_width=True)

if todo.empty:
    st.success("Audit complete. Run `python -m eval.clustering_comparison` to "
               "fold these into results/clustering_comparison.csv.")
    st.stop()

st.subheader("Does this review belong in this theme?")
st.caption("Judge the theme as a label for the review. If the theme is vague but "
           "not wrong, that still counts as belonging.")

batch = todo.head(PAGE)
with st.form("audit"):
    marks = {}
    for i, r in batch.iterrows():
        st.divider()
        left, right = st.columns([3, 1], vertical_alignment="center")
        with left:
            st.markdown(f"**Theme:** `{r.label}`")
            st.write(" ".join(str(r.content).split())[:400])
        with right:
            choice = st.radio("belongs?", ["Yes", "No"], key=f"a{i}",
                              index=None, horizontal=True,
                              label_visibility="collapsed")
        if choice is not None:
            marks[(r.review_id, r.model)] = (r.theme_id, choice == "Yes")
    if st.form_submit_button("Save this page", type="primary",
                             use_container_width=True):
        if len(marks) < len(batch):
            st.warning(f"{len(batch) - len(marks)} still unanswered — "
                       "every row needs a yes or no.")
        else:
            save(marks)
            st.cache_data.clear()
            st.rerun()
