"""Relevance labelling for the retrieval evaluation set.

    streamlit run app/label.py

One query per screen with every candidate as a checkbox, and a single save. That
is ~50 screens instead of ~1,000 individual decisions, which is the only reason
this is faster than editing a spreadsheet.

Two deliberate choices that protect the evaluation:

  * Which system found a candidate is NOT shown. Knowing that TF-IDF surfaced a
    review would bias the judgement toward the systems being measured.
  * Candidates are shuffled with a fixed per-query seed, so one system's top hit
    does not always sit at the top of the page and collect the attention.

Judgements are written to Postgres on each save, so closing the browser loses
nothing and reopening resumes at the first unjudged query.
"""

import random
import sys
from pathlib import Path

# Streamlit sets sys.path[0] to this file's directory, not the project root, so
# `src` is invisible unless we put the root on the path ourselves. Running via
# `python -m` adds the root automatically, which is why every other module in
# this project works without this. Any future page under app/ needs these lines.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.db.connection import connection

st.set_page_config(page_title="Echo — relevance labelling", layout="wide")


@st.cache_data(ttl=5)
def progress():
    with connection() as conn:
        return pd.read_sql("""
            SELECT q.query_id, q.query_text,
                   count(p.review_id)                       AS candidates,
                   count(j.review_id)                       AS judged,
                   count(*) FILTER (WHERE j.relevant)       AS relevant
              FROM eval_queries q
              LEFT JOIN eval_pool p ON p.query_id = q.query_id
              LEFT JOIN eval_judgements j
                     ON j.query_id = q.query_id AND j.review_id = p.review_id
             GROUP BY q.query_id, q.query_text
             ORDER BY q.query_id""", conn)


def candidates(query_id):
    with connection() as conn:
        df = pd.read_sql("""
            SELECT p.review_id, r.content, r.score, r.reviewed_at::date AS day
              FROM eval_pool p
              JOIN reviews r ON r.app = p.app AND r.review_id = p.review_id
             WHERE p.query_id = %s""", conn, params=(query_id,))
    order = list(df.index)
    random.Random(query_id).shuffle(order)   # fixed seed: same order every visit
    return df.loc[order].reset_index(drop=True)


def save(query_id, marks):
    with connection() as conn, conn.cursor() as cur:
        for review_id, relevant in marks.items():
            cur.execute("""
                INSERT INTO eval_judgements (query_id, app, review_id, relevant)
                VALUES (%s, 'swiggy', %s, %s)
                ON CONFLICT (query_id, app, review_id)
                DO UPDATE SET relevant = EXCLUDED.relevant, judged_at = now()
            """, (query_id, review_id, relevant))


state = progress()
done = state[state.judged > 0]
todo = state[(state.judged == 0) & (state.candidates > 0)]

with st.sidebar:
    st.header("Progress")
    st.metric("Queries labelled", f"{len(done)} / {len(state)}")
    st.progress(len(done) / max(len(state), 1))
    st.metric("Judgements made", int(state.judged.sum()))
    st.metric("Marked relevant", int(state.relevant.sum()))
    st.caption("Saved to Postgres on every submit. Close the tab whenever you "
               "like — reopening resumes here.")
    if len(done):
        st.divider()
        st.caption("Done so far")
        for _, r in done.iterrows():
            st.caption(f"{r.relevant}/{r.judged} · {r.query_text[:44]}")

if todo.empty:
    st.success("All queries labelled. Run `python -m eval.run_retrieval` next.")
    st.dataframe(state[["query_text", "candidates", "judged", "relevant"]],
                 use_container_width=True, hide_index=True)
    st.stop()

row = todo.iloc[0]
st.caption(f"Query {len(done) + 1} of {len(state)}")
st.subheader(row.query_text)
st.caption("Tick every review that a person searching this would want to see. "
           "When unsure, leave it unticked — a doubtful yes is worse than a "
           "confident no, because it inflates every model's score equally.")

pool = candidates(int(row.query_id))
with st.form(key=f"q{row.query_id}"):
    marks = {}
    for i, c in pool.iterrows():
        left, right = st.columns([1, 11])
        with left:
            marks[c.review_id] = st.checkbox("relevant", key=f"{row.query_id}_{i}",
                                             label_visibility="collapsed")
        with right:
            st.markdown(f"**{c.score}★** · {c.day}  \n{c.content}")
        st.divider()
    if st.form_submit_button("Save and go to next query", type="primary",
                             use_container_width=True):
        save(int(row.query_id), marks)
        st.cache_data.clear()
        st.rerun()
