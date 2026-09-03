"""Search — find reviews by meaning, not by keyword.

Wraps the Phase 6 search stack. The point of the page is that it finds reviews
that share no words with the query: searching "app keeps crashing" should return
"closes by itself".

The two-stage toggle exposes the measured trade rather than hiding it. Numbers
from results/search_benchmark.csv:

    single-stage   8.56 ms p50   61.15 Precision@10
    two-stage     71.26 ms p50   75.77 Precision@10

Both models are held by @st.cache_resource, so they load once per process. Without
it Streamlit would reload roughly 400 MB of weights on every keystroke, because
every interaction re-runs this file from the top.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared
from shared import st

shared.page("Search", "🔎", "Semantic search over 45,864 distinct review texts.")

EXAMPLES = ["app keeps crashing", "my refund never arrived",
            "delivery was much later than promised", "charged extra fees",
            "khana thanda tha"]

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("Search reviews by meaning",
                          placeholder="e.g. the order arrived cold",
                          label_visibility="collapsed")
with col2:
    k = st.number_input("Results", 5, 50, 10, step=5)

st.caption("Try: " + " · ".join(f"`{e}`" for e in EXAMPLES))

two_stage = st.toggle(
    "Two-stage: rerank with a cross-encoder",
    value=False,
    help="Stage one (FAISS) narrows 45,864 reviews to 50. Stage two rescores "
         "those 50 with a model that reads the query and review together. "
         "+14.62 Precision@10, ~8x slower — measured, not estimated.")

if not query:
    st.info("Type a query above. Results are ranked by cosine similarity, where "
            "1.00 means identical in meaning and 0 means unrelated.")
    st.stop()

# Touch the cached resources first so the spinner appears here rather than
# mid-search, and so the timing below measures the query rather than the load.
shared.get_search()
if two_stage:
    shared.get_reranker()

start = time.perf_counter()
if two_stage:
    from src.search.rerank import search_reranked
    hits = search_reranked(query, k=int(k))
    hits = hits.rename(columns={"cross_score": "score"})
    extra = ["faiss_rank"]
else:
    from src.search.query import search
    hits = search(query, k=int(k))
    extra = []
elapsed = (time.perf_counter() - start) * 1000

m1, m2, m3 = st.columns(3)
m1.metric("Results", len(hits))
m2.metric("Query time", f"{elapsed:.0f} ms")
m3.metric("Mode", "Two-stage" if two_stage else "Single-stage",
          help="Measured p50: 8.56 ms single-stage, 71.26 ms two-stage.")

cols = ["rank", "score"] + extra + ["content"]
st.dataframe(
    hits[cols], hide_index=True, width="stretch", height=460,
    column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "score": st.column_config.NumberColumn(
            "Score", format="%.3f",
            help="Cosine similarity for single-stage; cross-encoder logit for "
                 "two-stage, which is unbounded and not a similarity."),
        "faiss_rank": st.column_config.NumberColumn(
            "Was #", width="small",
            help="Where stage one ranked this before reranking."),
        "content": "Review",
    })

if two_stage:
    moved = (hits.faiss_rank != hits["rank"]).sum()
    st.caption(f"The cross-encoder moved **{moved} of {len(hits)}** results from "
               "where FAISS had them. The 'Was #' column shows the original rank — "
               "a result promoted from #40 is one a keyword search would never "
               "have surfaced.")
else:
    st.caption("Cosine similarity: 1.00 is identical in meaning, 0 is unrelated. "
               "Reviews with no words in common with your query can still score "
               "highly — that is the entire point.")

with st.expander("Why is searching Hinglish worse?"):
    st.markdown("""
Try `khana thanda tha` (*the food was cold*). The top result is often
`kahana bht acha tha` (*the food was very good*) — the **opposite** meaning.

The model matches Hinglish by surface form rather than meaning, because it never
saw Hinglish paired with its English equivalent during training. Phase 4
established this and Phase 5 shows the consequence: the single largest theme
groups Hinglish reviews *by language* rather than by what they complain about.

It is a real limitation of this model, reported rather than patched around.
""")
