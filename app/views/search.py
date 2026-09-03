"""Search — find reviews by what they mean, not the words they use."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import design
import shared
from shared import st

design.appbar("Understand", "Search",
              right="Searching <b>45,864</b> reviews")

EXAMPLES = ["the app keeps crashing", "my refund never arrived",
            "delivery was later than promised", "I was charged extra"]

c1, c2 = st.columns([5, 1], gap="medium")
with c1:
    query = st.text_input("Search", placeholder="Type what a customer might say",
                          label_visibility="collapsed")
with c2:
    k = st.selectbox("Results", [10, 25, 50], label_visibility="collapsed",
                     format_func=lambda n: f"Top {n}")

careful = st.toggle("More careful search — slower, finds better matches",
                    value=False)

if not query:
    design.hero(
        eyebrow="Search by meaning",
        headline="Find reviews that mean the same thing, in different words",
        value="45,864",
        unit="reviews searched at once",
        side="Try:<br>" + "<br>".join(f"<b>{e}</b>" for e in EXAMPLES[:3]))
    st.stop()

shared.get_search()
if careful:
    shared.get_reranker()

start = time.perf_counter()
if careful:
    from src.search.rerank import search_reranked
    hits = search_reranked(query, k=int(k)).rename(columns={"cross_score": "score"})
else:
    from src.search.query import search
    hits = search(query, k=int(k))
elapsed = (time.perf_counter() - start) * 1000

best = hits.iloc[0]
design.hero(
    eyebrow=f"Closest match for “{query}”",
    headline=f"“{' '.join(str(best.content).split())[:150]}”",
    value=f"{len(hits)}",
    unit="matching reviews",
    side=(f"Found in <b>{elapsed:.0f} ms</b><br>"
          + ("Checked twice for accuracy" if careful else "Fast search")))

st.markdown("## All matches")

table = hits[["rank", "content"]].copy()
if not careful:
    table["match"] = hits.score.clip(0, 1)
    cfg = {"rank": st.column_config.NumberColumn("#", width="small"),
           "match": st.column_config.ProgressColumn(
               "How close", min_value=0.0, max_value=1.0),
           "content": "Review"}
    table = table[["rank", "match", "content"]]
else:
    cfg = {"rank": st.column_config.NumberColumn("#", width="small"),
           "content": "Review"}

st.dataframe(table, hide_index=True, width="stretch", height=440,
             column_config=cfg)

if careful:
    moved = int((hits.faiss_rank != hits["rank"]).sum())
    design.note(f"The careful search moved {moved} of these {len(hits)} results, "
                "promoting reviews the fast search had ranked lower.")

with st.expander("Where this struggles"):
    st.markdown(
        "Hindi and Hinglish reviews can come back with the opposite meaning — "
        "searching for *food was cold* may return *food was very good*. "
        "About 4% of reviews are affected.")
