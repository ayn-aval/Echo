"""Search — find reviews by what they mean."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import design
import shared
from shared import st

design.appbar("Understand", "Find reviews")

EXAMPLES = ["the app keeps crashing", "my refund never arrived",
            "delivery was later than promised", "I was charged extra"]

c1, c2 = st.columns([5, 1], gap="medium")
with c1:
    query = st.text_input("Search", placeholder="e.g. the order arrived cold",
                          label_visibility="collapsed")
with c2:
    k = st.selectbox("Results", [10, 25, 50], label_visibility="collapsed")

design.note("Try: " + "  ·  ".join(EXAMPLES))

careful = st.toggle(
    "More careful search (slower, more accurate)",
    value=False,
    help="Reads your search and each review together, one pair at a time, "
         "instead of comparing them at a distance. Finds noticeably better "
         "matches and takes about a tenth of a second instead of instantly.")

if not query:
    st.markdown(
        f"<div class='card' style='color:{design.INK_2};'>"
        "Type what a customer might say. Reviews that mean the same thing come "
        "back, even when they use completely different words.</div>",
        unsafe_allow_html=True)
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

design.tiles([
    ("Reviews searched", "45,864", "every distinct review"),
    ("Matches shown", f"{len(hits)}", "ranked by how closely they match"),
    ("Time taken", f"{elapsed:.0f} ms",
     "more careful search" if careful else "instant search"),
])

best = hits.iloc[0]
st.markdown(
    f"<div class='card'><div style='color:{design.MUTED};font-size:.75rem;"
    f"text-transform:uppercase;letter-spacing:.06em;'>Closest match</div>"
    f"<div style='margin-top:10px;font-size:1.05rem;color:{design.INK};"
    f"line-height:1.55;'>“{str(best.content)[:400]}”</div></div>",
    unsafe_allow_html=True)

st.markdown("## All matches")

table = hits[["rank", "content"]].copy()
if not careful:
    table["match"] = hits.score.clip(0, 1)
    cfg = {"rank": st.column_config.NumberColumn("#", width="small"),
           "match": st.column_config.ProgressColumn(
               "Match", min_value=0.0, max_value=1.0,
               help="How close in meaning, from 0 to 1."),
           "content": "Review"}
    table = table[["rank", "match", "content"]]
else:
    cfg = {"rank": st.column_config.NumberColumn("#", width="small"),
           "content": "Review"}

st.dataframe(table, hide_index=True, width="stretch", height=440,
             column_config=cfg)

if careful:
    moved = int((hits.faiss_rank != hits["rank"]).sum())
    design.note(f"The careful search reordered {moved} of these {len(hits)} "
                "results, promoting reviews the fast search had ranked lower.")

with st.expander("Where this struggles"):
    st.markdown(
        "Hindi and Hinglish reviews can return the opposite meaning — searching "
        "for *food was cold* may return *food was very good*. About 4% of "
        "reviews are affected."))
