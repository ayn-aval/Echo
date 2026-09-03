"""Echo — customer feedback intelligence from app reviews.

    streamlit run app/main.py

The landing page. Streamlit discovers app/pages/*.py automatically and builds the
sidebar from their filenames, so there is no navigation code here.

app/label.py and app/audit.py deliberately live outside pages/ — they are internal
annotation tools that write to the database, and they have no place in a
dashboard someone is reading.
"""

import shared
from shared import st

shared.page("Echo", "📣",
            "Turning 100,000 Swiggy Play Store reviews into tracked themes.")

st.markdown("""
Keyword counting cannot group app reviews, because people describe the same
problem in completely different words — *"app keeps crashing"*, *"closes by
itself"*, *"shuts down when I open it"*. This project groups them by **meaning**,
using a sentence embedding model reproduced from
**Sentence-BERT (Reimers & Gurevych, 2019)** and then adapted to this corpus.
""")

left, right = st.columns(2)
with left:
    st.subheader("The pages")
    st.markdown("""
| page | what it answers |
|---|---|
| **Overview** | How much data is there, and what does it look like? |
| **Themes** | What are people talking about, and how many said it? |
| **Trends** | What got worse after the last release? |
| **Search** | Show me every review about *this*, by meaning. |
| **Model comparison** | Does any of this actually work? |
""")

with right:
    st.subheader("Headline results")
    st.markdown("""
| result | number |
|---|---|
| STS reproduction vs the paper's 74.21 | **74.54** |
| Review retrieval, Precision@10 | **75.77** (TF-IDF: 65.00) |
| Theme quality, blind audit | **82.4%** (GloVe: 44.1%) |
| Search latency, p50 | **8.56 ms** |
""")
    st.caption("Every number is reproducible from a script in `eval/` and is "
               "shown with its caveats on the Model comparison page.")

st.divider()
shared.corpus_note()
