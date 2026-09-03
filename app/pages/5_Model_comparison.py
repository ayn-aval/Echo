"""Model comparison — the evidence, without reading the code.

Presentation only. Nothing is computed here; every table is read from results/,
each written by a named script in eval/. The caveats shown alongside are the ones
already recorded in the phase notes, and they are on the page rather than in a
footnote because several of these numbers mislead when quoted alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared
from shared import csv, st

shared.page("Model comparison", "🧪",
            "Does any of this work? Every number below comes from a script in "
            "`eval/` and is saved in `results/`.")

tabs = st.tabs(["Sentence embeddings (STS)", "Review retrieval",
                "Theme quality", "Search performance", "Ablations"])

# ─────────────────────────────── STS ────────────────────────────────
with tabs[0]:
    st.subheader("Reproducing Sentence-BERT, Table 1")
    t1 = csv("table1_comparison.csv").rename(columns={"Unnamed: 0": "model"})
    st.dataframe(t1, hide_index=True, width="stretch")
    st.markdown("""
**The paper's central motivation reproduces.** Averaged GloVe (53.88) beats
mean-pooled BERT (52.64) — raw BERT embeddings are worse than word vectors from
2014. That is the problem Sentence-BERT was written to solve.

**Our reproduction reached 72.17 against the paper's 74.21**, a 2.04 gap on 32%
of its training data.
""")
    st.info("**Honest note.** Our GloVe baseline is 7.4 points below the paper's "
            "(53.88 vs 61.32) because we compute one Spearman over all pairs "
            "while SentEval averages per sub-dataset. So any \"improvement over "
            "GloVe\" we quote is inflated — the paper's own margin of +12.89 is "
            "the right sanity check.")

    st.subheader("After domain adaptation")
    trained = csv("sts_trained.csv")
    if not trained.empty:
        wide = trained.pivot(index="model", columns="dataset", values="spearman")
        order = [c for c in ["STS12", "STS13", "STS14", "STS15", "STS16",
                             "STS-B", "SICK-R", "Avg"] if c in wide.columns]
        st.dataframe(wide[order].reset_index(), hide_index=True, width="stretch")
    st.markdown("""
Fine-tuning on Swiggy reviews **raised** generic STS from 72.17 to **74.54**,
six of seven datasets improving. This was predicted to fall. Contrastive training
spreads out the embedding space — the same anisotropy problem behind BERT's 52.64
— so training on app reviews fixed a defect that had nothing to do with Swiggy.
""")
    st.warning("**Do not read this as beating the paper.** 74.54 exceeds the "
               "paper's 74.21, but their number comes from NLI training alone "
               "and ours adds a second stage they never ran. Different recipe, "
               "not a better result on the same one.")

# ───────────────────────────── Retrieval ─────────────────────────────
with tabs[1]:
    st.subheader("Finding relevant reviews — 26 hand-judged queries")
    base = csv("baselines.csv")
    ret = base[base.task == "retrieval"] if "task" in base.columns else base
    if not ret.empty:
        wide = ret.pivot_table(index="model", columns="dataset", values="score")
        keep = [c for c in ["recall@10", "precision@10", "mrr"] if c in wide.columns]
        st.dataframe(wide[keep].reset_index(), hide_index=True, width="stretch")

    bench = csv("search_benchmark.csv")
    acc = bench[bench.measure == "accuracy"] if "measure" in bench.columns else bench
    if not acc.empty:
        st.markdown("**With the Phase 6 search stack:**")
        st.dataframe(acc[["config", "recall@10", "precision@10", "mrr"]],
                     hide_index=True, width="stretch")
    st.markdown("""
TF-IDF won this benchmark from Phase 2 until Phase 6. Simple word matching is a
genuinely strong baseline on short reviews, and saying so is more useful than
pretending otherwise.

**Two-stage search finally beats it: 75.77 Precision@10 against TF-IDF's 65.00.**
Domain adaptation took the single-stage model from 45.77 to 61.15; reranking
added the rest.
""")
    st.info("**A number this project almost reported wrong.** The first "
            "measurement said reranking *dropped* precision to 40.77. Only 47.3% "
            "of the reranker's results had ever been judged, against 96.2% for "
            "FAISS — it surfaces reviews from ranks 11–50 that nothing else ever "
            "pooled, and unjudged counts as irrelevant. After re-pooling and 127 "
            "more judgements, both sit at 96.2% and the real answer appeared.")

# ─────────────────────────── Theme quality ───────────────────────────
with tabs[2]:
    st.subheader("Do the themes make sense?")
    clus = csv("clustering_comparison.csv")
    st.dataframe(clus, hide_index=True, width="stretch")
    st.markdown("""
The same clustering pipeline run on three embeddings, so only the model varies.
**The blind hand-audit is the measure that counts**: a person reads a review and
the theme it was placed in, without being told which model produced it.
`sbert-domain` scored **82.4%** against GloVe's 44.1%.
""")
    st.error("**Never quote the noise column on its own — it ranks the worst "
             "model first.** GloVe posts the best noise figure (15.04%) by "
             "refusing to separate anything: 58.6% of the corpus lands in one "
             "cluster mixing five-star praise with refund complaints. Read it "
             "with `largest_pct`.")
    st.warning("**`sbert-domain` vs `bert-mean` is not statistically "
               "significant.** 82.4% vs 73.5% at 34 judgements each gives "
               "p=0.56 by Fisher exact. The case for the trained model rests on "
               "structure instead — 110 themes against 42, and 5.88% against "
               "41.15% of the corpus in the largest theme — which is not a "
               "sampling question.")
    st.caption("Silhouette is computed inside each model's own embedding space, "
               "which differ in geometry and dimensionality. It is a "
               "within-model diagnostic and is **not** comparable across models.")

# ──────────────────────── Search performance ─────────────────────────
with tabs[3]:
    st.subheader("Latency, p50 and p95 over 200 queries")
    bench = csv("search_benchmark.csv")
    lat = bench[bench.measure == "latency"] if "measure" in bench.columns else bench
    if not lat.empty:
        st.dataframe(lat[["stage", "p50_ms", "p95_ms", "n"]], hide_index=True,
                     width="stretch")
    st.markdown("""
**Single-stage search answers in 8.56 ms.** Reranking costs 8.3x for +14.62
Precision@10 and is still interactive at 71 ms.

**FAISS is not the bottleneck.** Exhaustive search over 45,864 vectors takes
1.52 ms while *encoding the query* takes 6.89 ms.

**Why not cross-encode every review?** A bi-encoder embeds the query and each
review separately, so review vectors are computed once, offline. A cross-encoder
scores a *pair*, so it precomputes nothing. 50 candidates cost 61.20 ms, so all
45,864 would cost roughly **56 seconds per query** — about 6,500x slower. That is
the Sentence-BERT paper's opening argument, measured on this corpus.
""")
    st.caption("First 10 queries discarded as warm-up: the first call pays model "
               "load and MPS kernel compilation. Single-query timings on an idle "
               "Mac — no concurrency, no network.")

# ───────────────────────────── Ablations ─────────────────────────────
with tabs[4]:
    st.subheader("The paper's Table 6 ablation")
    ab = csv("ablation.csv")
    st.dataframe(ab, hide_index=True, width="stretch")
    st.markdown("""
**Claim 1 — `|u−v|` is the critical component. HOLDS.** `(u,v)` 52.98 →
`(u,v,|u−v|)` 68.18, a margin of **+15.20 against the paper's +14.74**. Every
configuration containing `|u−v|` scores 62.22–70.93; every one without it scores
52.98–60.38, with **no overlap**.

**Claim 2 — adding `u*v` hurts. DOES NOT HOLD for us.** +2.75 where the paper
reports −0.34 — eight times the paper's margin and opposite in sign. Hypothesis,
not measurement: at 100k pairs the model is undertrained and richer features still
help. One seed per configuration, so variance cannot be fully excluded. Reported
as a finding to explain, never as a correction to the paper.
""")
    st.subheader("Are Swiggy's replies a usable training signal?")
    reply = csv("phase4_reply_signal.csv")
    st.dataframe(reply, hide_index=True, width="stretch")
    st.markdown("""
The project plan assumed replies were templated by complaint category, which
would make two reviews sharing a reply a free positive pair. **Measured, they are
templated by star rating and nothing else.** Hold the rating fixed and a
classifier cannot predict which of 23 one-star templates was sent (8.12% against
an 8.63% majority baseline). The strategy was dropped before any training ran.
""")
