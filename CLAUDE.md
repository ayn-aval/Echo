# Echo — customer feedback intelligence from app reviews

Full phased plan: `docs/PROJECT_PLAN.md`. Read it before starting any new phase.
Current phase and progress: `docs/PROGRESS.md`. Read it at the start of every session.

## Who you're working with

I am a final-year engineering student applying for Data Science and Data Analyst roles.
I know Python, SQL (MySQL/PostgreSQL), and Power BI. I have **no software development
background** — no web dev, no Docker, no cloud, no prior git experience.

Explain before you code. When you introduce a library, tool, or concept, give me 3–4
plain-language sentences on what it is and why we're using it. When there's a simple way
and a "proper" way, take the simple way and name the tradeoff.

## Working rules

- Use plan mode for anything spanning more than one file. Show me the plan, wait for approval.
- Work in small steps. After each meaningful unit, stop, tell me how to run it and what
  output to expect, and wait for me to confirm before continuing.
- Never write more than ~80 lines of new code without pausing to explain it.
- Never start the next phase without me explicitly saying so.
- Ask when you need a decision. Do not silently assume.
- If a step is likely to fail on my machine, warn me first with the fix.
- Never run destructive commands (`DROP`, `rm -rf`, force push, migrations that lose data)
  without asking. Never commit secrets or data files.

## Hardware

- Local: **Mac with Apple Silicon**. PyTorch uses the `mps` backend, not CUDA. Always route
  device selection through `src/utils/device.py` (mps → cuda → cpu). Set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`. Do not use `fp16` locally.
- Training (Phases 3 and 4 only): **Google Colab free tier, T4**. Sessions disconnect —
  checkpoint to Google Drive and support resume before any long run.
- Install `faiss-cpu`, not `faiss-gpu`.

## Stack — do not substitute without asking

Python · PostgreSQL · Streamlit (the entire UI — no React, no HTML/CSS/JS, no FastAPI) ·
PyTorch + HuggingFace Transformers · sentence-transformers (Phase 4 onward only) ·
FAISS · HDBSCAN + UMAP · pandas / numpy / scikit-learn / plotly · google-play-scraper

Environment: `venv` + `requirements.txt`. No Docker, no Poetry, no Conda.

## Layout

```
src/ingest/       scraping, loading to Postgres
src/embeddings/   encoding, model loading
src/clustering/   theme discovery and labelling
src/search/       FAISS index and query
src/db/           connection, schema, queries
src/utils/        device detection, config, logging
app/              Streamlit pages
eval/             evaluation scripts — every reported metric comes from here
notebooks/        Colab training notebooks
results/          metrics as CSV, plots as PNG — committed
data/ models/     gitignored
docs/             PROJECT_PLAN.md, PROGRESS.md
```

## Commands

```bash
source venv/bin/activate
streamlit run app/main.py
python -m eval.run_sts          # STS benchmark evaluation
python -m eval.run_retrieval    # review retrieval evaluation
```

## Non-negotiables

- **Phase 3 must use raw PyTorch.** Write the siamese training loop by hand. Do not use
  `sentence-transformers` in Phase 3 — reproducing the paper is the point of the project.
  If you think the library is easier here, you are right, and it is still not allowed.
- Never skip a phase's evaluation step to move faster.
- Every reported metric must be reproducible from a script in `eval/` and saved to `results/`.
- If a result looks surprisingly good, check for data leakage before we believe it.
- Report honest numbers. If my results fall short of the paper, we explain why — we do not
  quietly tune until they match.

## End of session

Before we stop, update `docs/PROGRESS.md` with: what we completed, what's in flight,
any decision I made and why, and the exact next step.
