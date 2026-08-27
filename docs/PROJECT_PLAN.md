# Project brief for coding agent — "Pulse": customer feedback intelligence from app reviews

## How I want you to work with me

Read this section first and follow it for the entire project.

- I am **not** a software developer. I know Python, SQL (MySQL and PostgreSQL), and Power BI. I do **not** know web development, Docker, cloud deployment, or software architecture. Assume no prior knowledge of those.
- Work **one phase at a time**. Do not jump ahead. At the end of each phase, stop and wait for me to confirm before starting the next.
- Within a phase, work in **small steps**. Give me one file or one function at a time, explain what it does in plain language, tell me exactly how to run it and what output I should expect. Then wait.
- Every time you introduce a new library, tool, or concept, explain in 3–4 sentences what it is and why we're using it before showing code.
- If something can be done in two ways — one simple and one "proper" — default to the simple one and tell me what the tradeoff is.
- If a step could fail on my machine, tell me the likely error and the fix in advance.
- Ask me questions when you need a decision. Do not silently assume.
- Never give me more than ~80 lines of code in one message without stopping to explain it.

## What we are building

A web dashboard that reads tens of thousands of unstructured app-store reviews and automatically turns them into a small set of tracked **themes** — what people are complaining about, how many said it, and which themes are getting worse.

Keyword counting cannot do this, because people describe the same problem in completely different words ("app keeps crashing" / "closes by itself" / "shuts down when I open it"). The system groups by **meaning**, not words.

The meaning-matching is done by a sentence embedding model that I will train myself, reproducing the paper **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019)**. That reproduction is a core part of the project — the whole product exists to demonstrate it working on a real problem.

This is a portfolio project for Data Science / Data Analyst job applications. So alongside working code, I need **measurable results**, honest evaluation, and a README that a hiring manager can understand in two minutes.

## Tech stack (fixed — do not substitute without asking)

- **Python** — main language
- **PostgreSQL** — all raw and processed data lives here
- **Streamlit** — the entire user interface. No React, no HTML/CSS, no JavaScript, no FastAPI unless I explicitly ask.
- **PyTorch + HuggingFace Transformers** — for training the model in Phase 3
- **sentence-transformers** — allowed from Phase 4 onward only (see the rule in Phase 3)
- **FAISS** (CPU version) — vector similarity search
- **HDBSCAN + UMAP** — clustering
- **pandas, numpy, scikit-learn, matplotlib/plotly** — analysis and charts
- **google-play-scraper** — data collection
- Environment: plain `venv` + `requirements.txt`. No Docker, no Poetry, no Conda.

### My actual hardware — plan around this

- **Local machine: a Mac with an Apple Silicon GPU.** PyTorch reaches it through the `mps` backend, not CUDA. Write a single device-detection helper used everywhere: `mps` if available, else `cuda`, else `cpu`. Some operations are unsupported on MPS, so set `PYTORCH_ENABLE_MPS_FALLBACK=1` and tell me when a fallback is silently making things slow. Mixed precision (`fp16`) is unreliable on MPS — leave it off locally.
- **Google Colab free tier (T4 GPU)** for the two training runs only: Phase 3 and Phase 4.
- Everything else — scraping, encoding the review corpus, clustering, FAISS, Streamlit — runs on the Mac. Encoding a few hundred thousand short reviews on MPS is fine; batch it and show me a progress bar.

Colab free sessions disconnect. For any training run: mount Google Drive, checkpoint every N steps to Drive, and make the script resume from the last checkpoint. Set this up **before** the first long run, not after one dies.

Install `faiss-cpu` on the Mac, not `faiss-gpu`. If any library refuses to install on Apple Silicon, tell me the alternative rather than making me fight it.

## Repository structure

Set this up in Phase 0 and stick to it:

```
pulse/
  data/                 # local files, gitignored
  notebooks/            # Colab notebooks for training
  src/
    ingest/             # scraping and loading into Postgres
    embeddings/         # encoding, model loading
    clustering/         # theme discovery and labelling
    search/             # FAISS index build and query
    db/                 # connection, schema, queries
  app/                  # Streamlit application
  eval/                 # evaluation scripts and results
  models/               # saved model checkpoints, gitignored
  results/              # metrics tables, plots, saved as CSV/PNG
  README.md
  requirements.txt
```

---

# The phases

## Phase 0 — Setup

Goal: a working skeleton before any real work.

- Create the folder structure above, a virtual environment, `requirements.txt`, and `.gitignore`
- Set up PostgreSQL locally; create the database and a `config.py` / `.env` for credentials
- Write `src/db/connection.py` with a tested connection function
- Initialise git, first commit
- A `hello world` Streamlit page that reads one row from Postgres and displays it

**Done when:** I can run `streamlit run app/main.py` and see data from my own database on screen.

Teach me the git commands as we go. Assume I have never used git.

## Phase 1 — Data collection

Goal: a real, messy dataset in Postgres.

- Scrape reviews for **Swiggy** (`in.swiggy.android`) using `google-play-scraper`. Target 100,000–200,000 reviews. Optionally also scrape **Zomato** (`com.application.zomato`) as a second app — a competitor comparison view is a strong dashboard feature later, so design the schema with an `app` column from the start even if I only load one.
- Sort by newest and page through; the scraper is rate-limited, so write it to save incrementally and resume if interrupted. Do not lose four hours of scraping to one timeout.
- Design and create the Postgres schema. At minimum: review id, app name, review text, star rating, review date, thumbs-up count, app version, ingestion timestamp. Add indexes on date and rating.
- Write the loader with de-duplication, so re-running does not create duplicate rows
- Handle the practical problems: reviews in Hindi or Hinglish, one-word reviews ("good", "nice"), emoji-only reviews, extremely long reviews. Decide with me what to keep and what to filter, and record those rules.
- Produce a short exploratory analysis: review volume over time, rating distribution, length distribution, language mix. Save the plots to `results/`.

**Done when:** the database holds cleaned reviews and I can describe the dataset in three sentences.

## Phase 2 — Baselines and the evaluation harness

Build the measuring stick **before** building the model. This is important — do not skip it.

- Download the STS Benchmark and the STS12–STS16 datasets
- Write one evaluation function: given any "encode a list of sentences" function, compute Spearman rank correlation against the gold similarity scores
- Evaluate three baselines with it:
  1. Averaged GloVe embeddings
  2. Mean-pooled `bert-base-uncased` output
  3. The `[CLS]` token from `bert-base-uncased`
- Build a **retrieval evaluation set** from my own review data: I will hand-pick ~50 queries and mark which reviews are relevant. Guide me through doing this efficiently. Write the scorer for Recall@10 and MRR.
- Save every result to `results/baselines.csv`

**Expected finding:** raw BERT embeddings score *worse* than GloVe. If that's what we see, it reproduces the paper's central motivation. Flag it clearly.

## Phase 3 — Reproduce SBERT

This is the research core. **Write the siamese training loop yourself in PyTorch — do not use the `sentence-transformers` library in this phase.** Using it here would remove the entire point.

- Download SNLI + MultiNLI
- Build the siamese architecture: one shared BERT encoder, mean pooling over token outputs, then the classification head on the concatenation `(u, v, |u−v|)` into a 3-way softmax
- Training config from the paper: 1 epoch, batch size 16, Adam, learning rate 2e-5, linear warmup over the first 10% of steps
- Evaluate on all seven STS datasets using the Phase 2 harness. Compare my numbers to the paper's Table 1 in a side-by-side table.
- Then run the **ablation study** from the paper's Table 6: pooling strategies (MEAN vs MAX vs CLS), and the concatenation variants — `(u,v)`, `(|u−v|)`, `(u*v)`, `(|u−v|, u*v)`, `(u,v,u*v)`, `(u,v,|u−v|)`, `(u,v,|u−v|,u*v)`. The paper finds `|u−v|` is the critical component and that adding `u*v` hurts. Check whether that holds for me.
- Save checkpoints and all metrics

**Model size and training budget.** A full epoch over all ~940k SNLI+MultiNLI pairs at batch size 16 is far too long for a free Colab session. Do this instead:

1. Develop and debug with `all-MiniLM-L6-v2`'s base architecture (6 layers) on a 50k-pair subset until the loop is definitely correct.
2. Run the real reproduction with `distilroberta-base` or `bert-base-uncased` on a stratified subset of roughly 200k–300k pairs, with checkpointing to Drive.
3. Report the subset size honestly in the results table — training on less data than the paper is a legitimate constraint, and stating it is better than implying a full replication.

Increase batch size as far as T4 memory allows and scale the learning rate accordingly; tell me what you chose and why. Numbers will not match the paper exactly. Help me write an honest paragraph explaining the gap, distinguishing "less training data" from "a bug in my implementation" — and help me rule out the second before blaming the first.

**Done when:** I have a trained encoder that clearly beats all three Phase 2 baselines, plus a completed ablation table.

## Phase 4 — Domain adaptation

Generic STS performance is not the same as performance on app reviews. Now specialise the model.

- Build training pairs from my own review data using weak supervision. Swiggy replies to a large share of its Play Store reviews, and those replies are templated by complaint category — a review paired with its reply, or two reviews that received near-identical replies, is a strong free positive signal. Other options: same app version plus same star rating within a narrow time window, or back-translation. Propose two or three strategies, note which gives the cleanest positives, and let me pick.
- Continue fine-tuning the Phase 3 model using `MultipleNegativesRankingLoss` (the library is allowed from here on)
- Re-run the **review retrieval** evaluation from Phase 2 and compare: baseline vs my SBERT vs domain-adapted SBERT
- Also check STS scores didn't collapse, and explain the tradeoff if they did

**Done when:** I have a three-way comparison table showing domain adaptation improves Recall@10 and MRR on my data.

## Phase 5 — Theme discovery

- Encode every review in the database with the final model; store the vectors (start with `.npy` files plus a review-id mapping; we can move to pgvector later if needed)
- Reduce dimensions with UMAP, then cluster with HDBSCAN. Tune `min_cluster_size` and `min_samples` together with me — explain what each one changes.
- Generate a readable name for each cluster from its top TF-IDF terms, plus pick the most representative review (the one nearest the centroid) as an example
- Persist themes and review-to-theme assignments back into Postgres
- **Critical comparison:** run this exact clustering pipeline three times — with GloVe, with plain BERT, and with my domain-adapted SBERT. Compare silhouette scores, the noise/unclustered percentage, and let me hand-audit 100 reviews for correctness. Save to `results/clustering_comparison.csv`.

**Done when:** the database contains named themes, and I have evidence that the trained model produces better ones.

## Phase 6 — Semantic search

- Build a FAISS index over the review vectors and save it to disk
- Write a query function: text in, ranked matching reviews out, with similarity scores
- Add a two-stage option: FAISS retrieves the top 50, then a cross-encoder reranks those 50. Explain to me why we don't just cross-encode everything, and measure the accuracy gain versus the latency cost.
- Benchmark and record p50 and p95 query latency over the full corpus

**Done when:** I can type a sentence and get semantically matching reviews back in well under a second, with recorded timings.

## Phase 7 — The Streamlit dashboard

Build it page by page. Show me one page working before starting the next.

1. **Overview** — total reviews, date range, average rating, rating distribution
2. **Themes** — ranked list of themes with counts, example reviews, average rating per theme; click one to drill into its reviews
3. **Trends** — theme volume over time, with filters for date range and app version. This is the most important page for a product manager.
4. **Search** — free-text box returning semantically matched reviews with scores
5. **Model comparison** — a page displaying the evaluation tables from Phases 2–5, so a visitor sees the proof without reading code

Use Streamlit caching (`@st.cache_data`, `@st.cache_resource`) so the model and index load once. Explain the difference between the two.

## Phase 8 — Trends and alerts

- Weekly theme volume time series stored in Postgres
- Emerging-theme detection: flag clusters that are new this period, or whose volume jumped beyond a threshold. Start with a simple statistical rule (z-score against the trailing mean) and explain it.
- Surface alerts on the dashboard
- Optional: since I know Power BI, help me connect it to the same Postgres tables and build one executive-style report as an extra portfolio artifact

## Phase 9 — Write-up and polish

- README with: the problem, a screenshot/GIF of the dashboard, architecture diagram, all results tables, how to run it, and honest limitations
- A short "reproduction notes" section comparing my STS numbers to the paper's
- A business-impact paragraph: translate the search latency and theme accuracy into a rough estimate of analyst hours saved, with assumptions stated explicitly
- Clean up the code, add docstrings, pin `requirements.txt`
- Optionally deploy to Streamlit Community Cloud (free) — walk me through it, assuming I've never deployed anything

---

## Non-negotiables

- Never skip the evaluation step in a phase to move faster
- If a result looks too good, help me check for data leakage before I believe it
- Every metric I report must be reproducible from a script in `eval/`
- Explain concepts before code, always

## Start here

Do not write any code yet. First, read this whole brief and tell me:

1. Anything that is unclear or that you'd change
2. Any risk you see given my experience level
3. Your plan for Phase 0 specifically, broken into numbered steps

Then wait for my go-ahead.
