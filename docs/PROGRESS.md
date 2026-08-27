# Progress

## Current status

**Phase 1 — Data collection.** Scrape complete: **100,000 Swiggy reviews** in
Postgres, 18 Jan – 26 Aug 2026 (220 days), no duplicates. Remaining in this phase:
cleaning rules, then the exploratory analysis.

## Decisions made

| Decision | Why |
|---|---|
| GitHub account is `ayn-aval`, repo `ayn-aval/Echo` | Personal portfolio account, not the work identity in the global git config. Identity is set repo-locally so work repos are unaffected. |
| Project renamed Pulse → Echo | Match the repository name. |
| Storage stays PostgreSQL, not CSV | Considered CSV. Rejected: Phase 5 needs 4 joined tables, the scraper's resume needs transactional writes, and SQL on the CV matters for the roles being applied for. A CSV export helper will be added so data can still be pulled into Excel/Power BI on demand. |
| **Corpus capped at 100,000 reviews** (brief said 100k–200k) | User's call, 2026-08-27. Keeps scrape time and Phase 5 encoding time down. |
| Multi-stream scrape: 6 streams (unfiltered + one per star rating) | Google Play limits pagination depth per stream. Separate streams multiply reachable volume and guarantee 1-star coverage, where complaint themes live. |
| `reply_content` / `replied_at` stored from the start | Phase 4 builds training pairs from Swiggy's templated replies. Verified live: 3 of 3 sampled reviews had replies. Omitting these columns would mean re-scraping later. |
| No `user_name` / `user_image` stored | Not used by any phase; avoids collecting other people's names in a public portfolio repo. |
| Phase 1 dependencies only in `requirements.txt` | PyTorch is ~2 GB and not needed until Phase 3. File grows per phase, pinned in Phase 9. |
| Phase 0's hello-world Streamlit page skipped | Streamlit isn't needed until Phase 7; building it now de-risks nothing. |

## Things found and fixed

- **The venv was broken.** It had been created while the parent folder was named
  `Placements ` (trailing space); a venv hard-codes absolute paths, so its interpreter
  symlink was dangling and zero packages were installed. Rebuilt with Homebrew Python
  3.11 — not the system 3.14, which has no PyTorch wheels.
- **Phase 0 was never completed** despite the folder skeleton existing: no
  `requirements.txt`, no `.env`, no `src/db/connection.py`, empty `PROGRESS.md`.
- **Timestamp timezone bug caught before it landed.** `google-play-scraper` builds its
  `at` field with `datetime.fromtimestamp()`, which returns naive *local* time (IST here),
  not UTC. The loader must attach the local offset before writing to `TIMESTAMPTZ`,
  otherwise every review date is silently off by 5.5 hours.

## Done

- Git repo initialised, remote `git@github.com:ayn-aval/Echo.git`, pushed
- venv rebuilt (Python 3.11.15), Phase 1 dependencies installed
- Database `echo` created; `reviews` and `scrape_checkpoints` tables live
- `requirements.txt`, `.env.example`, `.env` (local only)
- `src/utils/config.py`, `src/db/connection.py`, `src/db/init_db.py`
- `src/ingest/load.py` — upsert loader
- `src/ingest/scrape_play.py` — six-stream scraper with resume
- Committed and pushed as `0ce42b4`

## Verified, not just written

- **De-duplication:** loading the same batch twice gave `(5 new, 0 updated)` then
  `(0 new, 5 updated)`. Table held 5 rows, not 10.
- **Resume:** consecutive runs went 600 -> 800 rows with `fetched_count` 600 -> 800,
  and the date range extended backwards. It continued from the stored token rather
  than re-downloading the newest page.
- **Timezone:** scraper's naive `23:51:14` (IST) stores as `18:21:14 UTC` and reads
  back as `23:51:14 IST`.

## Final dataset (100,000 reviews, 220 days)

| | |
|---|---|
| Rows / distinct review ids | 100,000 / 100,000 — no duplicates |
| Date range | 2026-01-18 to 2026-08-26 |
| Average rating | 3.53 — bimodal: 30.7% 1-star, 53.8% 5-star |
| Has a Swiggy reply | **99.9%** (99,883) |
| Reviews with 4+ words | **36,885 (36.9%)** — the clusterable corpus |
| App versions seen | 285 (10.9% of rows have none) |
| Table size | 46 MB |

Volume is steady at 11k–15k reviews/month, so Phase 8's trend analysis has an even
series to work with. Scraped entirely from the unfiltered stream — the five
per-rating backup streams were never needed, so the corpus is a clean chronological
slice rather than a stitched-together one.

## What the trial data showed (800 reviews, 3 days)

- **100% of reviews have a Swiggy reply.** The Phase 4 weak-supervision signal is
  universal, not merely common — stronger than the brief assumed.
- **62% of reviews are 3 words or fewer** (36% are a single word: "good", "ok",
  "nice"). If this holds, 100k scraped reviews yields roughly 38k with enough text
  to carry a theme. This is the central input to the cleaning-rules decision.
- Ratings are bimodal: 30% 1-star, 55% 5-star, 15% in between. Complaints are well
  represented, which is what the themes need.
- ~270 reviews/day, so 100k is roughly a year of history — enough for Phase 8 trends.
- 0% empty text; 13% missing app version.

## Known rough edges

- The scraper overshoots `--limit` by up to one batch (200), because the budget is
  checked between batches. Harmless; worth knowing when counts look off by <200.
- `data/scrape.log` is nearly unreadable in a text editor: tqdm draws progress with
  carriage returns, so the file is one enormous line. Watch progress in SQL instead —
  `SELECT ..., now() - updated_at AS idle FROM scrape_checkpoints`. The `idle` column
  is the reliable signal; the progress bar keeps redrawing even when nothing arrives.

## Bug found and fixed mid-scrape

The first full run **hung silently for 84 minutes** at 83,800 reviews.
`google-play-scraper` calls `urlopen()` with no timeout, and Python's default is to
wait forever, so one silent connection stalled the process with no error. The
progress bar kept refreshing, which made it look alive; the giveaway was
`scrape_checkpoints.updated_at` going stale.

Fixed in `src/ingest/scrape_play.py`: a 30-second `socket.setdefaulttimeout()`, and
`fetch_batch` now catches request failures and retries with backoff. Crucially, if
the retries are all *errors* rather than genuine empty responses, it raises instead
of marking the stream exhausted — a failing network is not an exhausted stream, and
conflating the two would silently abandon the rest of a stream.

The checkpoint design absorbed this completely: the restart resumed from 83,800 and
lost nothing.

## Exact next step

Decide the cleaning rules (one-word, emoji-only,
Hindi/Hinglish, very long reviews) against the real data, record them here and in
PROJECT_PLAN.md, then write the exploratory analysis with plots into `results/`.

## Not started

Cleaning rules (one-word / emoji-only / Hinglish / very long reviews) and the
exploratory analysis with plots — both are separate steps after the trial run.
