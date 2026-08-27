# Progress

## Current status

**Phase 1 — Data collection.** In flight. Environment repaired, schema and
connection code written, blocked on two manual setup steps (below).

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
- `requirements.txt`, `.env.example`
- `src/db/schema.sql` — `reviews` and `scrape_checkpoints` tables
- `src/utils/config.py`, `src/db/connection.py`, `src/db/init_db.py`

## Blocked on (user action)

1. `/Library/PostgreSQL/18/bin/createdb -U postgres echo`
2. `cp .env.example .env`, then fill in the PostgreSQL password

## Exact next step

Once the two steps above are done: run `python -m src.db.init_db` to create the tables,
then write `src/ingest/load.py` (upsert with de-duplication) and
`src/ingest/scrape_play.py` (multi-stream scraper with resume), then the 500-review
trial run.

## Not started

Cleaning rules (one-word / emoji-only / Hinglish / very long reviews) and the
exploratory analysis with plots — both are separate steps after the trial run.
