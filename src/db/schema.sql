-- Echo — Phase 1 schema.
-- Safe to re-run: every statement is IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS reviews (
    app             TEXT        NOT NULL,           -- 'swiggy', later 'zomato'
    review_id       TEXT        NOT NULL,           -- Play Store's own id (a UUID)
    content         TEXT,                           -- review text; NULL = rating with no words
    score           SMALLINT    NOT NULL CHECK (score BETWEEN 1 AND 5),
    thumbs_up       INTEGER     NOT NULL DEFAULT 0,
    review_version  TEXT,                           -- app version reviewed; sometimes NULL
    reviewed_at     TIMESTAMPTZ NOT NULL,
    reply_content   TEXT,                           -- Swiggy's reply (Phase 4 training signal)
    replied_at      TIMESTAMPTZ,
    lang            TEXT,                           -- filled by the later cleaning step
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (app, review_id)
);

CREATE INDEX IF NOT EXISTS ix_reviews_app_date  ON reviews (app, reviewed_at DESC);
CREATE INDEX IF NOT EXISTS ix_reviews_app_score ON reviews (app, score);

-- One row per (app, score_filter) scrape stream. Lets an interrupted scrape resume
-- instead of starting over. Inspect progress with:  SELECT * FROM scrape_checkpoints;
CREATE TABLE IF NOT EXISTS scrape_checkpoints (
    app                TEXT        NOT NULL,
    score_filter       SMALLINT    NOT NULL,        -- 0 = unfiltered, 1..5 = that rating only
    continuation_token BYTEA,                       -- pickled resume token; NULL = not started
    fetched_count      INTEGER     NOT NULL DEFAULT 0,
    inserted_count     INTEGER     NOT NULL DEFAULT 0,
    exhausted          BOOLEAN     NOT NULL DEFAULT FALSE,
    last_error         TEXT,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (app, score_filter)
);

-- Cleaning columns, added after the corpus was scraped and inspected.
-- Raw reviews are never deleted; they are flagged instead, so the Overview
-- page can still report on all 100,000 while themes use the filtered subset.
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS word_count      INTEGER;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS keep_for_themes BOOLEAN;

CREATE INDEX IF NOT EXISTS ix_reviews_keep
    ON reviews (app) WHERE keep_for_themes;

-- Phase 2: the review-retrieval evaluation set.
-- eval_pool is persisted so the labelling task is reproducible — the same
-- candidates every time, rather than regenerated differently on each run.
CREATE TABLE IF NOT EXISTS eval_queries (
    query_id    SERIAL PRIMARY KEY,
    query_text  TEXT        NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_pool (
    query_id   INTEGER NOT NULL REFERENCES eval_queries(query_id) ON DELETE CASCADE,
    app        TEXT    NOT NULL,
    review_id  TEXT    NOT NULL,
    sources    TEXT    NOT NULL,   -- which retrieval systems surfaced this candidate
    PRIMARY KEY (query_id, app, review_id),
    FOREIGN KEY (app, review_id) REFERENCES reviews(app, review_id)
);

CREATE TABLE IF NOT EXISTS eval_judgements (
    query_id   INTEGER     NOT NULL REFERENCES eval_queries(query_id) ON DELETE CASCADE,
    app        TEXT        NOT NULL,
    review_id  TEXT        NOT NULL,
    relevant   BOOLEAN     NOT NULL,
    judged_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (query_id, app, review_id)
);

-- Phase 5: discovered themes.
--
-- Both tables carry a `model` column so the three clustering runs (glove-avg,
-- bert-mean, sbert-domain) coexist and can be compared directly in SQL. Phase 7
-- filters to the winning model; eval/clustering_comparison.py reads all three.
--
-- theme_id -1 means HDBSCAN judged the review to belong to no theme. Those rows
-- ARE stored, so "what share of reviews got a theme" is a query rather than a
-- number someone has to remember. There is deliberately no foreign key from
-- review_themes.theme_id to themes.theme_id, because -1 has no themes row.
CREATE TABLE IF NOT EXISTS themes (
    model             TEXT        NOT NULL,   -- which embedding produced it
    theme_id          INTEGER     NOT NULL,   -- HDBSCAN's cluster id
    label             TEXT        NOT NULL,   -- readable name from c-TF-IDF terms
    top_terms         TEXT        NOT NULL,   -- comma-separated, most distinctive first
    n_rows            INTEGER     NOT NULL,   -- reviews, counting duplicates
    n_texts           INTEGER     NOT NULL,   -- distinct texts
    avg_rating        NUMERIC(3,2),
    example_review_id TEXT,                   -- nearest the cluster centroid
    params            TEXT,                   -- the HDBSCAN settings used
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model, theme_id)
);

CREATE TABLE IF NOT EXISTS review_themes (
    app       TEXT    NOT NULL,
    review_id TEXT    NOT NULL,
    model     TEXT    NOT NULL,
    theme_id  INTEGER NOT NULL,               -- -1 = noise, no themes row
    strength  REAL,                           -- HDBSCAN membership probability
    PRIMARY KEY (app, review_id, model),
    FOREIGN KEY (app, review_id) REFERENCES reviews(app, review_id)
);

CREATE INDEX IF NOT EXISTS ix_review_themes_theme
    ON review_themes (model, theme_id);

-- Phase 5: the blind hand-audit of theme assignments.
-- Which model produced an assignment is never shown while judging, so the
-- verdicts are comparable across models rather than anchored by expectation.
CREATE TABLE IF NOT EXISTS theme_audit (
    audit_id  SERIAL      PRIMARY KEY,
    app       TEXT        NOT NULL,
    review_id TEXT        NOT NULL,
    model     TEXT        NOT NULL,
    theme_id  INTEGER     NOT NULL,
    belongs   BOOLEAN     NOT NULL,
    judged_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (app, review_id, model)
);

-- Phase 7: a human-readable name for a theme, curated once and verified against
-- the theme's own terms before use (see src/clustering/theme_names.py). The
-- machine label stays in themes.label as the evidence; display_name is what a
-- reader sees. NULL means "no curated name yet" and the machine label is shown.
ALTER TABLE themes ADD COLUMN IF NOT EXISTS display_name TEXT;

-- Phase 7: business area a topic rolls up to (src/clustering/theme_categories.py).
ALTER TABLE themes ADD COLUMN IF NOT EXISTS category TEXT;

-- Phase 7: a named filter combination a team member can return to.
CREATE TABLE IF NOT EXISTS saved_views (
    name       TEXT        PRIMARY KEY,
    payload    JSONB       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Phase 8: weekly time series and alerts ──────────────────────────────────
--
-- theme_weekly holds one row per topic per COMPLETE week. Partial weeks are
-- excluded when the table is built (src/analytics/weekly.py), so nothing
-- downstream has to remember that the newest bucket may be three days long.
--
-- `share` is stored alongside `reviews` because a raw count rises whenever total
-- review volume rises. Without the share a reader cannot tell "more people are
-- complaining about this" from "more people reviewed this week".
CREATE TABLE IF NOT EXISTS theme_weekly (
    model      TEXT        NOT NULL,
    theme_id   INTEGER     NOT NULL,
    week_start DATE        NOT NULL,
    reviews    INTEGER     NOT NULL,
    unhappy    INTEGER     NOT NULL,      -- rated 1 or 2 stars
    avg_rating NUMERIC(3,2),
    share      NUMERIC(7,5) NOT NULL,     -- of that week's themed reviews
    PRIMARY KEY (model, theme_id, week_start)
);

CREATE INDEX IF NOT EXISTS ix_theme_weekly_week
    ON theme_weekly (model, week_start DESC);

-- theme_alerts stores the baseline the verdict was reached against, not just the
-- verdict. An alert nobody can audit gets blindly followed or blindly ignored.
CREATE TABLE IF NOT EXISTS theme_alerts (
    model         TEXT         NOT NULL,
    theme_id      INTEGER      NOT NULL,
    week_start    DATE         NOT NULL,
    kind          TEXT         NOT NULL,   -- 'spike' | 'new'
    reviews       INTEGER      NOT NULL,
    baseline_mean NUMERIC(9,3),
    baseline_sd   NUMERIC(9,3),
    z             NUMERIC(6,2),
    share_z       NUMERIC(6,2),            -- same test on share; low means the
                                           -- week was simply busier overall
    threshold     NUMERIC(4,2) NOT NULL,
    baseline_weeks INTEGER     NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (model, theme_id, week_start, kind)
);

CREATE INDEX IF NOT EXISTS ix_theme_alerts_week
    ON theme_alerts (model, week_start DESC);
