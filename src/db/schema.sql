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
