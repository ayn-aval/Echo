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
