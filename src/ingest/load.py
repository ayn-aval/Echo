"""Writing scraped reviews into Postgres without ever creating duplicates."""

from datetime import timezone

from psycopg2.extras import execute_values

COLUMNS = (
    "app", "review_id", "content", "score", "thumbs_up",
    "review_version", "reviewed_at", "reply_content", "replied_at",
)

# ON CONFLICT is Postgres's "upsert": if this (app, review_id) is already stored,
# update it instead of erroring or inserting a second copy. We refresh the fields
# that legitimately change over time and leave the original text and ingested_at
# alone. COALESCE keeps an existing reply if this pass happens to return none.
# RETURNING (xmax = 0) is true for a genuinely new row, false for an update —
# that is how we count new reviews without scanning the whole table each batch.
UPSERT = f"""
INSERT INTO reviews ({", ".join(COLUMNS)}) VALUES %s
ON CONFLICT (app, review_id) DO UPDATE SET
    thumbs_up     = EXCLUDED.thumbs_up,
    reply_content = COALESCE(EXCLUDED.reply_content, reviews.reply_content),
    replied_at    = COALESCE(EXCLUDED.replied_at,    reviews.replied_at)
RETURNING (xmax = 0)
"""


def to_utc(value):
    """Convert a scraper timestamp to UTC.

    google-play-scraper builds its dates with datetime.fromtimestamp(), which
    returns a *naive* datetime in the machine's local zone (IST here), not UTC.
    Attaching the local zone before converting keeps the real instant. Skip this
    and Postgres reads the naive value in its own timezone — every review date
    silently off by hours.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.astimezone()  # naive is local time; make it explicit
    return value.astimezone(timezone.utc)


def to_row(app, review):
    """One scraper dict -> one database row, in COLUMNS order."""
    return (
        app,
        review["reviewId"],
        review.get("content"),
        review["score"],
        review.get("thumbsUpCount") or 0,
        # reviewCreatedVersion is sometimes null while appVersion is present
        review.get("reviewCreatedVersion") or review.get("appVersion"),
        to_utc(review["at"]),
        review.get("replyContent"),
        to_utc(review.get("repliedAt")),
    )


def upsert_reviews(conn, app, reviews):
    """Store a batch. Returns (new_rows, updated_rows)."""
    rows = [to_row(app, r) for r in reviews]
    if not rows:
        return 0, 0
    with conn.cursor() as cur:
        flags = execute_values(cur, UPSERT, rows, page_size=len(rows), fetch=True)
    new = sum(1 for (is_new,) in flags if is_new)
    return new, len(flags) - new
