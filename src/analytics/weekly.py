"""Build the weekly time series each topic's alerting is judged against.

    python -m src.analytics.weekly

Writes theme_weekly: one row per topic per COMPLETE week. Safe to re-run.

Partial weeks are dropped here rather than downstream. The corpus currently ends
on a Wednesday, so the newest bucket holds three days and about a third of a
normal week's reviews. Any rule comparing that bucket to a seven-day baseline
concludes that every topic collapsed at once. Excluding it at the source means
every consumer inherits the guarantee instead of each one having to remember.

Both a count and a share are stored. The count answers "how many people raised
this"; the share answers "how much of the conversation was this". They diverge
whenever total review volume moves, and a reader needs both to tell a real change
from a busy week.
"""

from src.db.connection import connection

MODEL = "sbert-domain"
DAYS_REQUIRED = 7


def main() -> None:
    with connection() as conn, conn.cursor() as cur:
        # Which week buckets are complete. A week qualifies only if all seven
        # dates appear somewhere in the review table — not just in this topic,
        # which would drop quiet weeks for small topics.
        cur.execute("""
            SELECT date_trunc('week', reviewed_at)::date AS week_start,
                   count(DISTINCT reviewed_at::date) AS days
              FROM reviews WHERE app = 'swiggy'
             GROUP BY 1 ORDER BY 1""")
        weeks = cur.fetchall()
        complete = [w for w, days in weeks if days >= DAYS_REQUIRED]
        partial = [(w, d) for w, d in weeks if d < DAYS_REQUIRED]

        cur.execute("DELETE FROM theme_weekly WHERE model = %s", (MODEL,))
        cur.execute("""
            WITH weekly AS (
              SELECT date_trunc('week', r.reviewed_at)::date AS week_start,
                     rt.theme_id,
                     count(*) AS reviews,
                     count(*) FILTER (WHERE r.score <= 2) AS unhappy,
                     round(avg(r.score)::numeric, 2) AS avg_rating
                FROM review_themes rt
                JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
               WHERE rt.model = %s AND rt.theme_id >= 0
               GROUP BY 1, 2
            ), totals AS (
              SELECT week_start, sum(reviews) AS total FROM weekly GROUP BY 1
            )
            INSERT INTO theme_weekly
                   (model, theme_id, week_start, reviews, unhappy, avg_rating, share)
            SELECT %s, w.theme_id, w.week_start, w.reviews, w.unhappy,
                   w.avg_rating, w.reviews::numeric / t.total
              FROM weekly w JOIN totals t USING (week_start)
             WHERE w.week_start = ANY(%s)""",
            (MODEL, MODEL, complete))

        cur.execute("""SELECT count(*), count(DISTINCT theme_id),
                              count(DISTINCT week_start),
                              min(week_start), max(week_start)
                         FROM theme_weekly WHERE model = %s""", (MODEL,))
        rows, topics, n_weeks, first, last = cur.fetchone()

    print(f"theme_weekly: {rows:,} rows · {topics} topics x {n_weeks} complete "
          f"weeks · {first} to {last}")
    for week, days in partial:
        print(f"  dropped {week} — only {days} of {DAYS_REQUIRED} days present")
    if not partial:
        print("  no partial weeks to drop")


if __name__ == "__main__":
    main()
