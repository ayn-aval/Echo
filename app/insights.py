"""Turn topic counts into a ranked list of what to fix first.

A dashboard that only reports counts leaves the reader to work out what matters.
This ranks complaints so the list itself is the recommendation, and every score
is explainable in one sentence — a number nobody can interrogate is worse than no
number, because it gets followed anyway.

Three things decide the ranking:

    size      how many customers raised it in the recent window
    severity  how angry they were, from the average star rating
    momentum  whether it is a bigger share of complaints than the window before

Momentum only ever raises a score, never lowers it: a large, long-standing
complaint is still worth fixing even when it is not growing.
"""

import re

import pandas as pd

from shared import MODEL, sql

WINDOW_WEEKS = 8

# Real topics, but nothing anyone can act on: two hold reviews that are angry
# without saying why, and one groups reviews by the language they are written in
# rather than their subject. They stay visible on the Topics page, where their
# size is the point; they are kept out of a list headed "fix this first",
# because that list is a recommendation.
NOT_ACTIONABLE = ("no detail given", "Reviews in Hindi and Hinglish")

EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2190-\u2BFF\u2600-\u27BF\uFE0F]")


def _severity(rating: float) -> float:
    """1.0 at one star, 0.0 at three. Above three is not a complaint."""
    return max(0.0, min(1.0, (3.0 - float(rating)) / 2.0))


def priorities(limit: int = 5, window_weeks: int = WINDOW_WEEKS,
               areas: tuple = (), days: int | None = None) -> pd.DataFrame:
    """Ranked complaints, with the numbers behind the rank kept alongside.

    areas/days come from the global filter bar, so the ranking answers for the
    slice the reader is looking at rather than always for the whole corpus.
    """
    area_sql = ""
    if areas:
        area_sql = " AND t.category IN (" + ", ".join(f"'{a}'" for a in areas) + ")"
    # Two windows are needed to measure momentum, so the period filter is
    # widened to twice the window rather than applied as the reader set it.
    span = ""
    if days:
        span = (f" AND r.reviewed_at >= (SELECT max(reviewed_at) FROM reviews"
                f" WHERE app='swiggy') - interval '{max(days, 2 * window_weeks * 7)} days'")

    weekly = sql(f"""
        SELECT date_trunc('week', r.reviewed_at)::date AS week,
               rt.theme_id,
               coalesce(t.display_name, t.label) AS name,
               coalesce(t.category, 'Other') AS area,
               t.avg_rating,
               count(*) AS reviews
          FROM review_themes rt
          JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
          JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
         WHERE rt.model = %s AND rt.theme_id >= 0 AND t.avg_rating <= 2.8
           {area_sql}{span}
         GROUP BY 1, 2, 3, 4, 5""", (MODEL,))
    if weekly.empty:
        return weekly

    weeks = sorted(weekly.week.unique())
    recent_weeks = weeks[-window_weeks:]
    prior_weeks = weeks[-2 * window_weeks:-window_weeks]
    if not prior_weeks:
        prior_weeks = weeks[:len(weeks) // 2]

    recent = weekly[weekly.week.isin(recent_weeks)]
    prior = weekly[weekly.week.isin(prior_weeks)]

    def totals(df):
        g = df.groupby(["theme_id", "name", "area", "avg_rating"],
                       as_index=False).reviews.sum()
        g["share"] = g.reviews / max(g.reviews.sum(), 1)
        return g

    now, was = totals(recent), totals(prior)
    m = now.merge(was[["theme_id", "reviews", "share"]], on="theme_id",
                  how="left", suffixes=("", "_prior")).fillna(
                      {"reviews_prior": 0, "share_prior": 0})

    m["severity"] = m.avg_rating.map(_severity)
    # Growth in *share*, so a theme is not credited for a busy month. Only the
    # upside counts; a shrinking complaint is not penalised below its size.
    m["growth"] = (m.share / m.share_prior.replace(0, pd.NA) - 1).fillna(0).clip(lower=0)
    m["score"] = m.share * m.severity * (1 + m.growth)

    m["change_pct"] = ((m.reviews - m.reviews_prior)
                       / m.reviews_prior.replace(0, pd.NA) * 100)
    m = m[~m.name.str.contains("|".join(NOT_ACTIONABLE), case=False, regex=True)]
    m = m[m.reviews >= 25].sort_values("score", ascending=False).head(limit)

    # Several candidates per topic, so a quote free of emoji can be preferred.
    # This selects between real reviews; it never edits one.
    pool = sql("""
        SELECT rt.theme_id, r.content, rt.strength
          FROM review_themes rt
          JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
         WHERE rt.model = %s AND rt.theme_id = ANY(%s) AND r.word_count >= 12
         ORDER BY rt.theme_id, rt.strength DESC""",
        (MODEL, [int(i) for i in m.theme_id]))
    picked = []
    for theme_id in m.theme_id:
        rows = pool[pool.theme_id == theme_id].head(40)
        clean = rows[~rows.content.astype(str).str.contains(EMOJI, regex=True)]
        best = (clean if not clean.empty else rows)
        picked.append({"theme_id": theme_id,
                       "content": best.content.iloc[0] if not best.empty else ""})
    m = m.merge(pd.DataFrame(picked), on="theme_id", how="left")

    m["why"] = [
        f"{int(r.reviews):,} customers in the last {window_weeks} weeks, "
        f"averaging {r.avg_rating:.1f} stars"
        + (f", up {r.change_pct:.0f}% on the period before"
           if pd.notna(r.change_pct) and r.change_pct > 5
           else f", down {abs(r.change_pct):.0f}% on the period before"
           if pd.notna(r.change_pct) and r.change_pct < -5
           else ", steady on the period before")
        for r in m.itertuples()]

    m["urgency"] = ["crit" if r.severity >= 0.75 and r.growth > 0.05
                    else "warn" if r.severity >= 0.6 or r.growth > 0.15
                    else "flat" for r in m.itertuples()]
    return m.reset_index(drop=True)


def headline_health() -> dict:
    """The three numbers an operator checks first."""
    row = sql("""
        SELECT count(*) AS reviews,
               round(avg(score)::numeric, 2) AS rating,
               count(*) FILTER (WHERE score <= 2) AS unhappy,
               max(reviewed_at)::date AS latest
          FROM reviews WHERE app = 'swiggy'""").iloc[0]

    recent = sql("""
        SELECT round(avg(score)::numeric, 2) AS rating
          FROM reviews WHERE app = 'swiggy'
           AND reviewed_at >= (SELECT max(reviewed_at) FROM reviews
                                WHERE app='swiggy') - interval '28 days'""").iloc[0]
    earlier = sql("""
        SELECT round(avg(score)::numeric, 2) AS rating
          FROM reviews WHERE app = 'swiggy'
           AND reviewed_at <  (SELECT max(reviewed_at) FROM reviews
                                WHERE app='swiggy') - interval '28 days'
           AND reviewed_at >= (SELECT max(reviewed_at) FROM reviews
                                WHERE app='swiggy') - interval '56 days'""").iloc[0]

    return {"reviews": int(row.reviews), "rating": float(row.rating),
            "unhappy": int(row.unhappy), "latest": row.latest,
            "rating_28d": float(recent.rating),
            "rating_delta": float(recent.rating) - float(earlier.rating)}
