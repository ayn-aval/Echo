"""Flag topics whose volume this week is out of line with their own history.

    python -m src.analytics.alerts
    python -m src.analytics.alerts --explain --threshold 2.0

THE METHOD, IN ONE PARAGRAPH
For each topic, take the previous BASELINE complete weeks. Compute their mean and
standard deviation — the typical week, and the typical wobble around it. The
z-score is how many wobbles this week sits above the mean:
z = (this week - mean) / sd. A z of 3 is roughly a 1-in-740 event if the counts
were normal and independent. Deliberately simple: a rule that can be checked by
hand is worth more than one nobody can defend in a meeting.

FOUR GUARDS, EACH FOR A SPECIFIC FAILURE

  MIN_REVIEWS   A topic going 1 -> 4 reviews has z above 3 and means nothing.
                Weekly counts behave like Poisson draws, so at small numbers the
                noise is the same size as the signal.

  zero variance A perfectly flat topic has sd = 0, making z infinite or NaN.
                The fallback is sd = sqrt(mean), the Poisson standard deviation,
                which is the correct scale for count data rather than an
                arbitrary epsilon.

  EFFECT        A jump can be statistically unusual and operationally trivial.
                Requiring at least EFFECT x the baseline mean keeps the list to
                things worth a manager's morning.

  share_z       A raw count rises whenever total volume rises. The same test is
                run on the topic's share of the week, and reported beside the
                count. If the count spiked and the share did not, the week was
                simply busier — the alert says so rather than leaving the reader
                to notice.

WHY THE THRESHOLD IS 3.0 AND NOT 2.0
110 topics are tested every week. At z >= 2 the per-test false-alarm rate is
about 2.3%, so roughly 2-3 topics would be flagged every week by chance alone —
enough to make the list worthless. At z >= 3 the expected rate is about 0.15 per
week. The script prints this number next to the alert count so the reader can
judge how much of the list is likely to be noise.

FOUR FAILURE MODES NO GUARD FIXES

  Slow burn is invisible. A topic growing 10% a week drags its own trailing mean
  up with it and never trips the threshold. This rule catches jumps, not trends.

  Counts are not normally distributed. They are right-skewed, so a positive z
  overstates its own rarity to some degree; treat "1 in 740" as an order of
  magnitude, not a probability.

  Weeks are not independent. An incident spanning two weeks lifts the baseline
  for the weeks after it and suppresses later alerts about the same problem.

  No seasonality. A festival or an outage week lifts everything at once, and this
  rule attributes it to the topics rather than to the calendar.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.db.connection import connection

MODEL = "sbert-domain"
BASELINE = 8          # complete weeks of history each week is judged against
MIN_REVIEWS = 15      # below this, weekly counts are mostly Poisson noise
EFFECT = 1.5          # must also be this multiple of the baseline mean
NEW_MIN = 20          # a "new" topic needs at least this many to count
THRESHOLD = 3.0


def load() -> pd.DataFrame:
    with connection() as conn:
        return pd.read_sql("""
            SELECT w.theme_id, w.week_start, w.reviews, w.share,
                   coalesce(t.display_name, t.label) AS name,
                   coalesce(t.category, 'Other') AS area
              FROM theme_weekly w
              JOIN themes t ON t.model = w.model AND t.theme_id = w.theme_id
             WHERE w.model = %s
             ORDER BY w.theme_id, w.week_start""", conn, params=(MODEL,))


def _z(current: float, history: np.ndarray) -> tuple:
    """z-score of `current` against `history`, with the Poisson sd fallback."""
    mean = float(history.mean())
    sd = float(history.std(ddof=1)) if len(history) > 1 else 0.0
    if sd <= 1e-9:
        # A flat baseline has no observed spread. sqrt(mean) is the standard
        # deviation a Poisson count process would have at that mean, so it keeps
        # the score on a meaningful scale instead of dividing by a fudge factor.
        sd = float(np.sqrt(max(mean, 1e-9)))
    return (current - mean) / sd, mean, sd


def detect(df: pd.DataFrame, threshold=THRESHOLD, baseline=BASELINE,
           min_reviews=MIN_REVIEWS, effect=EFFECT) -> pd.DataFrame:
    rows = []
    for theme_id, g in df.groupby("theme_id"):
        g = g.sort_values("week_start").reset_index(drop=True)
        for i in range(baseline, len(g)):
            window = g.iloc[i - baseline:i]
            now = g.iloc[i]
            counts = window.reviews.to_numpy(dtype=float)

            if counts.sum() == 0:
                if now.reviews >= NEW_MIN:
                    rows.append(dict(theme_id=theme_id, name=now["name"],
                                     area=now.area, week_start=now.week_start,
                                     kind="new", reviews=int(now.reviews),
                                     mean=0.0, sd=0.0, z=np.nan, share_z=np.nan,
                                     baseline_weeks=baseline))
                continue

            if now.reviews < min_reviews:
                continue
            z, mean, sd = _z(float(now.reviews), counts)
            if z < threshold or now.reviews < effect * mean:
                continue
            share_z, _, _ = _z(float(now.share),
                               window.share.to_numpy(dtype=float))
            rows.append(dict(theme_id=theme_id, name=now["name"], area=now.area,
                             week_start=now.week_start, kind="spike",
                             reviews=int(now.reviews), mean=round(mean, 2),
                             sd=round(sd, 2), z=round(z, 2),
                             share_z=round(share_z, 2), baseline_weeks=baseline))
    return pd.DataFrame(rows)


def persist(alerts: pd.DataFrame, threshold: float) -> None:
    from psycopg2.extras import execute_values
    with connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM theme_alerts WHERE model = %s", (MODEL,))
        if alerts.empty:
            return
        execute_values(cur, """
            INSERT INTO theme_alerts (model, theme_id, week_start, kind, reviews,
                baseline_mean, baseline_sd, z, share_z, threshold, baseline_weeks)
            VALUES %s""",
            [(MODEL, int(r.theme_id), r.week_start, r.kind, int(r.reviews),
              None if pd.isna(r.mean) else float(r.mean),
              None if pd.isna(r.sd) else float(r.sd),
              None if pd.isna(r.z) else float(r.z),
              None if pd.isna(r.share_z) else float(r.share_z),
              threshold, int(r.baseline_weeks)) for r in alerts.itertuples()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--baseline", type=int, default=BASELINE)
    ap.add_argument("--explain", action="store_true",
                    help="print each alert with the baseline it was judged against")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = load()
    if df.empty:
        raise SystemExit("theme_weekly is empty — run `python -m src.analytics.weekly`")

    alerts = detect(df, args.threshold, args.baseline)
    topics = df.theme_id.nunique()
    weeks_tested = df.week_start.nunique() - args.baseline
    per_test = norm.sf(args.threshold)
    expected = topics * per_test

    print(f"{topics} topics x {weeks_tested} weeks tested at z >= {args.threshold}")
    print(f"expected false alarms from chance alone: {expected:.2f} per week "
          f"({expected * weeks_tested:.1f} over the whole period)")
    print(f"alerts found: {len(alerts)}"
          + (f"  ({(alerts.kind == 'new').sum()} new topics)" if not alerts.empty else ""))

    if not alerts.empty:
        recent = alerts.sort_values("week_start", ascending=False)
        show = recent if args.explain else recent.head(10)
        print()
        for r in show.itertuples():
            if r.kind == "new":
                print(f"  {r.week_start}  NEW    {r.name} ({r.area}) — "
                      f"{r.reviews} reviews, none in the previous "
                      f"{r.baseline_weeks} weeks")
                continue
            volume = ("share rose too" if r.share_z >= 2
                      else "SHARE FLAT — the week was simply busier")
            print(f"  {r.week_start}  z={r.z:>5.2f}  {r.name} ({r.area})")
            print(f"              {r.reviews} reviews against a baseline of "
                  f"{r.mean:.1f} ± {r.sd:.1f} over {r.baseline_weeks} weeks; "
                  f"{volume}")

    if not args.dry_run:
        persist(alerts, args.threshold)
        print(f"\nwrote {len(alerts)} rows to theme_alerts")


if __name__ == "__main__":
    main()
