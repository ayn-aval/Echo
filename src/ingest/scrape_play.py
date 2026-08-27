"""Scrape Google Play reviews into Postgres — incrementally, and resumable.

    python -m src.ingest.scrape_play --app swiggy --limit 500     # trial
    python -m src.ingest.scrape_play --app swiggy                 # full run

Interrupt it with Ctrl+C whenever you like. Every batch is committed as it
arrives, so rerunning the same command picks up where it stopped.
"""

import argparse
import pickle
import time

import psycopg2
from google_play_scraper import Sort, reviews
from tqdm import tqdm

from src.db.connection import connection
from src.ingest.load import upsert_reviews

APP_IDS = {"swiggy": "in.swiggy.android", "zomato": "com.application.zomato"}

STREAMS = (0, 1, 2, 3, 4, 5)  # 0 = unfiltered, 1..5 = only that star rating
BATCH = 200                   # reviews per HTTP request
PAUSE = 1.5                   # seconds between requests, to stay polite
EMPTY_RETRIES = 3             # see fetch_batch()


def read_checkpoint(conn, app, stream):
    """Where did this stream get to? Creates the row on first use."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO scrape_checkpoints (app, score_filter) VALUES (%s, %s) "
            "ON CONFLICT (app, score_filter) DO NOTHING", (app, stream))
        cur.execute(
            "SELECT continuation_token, fetched_count, inserted_count, exhausted "
            "FROM scrape_checkpoints WHERE app=%s AND score_filter=%s", (app, stream))
        blob, fetched, inserted, exhausted = cur.fetchone()
    return (pickle.loads(blob) if blob else None), fetched, inserted, exhausted


def write_checkpoint(conn, app, stream, token, fetched, inserted, exhausted, error=None):
    """Save progress and commit, so a crash costs at most one batch."""
    blob = psycopg2.Binary(pickle.dumps(token)) if token is not None else None
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE scrape_checkpoints SET continuation_token=%s, fetched_count=%s, "
            "inserted_count=%s, exhausted=%s, last_error=%s, updated_at=now() "
            "WHERE app=%s AND score_filter=%s",
            (blob, fetched, inserted, exhausted, error, app, stream))
    conn.commit()


def fetch_batch(app_id, stream, token):
    """One request to Google. Returns (reviews, new_token).

    google-play-scraper catches network errors internally and returns an empty
    list with a null token — indistinguishable from a stream that has genuinely
    run out. So we retry an empty response a few times with growing delays
    before accepting that the stream is finished. Without this, a single blip
    would permanently abandon the rest of a stream.
    """
    if token is not None:
        # The token carries lang, country, sort and score filter with it.
        call = lambda: reviews(app_id, continuation_token=token)
    else:
        kwargs = {"lang": "en", "country": "in", "sort": Sort.NEWEST, "count": BATCH}
        if stream:
            kwargs["filter_score_with"] = stream
        call = lambda: reviews(app_id, **kwargs)

    delay, new_token = PAUSE, token
    for attempt in range(EMPTY_RETRIES + 1):
        batch, new_token = call()
        if batch:
            return batch, new_token
        if attempt < EMPTY_RETRIES:
            tqdm.write(f"      empty response — retry {attempt + 1}/{EMPTY_RETRIES} "
                       f"in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    return [], new_token


def run_stream(conn, app, app_id, stream, budget):
    """Page through one stream until it dries up or the budget is spent."""
    token, fetched, inserted, exhausted = read_checkpoint(conn, app, stream)
    label = "all ratings" if stream == 0 else f"{stream}-star    "
    if exhausted:
        print(f"  {label}: already exhausted ({fetched:,} fetched previously) — skipping")
        return 0

    bar = tqdm(total=budget, desc=f"  {label}", unit="rev")
    added = 0
    try:
        while added < budget:
            batch, token = fetch_batch(app_id, stream, token)
            if not batch:
                write_checkpoint(conn, app, stream, token, fetched, inserted, True)
                tqdm.write(f"  {label}: exhausted after {fetched:,} fetched")
                break
            new, _updated = upsert_reviews(conn, app, batch)
            fetched += len(batch)
            inserted += new
            added += new
            write_checkpoint(conn, app, stream, token, fetched, inserted, False)
            bar.update(min(new, budget - bar.n))
            time.sleep(PAUSE)
    finally:
        bar.close()
    return added


def count_reviews(conn, app):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM reviews WHERE app=%s", (app,))
        return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app", default="swiggy", choices=sorted(APP_IDS))
    ap.add_argument("--target", type=int, default=100_000,
                    help="total reviews to hold for this app (default 100000)")
    ap.add_argument("--limit", type=int,
                    help="add at most this many in this run — for trials")
    args = ap.parse_args()

    with connection() as conn:
        start = count_reviews(conn, args.app)
        target = min(args.target, start + args.limit) if args.limit else args.target
        print(f"{args.app}: {start:,} stored, aiming for {target:,}\n")
        if start >= target:
            print("Target already met — nothing to do.")
            return
        try:
            for stream in STREAMS:
                if count_reviews(conn, args.app) >= target:
                    break
                run_stream(conn, args.app, APP_IDS[args.app], stream,
                           target - count_reviews(conn, args.app))
        except KeyboardInterrupt:
            print("\nStopped. Progress is saved — rerun the same command to resume.")
        print(f"\n{args.app}: {count_reviews(conn, args.app):,} reviews stored")


if __name__ == "__main__":
    main()
