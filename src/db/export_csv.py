"""Export reviews from Postgres to a CSV file — for Excel, Power BI, or sharing.

    python -m src.db.export_csv                          # everything
    python -m src.db.export_csv --min-words 4            # only clusterable reviews
    python -m src.db.export_csv --limit 500 --out data/sample.csv

The database stays the source of truth; this just makes a snapshot you can open
in a spreadsheet.
"""

import argparse
from pathlib import Path

from src.db.connection import connection

COLUMNS = ("app, review_id, score, content, thumbs_up, review_version, "
           "reviewed_at, reply_content, replied_at")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app", default="swiggy")
    ap.add_argument("--out", default="data/reviews.csv")
    ap.add_argument("--limit", type=int, help="export at most this many rows")
    ap.add_argument("--min-words", type=int, metavar="N",
                    help="only reviews with at least N words")
    args = ap.parse_args()

    where = ["app = %s"]
    params = [args.app]
    if args.min_words:
        where.append("array_length(regexp_split_to_array(trim(content), '\\s+'), 1) >= %s")
        params.append(args.min_words)

    sql = (f"SELECT {COLUMNS} FROM reviews WHERE {' AND '.join(where)} "
           f"ORDER BY reviewed_at DESC")
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with connection() as conn, conn.cursor() as cur:
        query = cur.mogrify(sql, params).decode()
        with out.open("w", encoding="utf-8") as fh:
            cur.copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", fh)
        # Count in SQL, not by counting lines in the file: review text contains
        # newlines, so a multi-line review spans several lines of the CSV.
        cur.execute(f"SELECT count(*) FROM ({query}) t")
        rows = cur.fetchone()[0]

    print(f"{rows:,} rows -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
