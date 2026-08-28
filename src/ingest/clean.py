"""Apply the cleaning rules to the review corpus.

Nothing is ever deleted. Each review is labelled with a word count, a rough
language tag, and a keep_for_themes flag. The Overview and rating charts use
every row; only theme discovery uses the filtered subset.

Rules (decided against the real corpus on 2026-08-28 — see PROJECT_PLAN.md):
  drop  no text at all
  drop  no alphabetic characters (emoji-only, punctuation-only)
  drop  a single word          -- 1,830 distinct strings across 35,646 rows
  keep  two or more words, in any language

    python -m src.ingest.clean
"""

import re

from psycopg2.extras import execute_values

from src.db.connection import connection

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LETTER = re.compile(r"[A-Za-zऀ-ॿ]")

# A deliberately simple word-list heuristic, not real language detection.
# Off-the-shelf detectors classify romanised Hindi as English most of the time,
# so a transparent rule we can explain beats a black box that is quietly wrong.
HINGLISH = re.compile(
    r"\b(hai|hain|nahi|nahin|nhi|kya|kyu|bhai|accha|acha|bahut|bhut|bohot|kar|karo|"
    r"kiya|krne|mera|meri|apna|aap|thik|theek|sahi|paisa|paise|khana|bekar|bakwas|"
    r"ghatiya|zyada|jyada|matlab|liye|diya|mila|milta|baad|order krne|mat karo)\b",
    re.IGNORECASE,
)


def classify(content):
    """Return (word_count, lang, keep_for_themes) for one review."""
    text = (content or "").strip()
    if not text:
        return 0, None, False

    words = len(text.split())
    if DEVANAGARI.search(text):
        lang = "hi"
    elif HINGLISH.search(text):
        lang = "hinglish"
    else:
        lang = "en"

    keep = bool(LETTER.search(text)) and words >= 2
    return words, lang, keep


def main() -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT app, review_id, content FROM reviews")
            rows = cur.fetchall()
        print(f"classifying {len(rows):,} reviews...")

        updates = [(app, rid, *classify(content)) for app, rid, content in rows]

        with conn.cursor() as cur:
            execute_values(cur, """
                UPDATE reviews AS r
                   SET word_count = v.word_count,
                       lang = v.lang,
                       keep_for_themes = v.keep
                  FROM (VALUES %s) AS v(app, review_id, word_count, lang, keep)
                 WHERE r.app = v.app AND r.review_id = v.review_id
            """, updates, template="(%s,%s,%s::int,%s,%s::boolean)", page_size=5000)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("""SELECT count(*), count(*) FILTER (WHERE keep_for_themes),
                                  count(*) FILTER (WHERE NOT keep_for_themes)
                             FROM reviews""")
            total, keep, drop = cur.fetchone()
            print(f"\n  total       {total:,}")
            print(f"  kept        {keep:,}  ({keep/total:.1%})")
            print(f"  flagged out {drop:,}  ({drop/total:.1%})")

            cur.execute("""SELECT lang, count(*), count(*) FILTER (WHERE keep_for_themes)
                             FROM reviews GROUP BY lang ORDER BY 2 DESC""")
            print("\n  language      rows      kept")
            for lang, c, k in cur.fetchall():
                print(f"  {str(lang):10} {c:7,}   {k:7,}")


if __name__ == "__main__":
    main()
