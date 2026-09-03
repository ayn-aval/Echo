"""Roll the 110 topics up into the areas a business team actually owns.

    python -m src.clustering.theme_categories
    python -m src.clustering.theme_categories --show

110 topics is the right granularity for reading individual reviews and the wrong
granularity for deciding where to invest. A growth or operations team is
organised around delivery, payments, support and so on, so the topics are mapped
onto those areas and the dashboard leads with them.

A category is chosen by counting how many of a topic's distinctive terms belong
to each area and taking the strongest match, rather than by first-match keyword
order — "delivery" appears in nearly every topic, so an ordered rule list would
sweep most of them into one bucket. Topics with a curated name are assigned
directly, since a person has already read them.
"""

import argparse
from collections import Counter

from src.db.connection import connection

MODEL = "sbert-domain"

AREAS = {
    "Delivery": """late delay delayed delays slow waited waiting hours hour hrs
        mins minutes timing timely deliver delivered delivery quick fast fastest
        eta reached arrive""",
    "Orders & refunds": """refund refunds refunded cancel cancelled canceled
        cancellation deducted debited missing wrong item items order placed""",
    "Pricing & fees": """price prices pricing expensive costly cost cheap
        affordable reasonable budget gst fee fees charge charges charged platform
        rupees rs discount discounts coupon coupons offer offers cashback""",
    "Food & product quality": """food quality quantity taste tasty delicious
        stale spoiled cold fresh chicken egg eggs fried fries pizza cheese burger
        groceries grocery restaurant packaging""",
    "Customer support": """support care customer helpline complaint resolution
        response responding chat chatbot bot bots ai human agent executive""",
    "App experience": """app application install download update crash crashes
        working works login page screen easy smooth convenient simple use useful
        interface location map""",
    "Trust & fraud": """fraud fraudulent scam scamming fake cheating cheat
        looting loot steal stealing dishonest misleading unfair""",
    "Competitors": "zomato dominos blinkit zepto swiggy_vs competitor compared",
}
VOCAB = {area: set(words.split()) for area, words in AREAS.items()}

# Topics a person has already read and named keep that judgement.
OVERRIDES = {
    105: "Delivery", 103: "Delivery", 33: "Delivery", 86: "Delivery",
    109: "Orders & refunds",
    68: "Pricing & fees", 71: "Pricing & fees",
    88: "Food & product quality", 45: "Food & product quality",
    64: "Customer support",
    62: "Competitors",
}
PRAISE = "General praise"
OTHER = "Other"


def categorise(terms: str, rating: float, theme_id: int) -> str:
    if theme_id in OVERRIDES:
        return OVERRIDES[theme_id]
    words = {w.strip() for w in str(terms).lower().replace(",", " ").split()}
    hits = Counter({area: len(words & vocab) for area, vocab in VOCAB.items()})
    area, score = hits.most_common(1)[0]
    if score >= 2:
        return area
    # One weak keyword is not evidence. A high-rated topic with no clear subject
    # is praise; a low-rated one with no clear subject is a real complaint we
    # cannot place, and saying so beats inventing a home for it.
    if score == 1 and rating <= 3.5:
        return area
    return PRAISE if rating >= 4.0 else OTHER


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true", help="print, write nothing")
    args = ap.parse_args()

    with connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT theme_id, top_terms, avg_rating, n_rows,
                              coalesce(display_name, label)
                         FROM themes WHERE model = %s ORDER BY n_rows DESC""",
                    (MODEL,))
        rows = cur.fetchall()
        assigned = []
        for theme_id, terms, rating, n_rows, name in rows:
            area = categorise(terms, float(rating), theme_id)
            assigned.append((area, n_rows, name))
            if not args.show:
                cur.execute("UPDATE themes SET category = %s "
                            "WHERE model = %s AND theme_id = %s",
                            (area, MODEL, theme_id))

    totals = Counter()
    for area, n_rows, _ in assigned:
        totals[area] += n_rows
    print(f"{len(rows)} topics mapped to {len(totals)} areas\n")
    for area, n in totals.most_common():
        topics = sum(1 for a, _, _ in assigned if a == area)
        print(f"  {area:26} {n:>6,} reviews   {topics:>3} topics")
    if args.show:
        print("\n(--show: nothing written)")


if __name__ == "__main__":
    main()
