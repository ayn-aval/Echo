"""Give the significant themes names a person can read.

    python -m src.clustering.theme_names            # apply
    python -m src.clustering.theme_names --check    # verify only, change nothing

Clustering produces labels like "mere / karne / karte" and "mins / 45 / 15".
Those are honest evidence of what the algorithm found, and they are useless as
something to show a reader. This module maps the significant themes to names
written from their actual contents, keeping the machine label alongside as the
evidence.

The names are curated, so they can go stale: theme ids are assigned by HDBSCAN
and would be reassigned if the clustering were re-run with different parameters.
Each entry therefore records a term that must still be present in that theme's
top terms. If it is not, the name is refused rather than applied to whichever
theme inherited the id — a wrong name is worse than a raw one, because nobody
would think to check it.
"""

import argparse

from src.db.connection import connection

MODEL = "sbert-domain"

# theme_id -> (a term that must still appear in the theme's top_terms, name)
CURATED = {
    # complaints
    105: ("mins", "Long delivery times"),
    103: ("waited", "Waited an hour or more"),
    64: ("worst", "Poor customer support"),
    92: ("bad", "Bad experience, no detail given"),
    109: ("cancelled", "Cancelled orders and refunds"),
    88: ("chicken", "Food and grocery quality"),
    68: ("rupees", "Billing amount disputes"),
    71: ("gst", "High prices, taxes and fees"),
    86: ("working", "Delivery unavailable or not working"),
    85: ("bad", "Negative, no detail given"),
    62: ("zomato", "Compared with Zomato"),
    39: ("karne", "Reviews in Hindi and Hinglish"),
    # praise
    57: ("good", "Praise: 'good'"),
    49: ("nice", "Praise for the service"),
    56: ("excellent", "Praise: 'excellent'"),
    51: ("service", "Praise: good service"),
    45: ("delicious", "Praise for the food"),
    33: ("fast", "Praise for fast delivery"),
    37: ("best", "Praise: 'best app'"),
    32: ("superb", "Praise: 'superb'"),
    12: ("thank", "Thank-you messages"),
    47: ("app", "Praise: 'good app'"),
    48: ("nice", "Praise: 'nice app'"),
    28: ("swiggy", "Praise naming Swiggy"),
    58: ("aap", "Praise in Hinglish"),
    30: ("job", "Praise: 'good job'"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify, change nothing")
    args = ap.parse_args()

    applied, refused = 0, []
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT theme_id, top_terms FROM themes WHERE model = %s",
                    (MODEL,))
        terms = dict(cur.fetchall())

        for theme_id, (must_contain, name) in CURATED.items():
            found = terms.get(theme_id)
            if found is None:
                refused.append((theme_id, name, "theme no longer exists"))
                continue
            if must_contain not in found:
                refused.append((theme_id, name,
                                f"expected '{must_contain}' in top terms, got: "
                                f"{found[:60]}"))
                continue
            if not args.check:
                cur.execute("UPDATE themes SET display_name = %s "
                            "WHERE model = %s AND theme_id = %s",
                            (name, MODEL, theme_id))
            applied += 1

    verb = "would apply" if args.check else "applied"
    print(f"{verb} {applied} of {len(CURATED)} curated names")
    for theme_id, name, why in refused:
        print(f"  REFUSED [{theme_id}] {name!r} — {why}")
    if refused:
        print("\nThe clustering has changed since these names were written. "
              "Re-read the themes and update CURATED rather than forcing them.")


if __name__ == "__main__":
    main()
