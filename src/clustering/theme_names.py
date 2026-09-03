"""Give every theme a name a person can read, and mark the ones nobody can act on.

    python -m src.clustering.theme_names            # apply
    python -m src.clustering.theme_names --check    # verify only, change nothing

Clustering produces labels like "mere / karne / karte" and "poor / customer /
support". Those are honest evidence of what the algorithm found, and they are
useless as something to show a reader — a dashboard full of them looks unfinished
and tells the business nothing. This module maps every theme to a name written
from its actual contents, keeping the machine label alongside as the evidence.

The names are curated, so they can go stale: theme ids are assigned by HDBSCAN and
would be reassigned if the clustering were re-run with different parameters. Each
entry therefore records a term that must still be present in that theme's top
terms. If it is not, the name is refused rather than applied to whichever theme
inherited the id — a wrong name is worse than a raw one, because nobody would
think to check it.

NOT_ACTIONABLE marks the themes that are real but that no team can act on: two
hold reviews that are angry without saying why, and two group reviews by the
language they are written in rather than by their subject. They are excluded from
the business screens and from Power BI via themes.actionable, and stay visible on
the dashboard's "How it works" page, where being able to see the model's failures
is the point.
"""

import argparse

from src.db.connection import connection

MODEL = "sbert-domain"

# theme_id -> (a term that must still appear in the theme's top_terms, name)
#
# Names are written from each theme's top terms plus a sample of its
# highest-strength reviews, read rather than inferred from the terms alone. Where
# several themes share vocabulary — there are eleven distinct support complaints
# and eleven distinct lateness complaints — the name records what separates them,
# because two topics called "Poor customer support" are worth less than one.
CURATED = {
    # ── delivery: late ──────────────────────────────────────────────────────
    105: ("mins", "Much later than the time promised"),
    103: ("waited", "Waited an hour or more"),
    9: ("late", "Late delivery, no reply"),
    10: ("long", "Order takes too long overall"),
    79: ("everytime", "Late every single time"),
    7: ("dilevery", "Food arrived late"),
    8: ("tooo", "Order was far too late"),
    2: ("slow", "Delivery has got slower"),
    93: ("assign", "No delivery partner assigned"),
    96: ("arrives", "Delays repeating over months"),
    104: ("assigned", "Waited hours, then cancelled"),
    # ── delivery: other ─────────────────────────────────────────────────────
    86: ("location", "Delivery not available at my address"),
    90: ("rude", "Rude delivery staff"),
    15: ("night", "Problems with late-night orders"),
    81: ("cashback", "Cash on delivery not available"),
    21: ("cash", "Asking for a cash payment option"),
    67: ("buzz", "Free delivery offer did not apply"),
    66: ("seen", "Worst app I have ever used"),
    61: ("online", "Called the worst delivery app in India"),
    # ── customer support ────────────────────────────────────────────────────
    64: ("support", "Worst service and support"),
    89: ("poor", "Poor service overall"),
    65: ("care", "Worst customer care"),
    99: ("resolution", "Complaint never resolved"),
    3: ("pathetic", "Support called pathetic"),
    1: ("class", "Support called third class"),
    91: ("team", "Bad customer care"),
    87: ("horrible", "Support called horrible"),
    107: ("generic", "Only copy-paste replies"),
    100: ("chatbot", "Stuck with a chatbot, no human"),
    108: ("email", "Emails ignored, ticket closed"),
    # ── orders and refunds ──────────────────────────────────────────────────
    109: ("cancelled", "Cancelled orders and missing refunds"),
    83: ("percent", "Charged the full cancellation fee"),
    106: ("mistake", "Cannot cancel an order"),
    80: ("option", "Restaurant cancelled the order"),
    4: ("came", "Order never arrived, no refund"),
    6: ("train", "Food not delivered on trains"),
    # ── trust and fraud ─────────────────────────────────────────────────────
    82: ("fake", "Offers called fake"),
    94: ("looting", "Accused of cheating customers"),
    29: ("wasting", "Called a waste of time and money"),
    41: ("zero", "Would give zero stars"),
    97: ("uninstall", "Uninstalled or deleted the account"),
    102: ("gonna", "Will never use it again"),
    101: ("install", "Telling others not to install it"),
    # ── pricing and fees ────────────────────────────────────────────────────
    71: ("gst", "High prices, taxes and fees"),
    68: ("rupees", "Charged more than the bill showed"),
    72: ("platform", "Platform and delivery fees"),
    73: ("unnecessary", "Extra charges added at checkout"),
    11: ("rain", "Surge fee charged in the rain"),
    74: ("increase", "Prices keep going up"),
    75: ("original", "Priced above the restaurant's own price"),
    70: ("rates", "Delivery charges too high"),
    69: ("expensive", "More expensive than rivals"),
    98: ("coupon", "No usable offers left"),
    60: ("reduce", "Asking for lower prices"),
    # ── food and product quality ────────────────────────────────────────────
    88: ("chicken", "Wrong food item sent"),
    76: ("pizza", "Problems with pizza orders"),
    95: ("spoiled", "Spoiled food, nobody takes responsibility"),
    78: ("cake", "Birthday cakes arriving damaged"),
    77: ("icecream", "Ice cream arriving melted"),
    # ── app experience ──────────────────────────────────────────────────────
    54: ("attempts", "Cannot log in — too many attempts"),
    84: ("unable", "Cannot open or use the account"),
    # ── competitors and suggestions ─────────────────────────────────────────
    62: ("zomato", "Compared with Zomato"),
    63: ("improvement", "Suggestions for improvement"),
    34: ("1st", "First-time users trying it out"),
    # ── praise: delivery and food ───────────────────────────────────────────
    33: ("fastest", "Praise for fast delivery"),
    40: ("timely", "Praise for on-time delivery"),
    44: ("tasty", "Praise for fast, tasty food"),
    36: ("quick", "Praise for a quick response"),
    45: ("delicious", "Praise for the food"),
    38: ("quantity", "Praise for quality and quantity"),
    # ── praise: app and price ───────────────────────────────────────────────
    35: ("smooth", "Praise: easy to use"),
    43: ("groceries", "Praise for Instamart groceries"),
    23: ("useful", "Praise: useful app"),
    24: ("usefull", "Praise: helpful app"),
    47: ("app", "Praise: 'good app'"),
    46: ("application", "Praise: 'very good app'"),
    14: ("discounts", "Praise for the discounts"),
    13: ("affordable", "Praise for affordable prices"),
    18: ("offer", "Praise for the offers"),
    55: ("offers", "Praise: good offers"),
    # ── praise: general ─────────────────────────────────────────────────────
    57: ("good", "Praise: 'good'"),
    49: ("nice", "Praise for the service"),
    56: ("excellent", "Praise: 'excellent'"),
    51: ("service", "Praise: good service"),
    37: ("best", "Praise: 'best app'"),
    32: ("superb", "Praise: 'superb'"),
    12: ("thank", "Thank-you messages"),
    48: ("nice", "Praise: 'nice app'"),
    28: ("swiggy", "Praise naming Swiggy"),
    58: ("aap", "Praise in Hinglish"),
    30: ("job", "Praise: 'good job'"),
    42: ("food", "Praise: best food app"),
    50: ("services", "Praise: excellent service"),
    26: ("loved", "Praise: 'love it'"),
    20: ("love", "Praise: 'love Swiggy'"),
    53: ("okay", "Praise: 'okay'"),
    22: ("satisfied", "Praise: satisfied customer"),
    59: ("verry", "Praise: 'very good'"),
    19: ("fantastic", "Praise: 'amazing'"),
    17: ("awesome", "Praise: 'awesome app'"),
    31: ("forever", "Praise: 'always good'"),
    25: ("liked", "Praise: 'I like it'"),
    16: ("wow", "Praise: 'wow'"),
    27: ("overall", "Praise: 'overall good'"),
    52: ("gud", "Praise: 'gud service'"),
    5: ("stars", "Reviews about the star rating itself"),
    # ── real themes nobody can act on (see NOT_ACTIONABLE below) ────────────
    39: ("karne", "Reviews written in Hinglish"),
    0: ("कर", "Reviews written in Hindi script"),
    92: ("bad", "Bad experience, no detail given"),
    85: ("bad", "Negative, no detail given"),
}

# Excluded from the business screens and from Power BI, never from the totals.
# Two group reviews by the language they are written in rather than by subject —
# a real limitation of a model trained on English — and two hold reviews that are
# angry without saying why. All four are large enough to top a chart and lead a
# reader nowhere, which is worse than showing one topic fewer.
NOT_ACTIONABLE = {39, 0, 92, 85}


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
                cur.execute("UPDATE themes SET display_name = %s, actionable = %s "
                            "WHERE model = %s AND theme_id = %s",
                            (name, theme_id not in NOT_ACTIONABLE, MODEL, theme_id))
            applied += 1

        unnamed = sorted(set(terms) - set(CURATED))

    verb = "would apply" if args.check else "applied"
    print(f"{verb} {applied} of {len(CURATED)} curated names, "
          f"covering {applied}/{len(terms)} themes in the database")
    print(f"{len(NOT_ACTIONABLE)} marked not actionable: "
          f"{sorted(NOT_ACTIONABLE)}")
    for theme_id, name, why in refused:
        print(f"  REFUSED [{theme_id}] {name!r} — {why}")
    if unnamed:
        print(f"  {len(unnamed)} theme(s) still unnamed and will show raw "
              f"algorithm terms to the reader: {unnamed}")
    if refused:
        print("\nThe clustering has changed since these names were written. "
              "Re-read the themes and update CURATED rather than forcing them.")


if __name__ == "__main__":
    main()
