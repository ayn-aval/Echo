"""The standing prose: the one-line definition, the header meta, the dateline.

Kept out of the screens because the ink header shows the same sentence on every
one of them, and a definition that drifts between screens is worse than none.

Anything with a number in it is built from a query rather than typed, so the
copy cannot quietly go stale when the corpus is rebuilt.
"""

from datetime import timedelta

from shared import week_start

ONE_LINER = ("Reads every Google Play review of the Swiggy app and ranks the "
             "problems customers actually hit — so you know what to fix on "
             "Monday without reading 13,800 reviews a month.")


HEADER_META = "100,000 reviews<br>18 Jan – 26 Aug 2026"


def dateline() -> str:
    """Names the week the briefing reports on, from the data itself.

    The corpus is static and ends on 26 Aug 2026, so "this week" would be a
    fiction. The real last *complete* week is stated instead. The end date is a
    timedelta rather than day + 6, which would print "31-37 Aug" in any week
    that crosses a month boundary.
    """
    start = week_start()
    end = start + timedelta(days=6)
    return (f"Week of {start:%d} – {end:%d %b %Y} · latest complete week")


SCOPE_FOOTNOTES = [
    "<b>Scope:</b> Google Play only, 18 Jan – 26 Aug 2026, 100,000 reviews.",
    "<b>Reviewers skew angry</b> — these counts measure noise, not how many "
    "customers were affected.",
    "<b>Portfolio project</b> on public data. Not an internal Swiggy tool.",
]

FIVE_MINUTES = [
    "Walk into standup with the top three problems and the review count behind "
    "each one.",
    "Check whether last sprint's fix moved its number, on that problem's own "
    "weekly line.",
    "Pull verbatim quotes for a deck by searching on meaning, not by guessing "
    "keywords.",
    "Find out a problem has spiked from this screen, rather than from the store "
    "rating a month later.",
]
