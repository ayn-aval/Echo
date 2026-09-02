"""Does a Swiggy reply tell us the complaint category, or only the star rating?

    python -m eval.reply_signal

PROJECT_PLAN.md opens Phase 4 assuming replies are "templated by complaint
category", which would make two reviews sharing a reply a free positive pair.
This script tests that assumption three ways and writes the result to
results/phase4_reply_signal.csv. All three say the same thing: the reply encodes
the rating and nothing else, so reply-based pairing is really rating-based
pairing.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.db.connection import connection

RESULTS = Path("results/phase4_reply_signal.csv")
TOPIC_WORDS = r"\b(late|delay|refund|money|charge|coupon|crash|login|cold|stale)\b"


def load():
    with connection() as conn:
        return pd.read_sql("""SELECT content, score, reply_content FROM reviews
                               WHERE app='swiggy' AND keep_for_themes
                                 AND reply_content IS NOT NULL""", conn)


def main() -> None:
    df = load()
    rows = []
    print(f"{len(df):,} kept reviews with a reply · "
          f"{df.reply_content.nunique()} distinct replies\n")

    # 1. Do replies mention any complaint topic at all?
    named = df.reply_content.str.contains(TOPIC_WORDS, case=False, regex=True)
    print(f"replies naming a specific complaint topic: {named.mean() * 100:.1f}%")
    rows.append({"test": "replies_naming_a_topic_pct", "value": round(named.mean() * 100, 2)})

    # 2. Is the reply fixed by the star rating? std 0 means one rating per reply.
    g = df.groupby("reply_content").score.agg(["count", "std", "nunique"])
    g = g[g["count"] >= 50]
    single = int((g["nunique"] == 1).sum())
    print(f"replies used 50+ times: {len(g)} · always the same rating: {single} "
          f"· median within-reply rating std: {g['std'].median():.2f}")
    rows.append({"test": "frequent_replies", "value": len(g)})
    rows.append({"test": "frequent_replies_single_rating", "value": single})
    rows.append({"test": "median_within_reply_score_std", "value": round(float(g["std"].median()), 3)})

    # 3. The real test. Hold the rating fixed, then ask whether the review text
    #    predicts which template was sent. If templates were chosen by category
    #    this is easy; if they are rotated at random it is impossible.
    for star in (1, 5):
        sub = df[df.score == star]
        vc = sub.reply_content.value_counts()
        sub = sub[sub.reply_content.isin(vc[vc >= 300].index)]
        X = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True).fit_transform(sub.content)
        y = sub.reply_content.factorize()[0]
        acc = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=3, n_jobs=-1).mean()
        base = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=3).mean()
        print(f"{star}-star ({len(sub):,} reviews, {y.max() + 1} templates): "
              f"predict template from text {acc * 100:.2f}% vs majority {base * 100:.2f}% "
              f"-> {(acc - base) * 100:+.2f}")
        rows.append({"test": f"template_prediction_lift_{star}star",
                     "value": round((acc - base) * 100, 2)})

    RESULTS.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
