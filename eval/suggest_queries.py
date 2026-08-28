"""Mine candidate search queries from the complaint corpus.

Themes don't exist until Phase 5, so this is a data-driven stand-in for the theme
distribution: find the phrases that are common in 1- and 2-star reviews and rare in
4- and 5-star ones. Those phrases are what people actually complain about, in their
own words, which makes them realistic queries for the retrieval evaluation.

    python -m eval.suggest_queries          # writes results/candidate_queries.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.db.connection import connection

OUT = Path("results/candidate_queries.csv")
N_CANDIDATES = 70


def load():
    with connection() as conn:
        return pd.read_sql("""SELECT score, lower(content) AS content
                                FROM reviews
                               WHERE app='swiggy' AND keep_for_themes
                                 AND word_count >= 4""", conn)


def main() -> None:
    df = load()
    complaints = df[df.score <= 2].content.tolist()
    praise = df[df.score >= 4].content.tolist()
    print(f"{len(complaints):,} complaint reviews vs {len(praise):,} positive "
          f"(4+ words each)")

    # Document frequency of each phrase in both groups. A phrase that appears in
    # many complaints and few positives is a complaint topic rather than filler.
    vec = CountVectorizer(ngram_range=(2, 5), min_df=25, binary=True,
                          max_features=120000)
    c_counts = vec.fit_transform(complaints).sum(axis=0).A1
    p_counts = vec.transform(praise).sum(axis=0).A1
    terms = vec.get_feature_names_out()

    c_rate = c_counts / len(complaints)
    p_rate = (p_counts + 1) / (len(praise) + 1)   # +1 so a zero doesn't divide
    lift = c_rate / p_rate

    scored = pd.DataFrame({"phrase": terms, "complaints": c_counts,
                           "positives": p_counts, "lift": lift})

    # Frequency alone always crowns stopword bigrams — "the order", "is not",
    # "of the". A usable query needs at least two content words, so require that
    # explicitly rather than hoping the ranking sorts it out.
    def content_words(phrase):
        return [w for w in phrase.split() if w not in ENGLISH_STOP_WORDS]

    scored["content"] = scored.phrase.map(lambda p: len(content_words(p)))
    scored = (scored
              .query("complaints >= 40 and lift >= 4 and content >= 2")
              .sort_values("complaints", ascending=False))

    # Drop phrases that are just a longer/shorter form of one already chosen.
    kept = []
    for _, row in scored.iterrows():
        if any(row.phrase in k or k in row.phrase for k in kept):
            continue
        kept.append(row.phrase)
        if len(kept) >= N_CANDIDATES:
            break

    out = scored[scored.phrase.isin(kept)].copy()
    out["words"] = out.phrase.str.split().str.len()
    out = out.sort_values("complaints", ascending=False)
    OUT.parent.mkdir(exist_ok=True)
    out[["phrase", "complaints", "positives", "lift"]].to_csv(OUT, index=False)

    print(f"\n{len(out)} candidate queries -> {OUT}\n")
    print(f"  {'phrase':44} {'in 1-2*':>8} {'in 4-5*':>8} {'lift':>6}")
    for _, r in out.head(45).iterrows():
        print(f"  {r.phrase:44} {r.complaints:8,} {r.positives:8,} {r.lift:6.1f}")


if __name__ == "__main__":
    main()
