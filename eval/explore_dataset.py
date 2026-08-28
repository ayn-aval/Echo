"""Exploratory analysis of the review corpus.

Writes four PNGs and a summary CSV to results/, and prints a three-sentence
description of the dataset for the README.

    python -m eval.explore_dataset

Every number in the README comes from this script, so it can be re-run and
checked rather than taken on trust.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src.db.connection import connection

RESULTS = Path("results")

# Validated palette (light surface). Single-series charts use one blue;
# multi-series charts take categorical slots in fixed order, never cycled.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax, title, subtitle=None):
    """Recessive axes and grid; text in ink colours, never a series colour."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=18)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SOFT, fontsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    with connection() as conn:
        df = pd.read_sql("""SELECT score, content, reviewed_at, word_count, lang,
                                   keep_for_themes, reply_content
                            FROM reviews WHERE app='swiggy'""", conn)
    n = len(df)
    df["week"] = pd.to_datetime(df.reviewed_at, utc=True).dt.tz_convert(
        "Asia/Kolkata").dt.to_period("W").dt.start_time

    # 1. Volume over time — one series, so no legend; the title names it.
    weekly = df.groupby("week").size()
    weekly = weekly.iloc[1:-1]  # drop partial first and last weeks
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(weekly.index, weekly.values, color=BLUE, linewidth=2)
    ax.fill_between(weekly.index, weekly.values, color=BLUE, alpha=0.10)
    ax.set_ylim(0, weekly.max() * 1.15)
    style(ax, "Swiggy review volume, weekly",
          f"{n:,} reviews collected · {df.reviewed_at.min():%d %b %Y} to {df.reviewed_at.max():%d %b %Y}")
    ax.set_ylabel("reviews per week", color=INK_SOFT, fontsize=9.5)
    save(fig, "volume_over_time.png")

    # 2. Rating distribution — one series; label every bar, there are only five.
    counts = df.score.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"{s}★" for s in counts.index], counts.values, color=BLUE, width=0.62, zorder=3)
    for x, v in enumerate(counts.values):
        ax.text(x, v + n * 0.012, f"{v:,}\n{v/n:.1%}", ha="center",
                color=INK_SOFT, fontsize=9.5, linespacing=1.4)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_yticks([])
    ax.grid(False)
    style(ax, "Ratings are bimodal, not bell-shaped",
          "Most reviewers are either angry or delighted — few are in between")
    save(fig, "rating_distribution.png")

    # 3. Length distribution — two series, so a legend is required.
    bins = [(1, 1, "1 word"), (2, 3, "2–3"), (4, 10, "4–10"),
            (11, 50, "11–50"), (51, 10**6, "50+")]
    kept = [df[(df.word_count.between(lo, hi)) & df.keep_for_themes].shape[0] for lo, hi, _ in bins]
    dropped = [df[(df.word_count.between(lo, hi)) & ~df.keep_for_themes].shape[0] for lo, hi, _ in bins]
    labels = [lab for _, _, lab in bins]
    x = range(len(bins))
    fig, ax = plt.subplots(figsize=(8, 4))
    # Stacked, not grouped: the split is near all-or-nothing per bucket, so
    # grouped bars leave confusing empty slots. The 2px surface-coloured edge
    # is the gap between stacked segments.
    ax.bar(list(x), kept, width=0.58, color=BLUE, label="kept for themes",
           zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.bar(list(x), dropped, width=0.58, bottom=kept, color=ORANGE,
           label="filtered out", zorder=3, edgecolor=SURFACE, linewidth=2)
    for i, (k, d) in enumerate(zip(kept, dropped)):
        ax.text(i, k + d + n * 0.012, f"{k + d:,}", ha="center",
                color=INK_SOFT, fontsize=9.5)
    ax.set_ylim(0, max(k + d for k, d in zip(kept, dropped)) * 1.15)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("reviews", color=INK_SOFT, fontsize=9.5)
    style(ax, "Most reviews are too short to carry a theme",
          f"{sum(dropped):,} of {n:,} filtered out — single words and emoji-only reviews")
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    save(fig, "length_distribution.png")

    # 4. Language mix — 100% stacked bar; a sliver is the honest story here.
    lang = df.lang.value_counts()
    names = {"en": "English", "hinglish": "Hinglish (romanised)", "hi": "Devanagari"}
    fig, ax = plt.subplots(figsize=(9, 2.4))
    left = 0
    for (key, colour) in zip(["en", "hinglish", "hi"], [BLUE, ORANGE, AQUA]):
        v = lang.get(key, 0) / n * 100
        ax.barh([0], [v], left=left, color=colour, height=0.5,
                label=f"{names[key]} — {lang.get(key,0):,} ({v:.1f}%)", zorder=3)
        left += v
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("% of corpus", color=INK_SOFT, fontsize=9.5)
    ax.grid(False)
    style(ax, "Language mix",
          "Detected by a transparent word-list rule, not a language detector — see PROJECT_PLAN.md")
    leg = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.45), ncols=3, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    save(fig, "language_mix.png")

    # Summary CSV — every README number reproducible from here.
    kept_n = int(df.keep_for_themes.sum())
    stats = {
        "reviews_total": n,
        "reviews_kept_for_themes": kept_n,
        "distinct_texts_kept": int(df[df.keep_for_themes].content.nunique()),
        "date_from": df.reviewed_at.min().date(),
        "date_to": df.reviewed_at.max().date(),
        "days_covered": (df.reviewed_at.max() - df.reviewed_at.min()).days,
        "avg_rating_all": round(df.score.mean(), 2),
        "pct_1_star_all": round((df.score == 1).mean() * 100, 1),
        "pct_5_star_all": round((df.score == 5).mean() * 100, 1),
        "pct_1_star_kept": round((df[df.keep_for_themes].score == 1).mean() * 100, 1),
        "pct_with_developer_reply": round(df.reply_content.notna().mean() * 100, 1),
        "median_words_kept": int(df[df.keep_for_themes].word_count.median()),
        "max_words": int(df.word_count.max()),
        "pct_hinglish": round((df.lang == "hinglish").mean() * 100, 1),
        "pct_devanagari": round((df.lang == "hi").mean() * 100, 1),
    }
    pd.Series(stats).to_csv(RESULTS / "dataset_summary.csv", header=["value"])
    print(f"  wrote {RESULTS / 'dataset_summary.csv'}")

    print("\n--- three sentences for the README ---\n")
    print(f"Echo analyses {n:,} Google Play reviews of Swiggy, collected over "
          f"{stats['days_covered']} days from {stats['date_from']:%d %B %Y} to "
          f"{stats['date_to']:%d %B %Y}.")
    print(f"Ratings are sharply bimodal — {stats['pct_1_star_all']}% one-star and "
          f"{stats['pct_5_star_all']}% five-star — and {stats['pct_with_developer_reply']}% of "
          f"reviews carry a reply from Swiggy, which later becomes the training signal for "
          f"domain adaptation.")
    print(f"After filtering emoji-only and single-word reviews, {kept_n:,} reviews "
          f"({kept_n/n:.0%}) carry enough text to cluster into themes, containing "
          f"{stats['distinct_texts_kept']:,} distinct texts.")


if __name__ == "__main__":
    main()
