# Connecting Power BI to Echo

A second portfolio artifact: the same Postgres tables, read by Power BI, shaped as
an executive report. Nothing is recomputed — Power BI reads what the Python
pipeline already wrote, so the two never disagree.

## Connect

1. **Get Data → PostgreSQL database.** Server `localhost`, port `5432`, database
   as in your `.env`. Choose **Import**, not DirectQuery: the tables are small
   (3,342 rows in `theme_weekly`) and Import makes the report fast and portable.
2. Authenticate with the same credentials as `.env`. Power BI stores them in its
   own credential manager; do not put them in the report file.
3. **Load these four tables only:**

| table | rows | why |
|---|---|---|
| `theme_weekly` | 3,342 | the fact table — one row per topic per complete week |
| `themes` | 110 | topic names, business area, average rating |
| `theme_alerts` | 16 | flagged weeks with the baseline behind each |
| `reviews` | 100,000 | only if you need review-level drill-through |

`theme_weekly` was built for this. Partial weeks are already excluded, and it
carries both a count and a share, so no DAX has to reproduce that logic.

## Model

- Relationship: `themes[theme_id]` **1 → ∗** `theme_weekly[theme_id]`, single
  direction. Same for `theme_alerts[theme_id]`.
- Add a **date table** (`CALENDARAUTO()`) and relate it to
  `theme_weekly[week_start]`. Without a marked date table, time intelligence
  functions silently return wrong results.
- Mark `themes[category]` as the hierarchy level above `display_name`, so a
  reader can drill from a business area into its topics.

Two measures worth writing rather than dragging fields onto a canvas:

```
Unhappy share =
DIVIDE(SUM(theme_weekly[unhappy]), SUM(theme_weekly[reviews]))

Reviews vs 8-week average =
VAR Current = SUM(theme_weekly[reviews])
VAR Baseline =
    AVERAGEX(
        DATESINPERIOD('Date'[Date], MIN('Date'[Date]) - 1, -56, DAY),
        CALCULATE(SUM(theme_weekly[reviews])))
RETURN DIVIDE(Current - Baseline, Baseline)
```

## The report — one page, four visuals

**1. Cards, top row.** Total reviews, average rating, unhappy share, open alerts.
Four numbers, no chart. A single number does not need a visual.

**2. Line chart — unhappy reviews per week, by business area.** Weeks on the
axis, one line per area, at most six lines. This is the executive view: which
part of the business is getting worse.

**3. Bar chart — topics ranked by unhappy reviews, current period.** Horizontal,
one colour, sorted descending. Length already encodes the value, so a colour ramp
on the same field adds nothing.

**4. Table — alerts,** with topic, week, reviews, baseline mean, and z. Include
the baseline: an alert nobody can audit gets blindly followed or blindly ignored.

Slicers across the top: date range, business area. Match the app so the two tell
the same story.

## Visuals that would mislead here — and why

**A pie or donut of topic share.** Two independent problems. There are 110
topics, and a pie stops being readable past about six slices. Worse, under any
slicer the slices are the share *of the filtered subset*, so a topic can appear
to grow when the filter narrowed. Use a ranked bar.

**Dual-axis: review count and average rating on one chart.** The two y-scales are
aligned arbitrarily, so the chart invents a relationship the data does not
contain. This is the most common misleading chart in business reporting. Use two
charts stacked, sharing an x-axis.

**Stacked area over topics through time.** The set of topics with any reviews
changes week to week, so bands appear and vanish and every band above the bottom
one moves for reasons that have nothing to do with it. Use lines, capped at six
series.

**A map.** There is no location data in this dataset. A map of India shaded by
review count would be invented.

**Word cloud of review text.** It ranks by raw frequency, so it returns "order",
"app", "food" — the words this project exists to look past. The topic model is
the answer to what a word cloud pretends to answer.

**Week-on-week % change without a floor.** A topic going 1 → 3 reviews is +200%
and will top any sorted table. Filter to topics with at least 15 reviews, the
same floor `src/analytics/alerts.py` uses.

**The most recent week, if you rebuild the series yourself.** The corpus ends
mid-week. `theme_weekly` already drops partial weeks; if you point Power BI at
`reviews` directly and bucket by week in DAX, the newest bucket will be three
days long and every topic will appear to have collapsed.

## Refreshing

Re-run the Python pipeline, then refresh in Power BI:

```bash
python -m src.analytics.weekly      # rebuild the series
python -m src.analytics.alerts      # re-detect
```

Power BI Desktop with an Import model needs a manual **Refresh**. Scheduled
refresh needs the on-premises data gateway pointed at this Postgres instance,
which is out of scope here and worth saying rather than implying the report
updates itself.
