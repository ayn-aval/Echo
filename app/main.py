"""Echo — the results board.

    streamlit run app/main.py

One page, top to bottom: what the project claims, then the evidence for each
claim, then where it falls short. There is no navigation because there is
nothing to navigate to.

Section order and layout follow `app/results.py` for content and `app/theme.py`
for style. The layout rule that matters is in theme.py's docstring: a section
containing a Streamlit widget must sit inside `st.container(key=...)`, and a
section containing none is emitted as a single `st.markdown` call with its own
internal grid. Only Figures 04/05 and 07 hold widgets; everything else on this
page is one call, which is why none of it can drift out of its band.

app/label.py and app/audit.py are separate internal tools, run directly.
"""

import html as _html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import theme

st.set_page_config(page_title="Echo — building a review search engine from scratch",
                   page_icon=theme.MARK, layout="wide",
                   initial_sidebar_state="collapsed")
theme.boot()

import results as R  # noqa: E402  (after boot, so the stylesheet lands first)
import shared  # noqa: E402
from shared import MODEL, sql  # noqa: E402

WEEKS_SHOWN = 16    # enough to hold the July spike, few enough for legible bars


def esc(text, limit=None):
    out = " ".join(str(text).split())
    if limit and len(out) > limit:
        out = out[:limit].rstrip() + "…"
    return _html.escape(out)


# ── top bar ───────────────────────────────────────────────────────────────────
theme.html(
    '<div class="topbar"><span class="mark">ECHO</span><span class="sep"></span>'
    f'<span class="what">{R.STANDFIRST}</span>'
    '<span class="lab" style="margin-left:auto;color:var(--accent)">Results</span></div>')

# ── headline ──────────────────────────────────────────────────────────────────
# One call, one grid. The reference used three st.columns here, which put the
# KPIs outside the padded band and left them flush against the window edge.
cells = ""
for i, k in enumerate(R.HEADLINE):
    edge = ("border-right:1px solid var(--line);padding-right:44px"
            if i < len(R.HEADLINE) - 1 else "padding-left:4px")
    cells += (
        f'<div style="{edge}">'
        f'<div class="lab" style="color:var(--accent);margin-bottom:16px">{k["label"]}</div>'
        '<div style="display:flex;align-items:flex-end;gap:14px">'
        f'<span class="kpi-n">{k["value"]}</span>'
        f'<span class="chip{" ghost" if k["ghost"] else ""}" '
        f'style="margin-bottom:9px">{k["chip"]}</span></div>'
        f'<div class="kpi-note">{k["note"]}</div></div>')

theme.html(
    '<div style="padding:60px 56px 54px;border-bottom:1px solid var(--line)">'
    '<div class="lab" style="color:var(--faint);margin-bottom:26px">Headline</div>'
    '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:44px">'
    f'{cells}</div></div>')


# ── Figure 01 · the rebuild against the paper ─────────────────────────────────
def figure_01() -> str:
    lo, hi = R.STS_SCALE
    pos = lambda v: (v - lo) / (hi - lo) * 100
    rows = ""
    for task, paper, ours in R.STS:
        gap = ours - paper
        fill = "var(--accent)" if gap >= 0 else "var(--dim)"
        gap_col = "var(--accent)" if gap >= 0 else "var(--muted)"
        rows += (
            '<div style="display:grid;grid-template-columns:130px 1fr 112px;gap:20px;'
            'align-items:center;padding:11px 0;border-bottom:1px solid var(--hair)">'
            f'<span style="font-size:13px;color:var(--muted)">{task}</span>'
            '<div style="position:relative;height:22px;background:var(--track)">'
            f'<div style="position:absolute;left:0;top:0;height:22px;'
            f'width:{pos(ours):.1f}%;background:{fill}"></div>'
            f'<div style="position:absolute;top:-3px;left:{pos(paper):.1f}%;width:2px;'
            'height:28px;background:var(--text)"></div></div>'
            '<div style="display:flex;align-items:baseline;justify-content:flex-end;gap:10px">'
            f'<span class="n" style="font-size:15px;font-weight:800">{ours:.2f}</span>'
            f'<span class="n" style="font-size:12px;color:{gap_col};width:42px;'
            f'text-align:right">{"+" if gap >= 0 else "−"}{abs(gap):.2f}</span></div></div>')

    foot = "".join(
        f'<div><div class="lab" style="color:var(--faint);margin-bottom:8px">{label}</div>'
        f'<div class="n" style="font-size:28px;font-weight:800">{value}</div>'
        f'<div style="font-size:12px;color:var(--muted);margin-top:4px">{note}</div></div>'
        for label, value, note in R.STS_FOOT)

    return (
        '<div class="figlab"><span class="lab">Figure 01</span>'
        '<h2>How close the rebuild came</h2></div>'
        '<p class="figsub">Seven public test sets where people rated how alike two '
        'sentences are, scored out of 100. Bar = this project, line = the published '
        'paper.</p>'
        + rows +
        '<div style="display:flex;gap:24px;margin-top:16px;font-size:11.5px;'
        'color:var(--muted);align-items:center">'
        '<span style="display:flex;align-items:center;gap:7px">'
        '<span style="width:14px;height:8px;background:var(--accent)"></span>this project</span>'
        '<span style="display:flex;align-items:center;gap:7px">'
        '<span style="width:2px;height:14px;background:var(--text)"></span>the paper</span>'
        f'<span style="margin-left:auto">bars start at {R.STS_SCALE[0]}</span></div>'
        '<div style="margin-top:26px;padding-top:22px;border-top:1px solid var(--line);'
        f'display:grid;grid-template-columns:1fr 1fr;gap:20px">{foot}</div>')


# ── Figure 02 · search quality ────────────────────────────────────────────────
def figure_02() -> str:
    bars = ""
    for system, score, primary in R.RETRIEVAL:
        fill = "var(--accent)" if primary else "var(--dim)"
        num = "var(--accent)" if primary else "var(--text)"
        name = "var(--text)" if primary else "var(--muted)"
        bars += (
            '<div style="margin-bottom:24px">'
            '<div style="display:flex;justify-content:space-between;align-items:baseline;'
            'margin-bottom:9px">'
            f'<span style="font-size:13.5px;color:{name};'
            f'font-weight:{800 if primary else 500}">{system}</span>'
            f'<span class="n" style="font-size:22px;font-weight:800;color:{num}">'
            f'{score:.1f}</span></div>'
            + theme.hbar(score / 10 * 100, fill, 12) + "</div>")

    stats = "".join(
        f'<div><div class="lab" style="color:var(--faint);margin-bottom:8px">{label}</div>'
        f'<div class="n" style="font-size:28px;font-weight:800;'
        f'color:{"var(--accent)" if accent else "var(--text)"}">{value}</div>'
        f'<div style="font-size:12px;color:var(--muted);margin-top:4px">{note}</div></div>'
        for label, value, note, accent in R.RETRIEVAL_STATS)

    return (
        '<div class="figlab"><span class="lab">Figure 02</span>'
        '<h2>Search quality</h2></div>'
        '<p class="figsub">Out of the 10 reviews a search returns, how many are '
        'really about what was asked. Judged on 26 searches checked by hand.</p>'
        + bars +
        '<div style="margin-top:14px;padding-top:22px;border-top:1px solid var(--line);'
        f'display:grid;grid-template-columns:1fr 1fr;gap:20px">{stats}</div>')


theme.html(
    '<div style="display:grid;grid-template-columns:1.4fr 1fr;'
    'border-bottom:1px solid var(--line)">'
    '<div style="padding:42px 44px 46px 56px;border-right:1px solid var(--line)">'
    f'{figure_01()}</div>'
    f'<div style="padding:42px 56px 46px 44px">{figure_02()}</div></div>')

# ── Figure 03 · same meaning, different words ─────────────────────────────────
cards = "".join(
    '<div class="card" style="display:flex;flex-direction:column;gap:14px;min-height:184px">'
    f'<div class="lab" style="color:var(--label)">{label}</div>'
    f'<div style="font-size:15.5px;line-height:1.45">“{esc(text)}”</div>'
    '<div style="margin-top:auto;display:flex;align-items:baseline;gap:8px">'
    + (f'<span class="n" style="font-size:26px;font-weight:800;color:var(--accent)">'
       f'{match}</span><span style="font-size:11.5px;color:var(--faint)">'
       'as close in meaning to the first</span>'
       if match else
       '<span style="font-size:11.5px;color:var(--faint)">the review the other '
       'two are compared with</span>')
    + "</div></div>"
    for label, text, match in R.COHERENCE)

topic = R.COHERENCE_TOPIC
cards += (
    '<div style="background:var(--accent);color:var(--bg);padding:24px;display:flex;'
    'flex-direction:column">'
    f'<div class="lab" style="opacity:.72">{topic["label"]}</div>'
    f'<div style="font-size:19px;font-weight:800;line-height:1.2;margin-top:14px">'
    f'{topic["name"]}</div><div style="margin-top:auto">'
    f'<div class="n" style="font-size:34px;font-weight:800;line-height:1">'
    f'{topic["count"]}</div>'
    f'<div style="font-size:12px;font-weight:600;opacity:.78">{topic["meta"]}</div>'
    "</div></div>")

theme.html(
    '<div style="padding:44px 56px 50px;border-bottom:1px solid var(--line)">'
    '<div class="figlab"><span class="lab">Figure 03</span>'
    '<h2>Same meaning, different words</h2></div>'
    '<p class="figsub">Three complaints the model filed under one topic. They have '
    'no complaint word in common — it matched <b>install</b>, <b>use</b> and '
    '<b>order</b> as the same idea.</p>'
    '<div style="display:grid;grid-template-columns:repeat(3,1fr) 250px;gap:20px">'
    f'{cards}</div></div>')


# ── Figures 04 and 05 ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def biggest_topics(limit: int = 10):
    return sql("""SELECT coalesce(display_name, label) AS name, n_rows
                    FROM themes
                   WHERE model = %s AND actionable AND avg_rating <= 2.5
                   ORDER BY n_rows DESC LIMIT %s""", (MODEL, limit))


@st.cache_data(ttl=600, show_spinner=False)
def weekly(theme_id: int, weeks: int = WEEKS_SHOWN):
    """The topic's last N weeks, and which of them were flagged as unusual."""
    series = sql("""SELECT week_start, reviews FROM theme_weekly
                     WHERE model = %s AND theme_id = %s
                     ORDER BY week_start DESC LIMIT %s""", (MODEL, theme_id, weeks))
    series = series.sort_values("week_start").reset_index(drop=True)
    flagged = sql("""SELECT week_start FROM theme_alerts
                      WHERE model = %s AND theme_id = %s""", (MODEL, theme_id))
    weeks_flagged = set(flagged.week_start)
    return series, [i for i, w in enumerate(series.week_start) if w in weeks_flagged]


with st.container(key="split_topics"):
    left, right = st.columns([1, 1.15])

    with left:
        with st.container(key="padl_topics"):
            topics = biggest_topics()
            peak = int(topics.n_rows.max())
            rows = "".join(
                '<div style="display:grid;grid-template-columns:22px 1fr 124px 58px;'
                'gap:14px;align-items:center;padding:10px 0;'
                'border-bottom:1px solid var(--hair)">'
                f'<span class="n" style="font-size:11.5px;color:var(--faint)">{i:02d}</span>'
                f'<span style="font-size:13.5px">{esc(t.name)}</span>'
                + theme.hbar(100 * int(t.n_rows) / peak) +
                '<span class="n" style="font-size:13px;text-align:right;font-weight:600">'
                f'{int(t.n_rows):,}</span></div>'
                for i, t in enumerate(topics.itertuples(), 1))
            theme.fig_label(
                "Figure 04", "Biggest topics",
                "The 10 largest of 57 complaint topics. The model found 110 in all — "
                "the rest are praise, or too vague to act on.")
            theme.html(rows)

    with right:
        with st.container(key="padr_topics"):
            theme.fig_label(
                "Figure 05", "One topic, week by week",
                "Each topic is measured against its own normal level, never against "
                "another topic.")
            picked = theme.chip_row([s["label"] for s in R.SERIES], "series",
                                    widths=[1.3, 1.2, 1.15, 0.5])
            chosen = R.SERIES[picked]
            series, flags = weekly(chosen["theme_id"])
            note_col = "var(--accent)" if flags else "var(--muted)"
            first, last = series.week_start.iloc[0], series.week_start.iloc[-1]
            theme.html(
                '<div style="height:20px"></div>'
                + theme.column_chart(series.reviews.tolist(), flags) +
                '<div style="display:flex;justify-content:space-between;font-size:11px;'
                'margin-top:10px;color:var(--faint);gap:16px">'
                f'<span>{first:%d %b}</span>'
                f'<span style="color:{note_col};text-align:center">{chosen["note"]}</span>'
                f'<span>{last:%d %b}</span></div>')

# ── Figure 06 · where it falls short ──────────────────────────────────────────
tiles = "".join(
    '<div class="card" style="display:flex;flex-direction:column;gap:12px;min-height:176px">'
    f'<div class="n" style="font-size:38px;font-weight:800;line-height:1;'
    f'color:var(--accent)">{share}</div>'
    f'<div style="font-size:15px;font-weight:800;line-height:1.25">{name}</div>'
    f'<div style="font-size:12.5px;line-height:1.5;color:var(--muted);margin-top:auto">'
    f'{note}</div></div>'
    for share, name, note in R.FAILURES)

theme.html(
    '<div style="padding:44px 56px 50px;border-bottom:1px solid var(--line)">'
    '<div class="figlab"><span class="lab">Figure 06</span>'
    '<h2>Where it falls short</h2></div>'
    '<p class="figsub">Four limits found during testing and left in the results '
    'rather than tuned away.</p>'
    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px">'
    f'{tiles}</div></div>')

# ── pipeline ──────────────────────────────────────────────────────────────────
steps = ""
for i, (number, name, detail) in enumerate(R.PIPELINE):
    on = i == R.PIPELINE_HIGHLIGHT
    steps += (
        f'<div style="border-left:2px solid '
        f'{"var(--accent)" if on else "var(--step)"};padding:0 18px 0 16px">'
        f'<div class="n lab" style="margin-bottom:9px;'
        f'color:{"var(--accent)" if on else "var(--muted)"}">{number}</div>'
        f'<div style="font-size:13.5px;font-weight:800;line-height:1.3;'
        f'margin-bottom:5px">{name}</div>'
        f'<div style="font-size:11.5px;line-height:1.45;color:var(--muted)">'
        f'{detail}</div></div>')

theme.html(
    '<div style="padding:44px 56px 46px;border-bottom:1px solid var(--line)">'
    '<div class="lab" style="color:var(--faint);margin-bottom:24px">How it works</div>'
    f'<div style="display:grid;grid-template-columns:repeat(7,1fr)">{steps}</div></div>')

# ── Figure 07 · search it yourself ────────────────────────────────────────────
# The only part of the page that runs the model rather than reporting on it.
with st.container(key="band_search"):
    theme.fig_label(
        "Figure 07", "Search it yourself",
        "Type a complaint the way a customer would say it. The words do not have "
        "to appear in the review.")

    # Streamlit forbids writing to a key an instantiated widget owns, so the text
    # the example chips insert lives under its own key and is passed as `value`.
    st.session_state.setdefault("seed", "")
    box, _ = st.columns([1.45, 1.55])
    with box:
        typed = st.text_input("Search", value=st.session_state["seed"],
                              label_visibility="collapsed",
                              placeholder="refund never came back to my account")
    theme.html('<div class="lab" style="color:var(--faint);padding-top:18px">'
               "Or try one of these</div>")
    with st.container(key="chips_examples"):
        cols = st.columns([1.55, 1.05, 1.6, 2.8])
        for col, example in zip(cols, R.SEARCH_EXAMPLES):
            with col:
                if st.button(example, key=f"ex_{example}"):
                    st.session_state["seed"] = example
                    st.rerun()

    query = (typed or "").strip()
    if not query:
        theme.html('<div style="height:18px"></div><p class="figsub" '
                   'style="margin:0 !important">Nothing searched yet.</p>')
    else:
        shared.get_search()
        shared.get_reranker()
        from src.search.rerank import search_reranked

        # Stage one alone matches on any shared word — "refund never came back"
        # returned reviews whose only link was "never". This is the same two-stage
        # pipeline Figure 02 reports, so the demo shows what the number claims.
        hits = search_reranked(query, k=R.SEARCH_K)
        meta = sql("""SELECT review_id, score AS stars, reviewed_at::date AS day
                        FROM reviews
                       WHERE app = 'swiggy' AND review_id = ANY(%s)""",
                   (list(hits.review_id),))
        topics = sql("""SELECT rt.review_id,
                               coalesce(t.display_name, t.label) AS topic
                          FROM review_themes rt
                          JOIN themes t ON t.model = rt.model
                                       AND t.theme_id = rt.theme_id
                         WHERE rt.model = %s AND rt.review_id = ANY(%s)""",
                     (MODEL, list(hits.review_id)))
        hits = (hits.merge(meta, on="review_id", how="left")
                    .merge(topics, on="review_id", how="left"))
        unhappy = int((hits.stars <= 2).sum())

        found = "".join(
            '<div class="review"><div class="m">'
            f'<b style="color:var(--text)">{int(r.stars)} star'
            f'{"" if int(r.stars) == 1 else "s"}</b>'
            + (f'<span class="tag">{esc(r.topic)}</span>'
               if isinstance(r.topic, str) else "")
            + f'<span style="margin-left:auto">{r.day:%d %b %Y}</span></div>'
            f'<div style="font-size:14.5px;line-height:1.5">{esc(r.content)}</div></div>'
            for r in hits.itertuples())

        theme.html(
            '<div style="height:26px"></div>'
            '<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;'
            'margin-bottom:12px">'
            f'<h2 style="font-size:21px">{unhappy} of these {len(hits)} '
            'reviews are from unhappy customers</h2>'
            f'<span style="font-size:12.5px;color:var(--muted)">'
            f'{hits.stars.mean():.1f} stars on average · closest in meaning first'
            f'</span></div>{found}')

# ── footer ────────────────────────────────────────────────────────────────────
theme.html(
    '<div style="padding:32px 56px 46px;display:flex;gap:40px;align-items:flex-start;'
    'border-top:1px solid var(--line)">'
    f'<div class="foot" style="max-width:66ch">{R.FOOTER_LEFT}</div>'
    f'<div class="foot" style="margin-left:auto;text-align:right">{R.FOOTER_RIGHT}</div>'
    "</div>")
