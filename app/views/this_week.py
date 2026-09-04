"""This week — the briefing: what happened, and the one thing to decide.

Reports the last COMPLETE week, named by its real dates. The corpus is static
and ends on 26 Aug 2026, so a screen headed "this week" would be a fiction; the
dateline in the tab row says which week these numbers are.

The decision block deliberately uses the sharpest spike **on record** with its
own date rather than something from the latest week. Spikes are rare — sixteen
in seven months — and inventing urgency for a quiet week would be the dishonest
version of this screen.
"""

import html as _html

import design
import prose
import streamlit as st
from shared import ALL_REVIEWS, MODEL, in_week, sql, week_start


def clean(text, limit=None):
    out = " ".join(str(text).split())
    if limit and len(out) > limit:
        out = out[:limit].rstrip() + "…"
    return _html.escape(out)


def render() -> None:
    start = week_start()
    wk = sql(f"""
        SELECT count(*) AS reviews,
               count(*) FILTER (WHERE score <= 2) AS unhappy,
               round(avg(score)::numeric, 1) AS rating
          FROM reviews WHERE {ALL_REVIEWS} {in_week('reviews')}""").iloc[0]
    prior = sql(f"""
        SELECT round(avg(score)::numeric, 1) AS rating
          FROM reviews WHERE {ALL_REVIEWS}
           AND reviewed_at < '{start}' AND reviewed_at >= '{start}'::date - 28
        """).rating.iloc[0]

    # The largest single COMPLAINT that week. Without the rating filter the
    # biggest group is "Praise: 'good'", and a screen headed "unhappy" would
    # then name a compliment as its largest problem.
    top = sql("""
        SELECT coalesce(t.display_name, t.label) AS name, w.reviews
          FROM theme_weekly w
          JOIN themes t ON t.model = w.model AND t.theme_id = w.theme_id
         WHERE w.model = %s AND w.week_start = %s AND t.actionable
           AND t.avg_rating <= 2.5
         ORDER BY w.reviews DESC LIMIT 1""", (MODEL, start))

    share = 100 * int(wk.unhappy) / max(int(wk.reviews), 1)

    # ── verdict and the two numbers ─────────────────────────────────────────
    left, right = st.columns([1, 0.45], gap="small")
    with left:
        head = (f"{int(wk.unhappy):,} of last week's {int(wk.reviews):,} "
                f"reviewers were unhappy.")
        body = (f"The largest single problem was “{clean(top['name'].iloc[0])}”, "
                f"raised by <b>{int(top.reviews.iloc[0])}</b> of them."
                if not top.empty else "")
        design.html(f'<div class="pad"><div class="kicker">'
                    f'The week in one sentence</div>'
                    f'<div class="verdict">{head}</div>'
                    f'<p class="lede">{body}</p></div>')
        a, b, _ = st.columns([1.2, 1.4, 1.1])
        with a:
            if st.button("Open the fix list  →", type="primary", key="go_fix"):
                st.session_state["screen"] = "fix"
                st.rerun()
        with b:
            if st.button("Read the actual reviews", key="go_ask"):
                st.session_state["screen"] = "ask"
                st.rerun()
        design.html('<div style="height:30px"></div>')
    with right:
        for number, label in (
            (f"{share:.0f}%",
             f"of last week's reviews were 1 or 2 stars — "
             f"{int(wk.unhappy):,} unhappy customers."),
            (f"{float(wk.rating):.1f}",
             f"average stars last week, against {float(prior):.1f} over the "
             f"four weeks before it."),
        ):
            design.html(
                '<div style="padding:24px 32px;'
                f'border-left:1px solid {design.DIVIDER};'
                f'border-bottom:1px solid {design.DIVIDER}">'
                f'<div class="big">{number}</div>'
                f'<div class="bigl">{label}</div></div>')

    design.rule()

    # ── the one thing that needs a decision ─────────────────────────────────
    spike = sql("""
        SELECT a.week_start, a.reviews, a.baseline_mean, a.kind, a.theme_id,
               coalesce(t.display_name, t.label) AS name, t.category
          FROM theme_alerts a
          JOIN themes t ON t.model = a.model AND t.theme_id = a.theme_id
         WHERE a.model = %s AND t.actionable
         ORDER BY a.z DESC NULLS LAST LIMIT 1""", (MODEL,))
    if not spike.empty:
        s = spike.iloc[0]
        times = (s.reviews / float(s.baseline_mean)) if s.baseline_mean else 0
        quote = sql("""
            SELECT r.content FROM review_themes rt
              JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
             WHERE rt.model = %s AND rt.theme_id = %s AND r.word_count BETWEEN 10 AND 40
             ORDER BY rt.strength DESC LIMIT 1""", (MODEL, int(s.theme_id)))
        design.poster(
            kicker="The sharpest jump on record",
            head=(f"“{clean(s['name'])}” ran at {times:.1f} times its normal "
                  f"rate in the week of {s.week_start:%d %B}."),
            body=(f"{s.reviews} reviews that week against about "
                  f"{s.baseline_mean:.0f} in a normal one, in "
                  f"{clean(s.category).lower()}. Every problem is compared "
                  f"against its own history, never against other problems."),
            side_kicker="What one of them said",
            side=(f"“{clean(quote.content.iloc[0], 190)}”"
                  if not quote.empty else ""))
        design.rule()

    # ── movers ──────────────────────────────────────────────────────────────
    design.html(
        '<div style="padding:30px 48px 14px;display:flex;align-items:baseline;'
        'gap:16px;flex-wrap:wrap"><h2 style="font-size:25px">What moved last '
        'week</h2><span style="font-size:13px;color:' + design.MUTED + '">'
        "each problem against its own eight-week average, never against other "
        "problems. Only problems averaging ten or more reviews a week — below "
        "that a percentage swing is noise.</span></div>")

    movers = sql("""
        WITH norm AS (
            SELECT theme_id, avg(reviews) AS usual
              FROM theme_weekly
             WHERE model = %s AND week_start < %s AND week_start >= %s::date - 56
             GROUP BY 1)
        SELECT coalesce(t.display_name, t.label) AS name, t.category,
               w.reviews, n.usual,
               round(100 * (w.reviews - n.usual) / nullif(n.usual, 0)) AS pct
          FROM theme_weekly w
          JOIN norm n ON n.theme_id = w.theme_id
          JOIN themes t ON t.model = w.model AND t.theme_id = w.theme_id
         WHERE w.model = %s AND w.week_start = %s AND t.actionable
           AND t.avg_rating <= 2.5 AND n.usual >= 10
         ORDER BY pct DESC NULLS LAST LIMIT 4""",
        (MODEL, start, start, MODEL, start))

    for col, m in zip(st.columns(4), movers.itertuples()):
        with col:
            up = (m.pct or 0) > 0
            colour = design.ACCENT if up else design.NEUTRAL_700
            design.html(
                '<div class="cell"><div style="display:flex;align-items:baseline;'
                f'gap:10px"><span class="mover-n" style="color:{colour}">'
                f'{"+" if up else "−"}{abs(int(m.pct or 0))}%</span>'
                f'<span style="font-size:12px;color:{design.MUTED}">'
                f'{int(m.reviews)} reviews</span></div>'
                f'<div class="mover-t">{clean(m.name)}</div>'
                f'<div class="mover-s">{clean(m.category)} · usually about '
                f'{float(m.usual):.0f} a week</div></div>')
    design.rule()

    # ── why keywords miss this, and what to do with it ──────────────────────
    a, b = st.columns(2, gap="large")
    with a:
        demo = sql("""
            SELECT n_rows FROM themes
             WHERE model = %s
               AND coalesce(display_name, label) = 'Waited an hour or more'""",
            (MODEL,))
        size = int(demo.n_rows.iloc[0]) if not demo.empty else 0
        quotes = sql("""
            SELECT r.content FROM review_themes rt
              JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
              JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
             WHERE rt.model = %s AND t.actionable
               AND coalesce(t.display_name, t.label) = 'Waited an hour or more'
               AND r.word_count BETWEEN 5 AND 14
             ORDER BY rt.strength DESC LIMIT 3""", (MODEL,))
        chips = "".join(f'<div class="quote">“{clean(q)}”</div>'
                        for q in quotes.content)
        design.html(
            '<div style="padding:32px 44px 36px 48px">'
            '<h3 style="font-size:21px;margin-bottom:10px !important">Why a '
            "keyword report would have missed this</h3>"
            '<p class="sub" style="margin-bottom:18px">Customers never use your '
            "words. These three real reviews share almost nothing on the page, "
            "and Echo counts all three against one problem — which is why that "
            f"problem shows {size:,} reviews and not a fraction of it.</p>"
            f"{chips}</div>")
    with b:
        steps = "".join(f'<div class="step"><b>{i:02d}</b><span>{s}</span></div>'
                        for i, s in enumerate(prose.FIVE_MINUTES, 1))
        design.html(
            f'<div style="padding:32px 48px 36px 44px;border-left:1px solid '
            f'{design.DIVIDER}"><h3 style="font-size:21px;'
            'margin-bottom:10px !important">What you can do with it in five '
            f'minutes</h3>{steps}</div>')

    design.rule()
    design.html('<div class="foot">'
                + "".join(f"<span>{f}</span>" for f in prose.SCOPE_FOOTNOTES)
                + "</div>")
