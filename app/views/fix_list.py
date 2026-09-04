"""The fix list — every problem ranked, with the reviews behind it.

Eight weeks, not one. A single week of this corpus puts about 47 reviews behind
the biggest problem and shows most problems falling, so a weekly ranking makes a
real problem look like nothing happening. The window is stated on screen.

The reference this is built from also gives every row a suggested owner and a
recommended action, and footnotes both as illustrative. There is no owner or
action data in this database, so neither is shown: a row here carries only
things that came out of the reviews.
"""

import html as _html

import design
import insights
import streamlit as st
from shared import MODEL, WEEKS_ON_FIX_LIST, in_window, sql

AREA_KEY = "area"
SELECTED = "selected_theme"


def clean(text, limit=None):
    out = " ".join(str(text).split())
    if limit and len(out) > limit:
        out = out[:limit].rstrip() + "…"
    return _html.escape(out)


def render() -> None:
    st.session_state.setdefault(AREA_KEY, "All")
    area = st.session_state[AREA_KEY]

    areas = ["All"] + sql(f"""
        SELECT t.category AS area, count(*) AS n
          FROM review_themes rt
          JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
          JOIN themes t ON t.model = rt.model AND t.theme_id = rt.theme_id
         WHERE rt.model = %s AND rt.theme_id >= 0 AND t.actionable
           AND t.category NOT IN ('General praise', 'Other')
           {in_window('r')}
         GROUP BY 1 ORDER BY 2 DESC LIMIT 5""", (MODEL,)).area.tolist()

    # Title and chips on separate rows. Sharing one row gives the chips about
    # 570px, and five nowrap labels at that width collide and clip the last one.
    design.html(
        '<div style="padding:32px 48px 14px">'
        '<h1 style="font-size:38px;margin-bottom:8px !important">The fix '
        "list</h1>"
        f'<p class="sub">Every problem from the last {WEEKS_ON_FIX_LIST} '
        "weeks, ranked by how many people hit it. Click a row for the reviews "
        "behind it.</p></div>")
    with st.container(key="chips"):
        cols = st.columns([1] * len(areas) + [max(1, 8 - len(areas))])
        for col, name in zip(cols, areas):
            with col:
                if st.button(name, key=f"chip_{name}",
                             type="primary" if name == area else "secondary"):
                    st.session_state[AREA_KEY] = name
                    st.rerun()
    design.html('<div style="height:18px"></div>')
    design.rule()

    rows = insights.priorities(
        limit=8, window_weeks=WEEKS_ON_FIX_LIST,
        areas=() if area == "All" else (area,), days=WEEKS_ON_FIX_LIST * 7)
    if rows.empty:
        design.html('<div class="pad"><p class="sub">No problem in this area '
                    'clears the threshold in this window.</p></div>')
        return

    st.session_state.setdefault(SELECTED, int(rows.theme_id.iloc[0]))
    if st.session_state[SELECTED] not in set(rows.theme_id):
        st.session_state[SELECTED] = int(rows.theme_id.iloc[0])
    chosen = st.session_state[SELECTED]

    # Highlight the open row. A keyed container is a real DOM element, so its
    # background can be set by rule; wrapping widgets in a markdown div cannot
    # work — Streamlit closes the div before the widget is emitted.
    st.markdown(f"<style>.st-key-row_{chosen} {{ background:{design.SURFACE}; }}"
                "</style>", unsafe_allow_html=True)

    listcol, evidence = st.columns([1, 0.5], gap="small")

    with listcol:
        biggest = int(rows.reviews.max()) or 1
        history = sql("""
            SELECT theme_id, week_start, reviews FROM theme_weekly
             WHERE model = %s AND theme_id = ANY(%s)
             ORDER BY theme_id, week_start""",
            (MODEL, [int(i) for i in rows.theme_id]))

        for i, r in enumerate(rows.itertuples(), 1):
            tid = int(r.theme_id)
            on = " on" if tid == chosen else ""
            weeks = history[history.theme_id == tid].reviews.tolist()[-8:]
            if r.change_pct != r.change_pct:              # NaN: no prior window
                move, cls = "new", "down"
            else:
                cls = "up" if r.change_pct > 0 else "down"
                move = f"{'+' if r.change_pct > 0 else '−'}{abs(r.change_pct):.0f}%"

            with st.container(key=f"row_{tid}"):
                rank, body, num, trend = st.columns([0.5, 4, 1.5, 1.5])
                with rank:
                    design.html(f'<div class="rank{on}">{i:02d}</div>')
                with body:
                    design.html(
                        f'<div class="rtitle">{clean(r.name)}</div>'
                        f'<div class="rmeta">{clean(r.area)} · averaging '
                        f'{r.avg_rating:.1f} stars</div>')
                    if st.button("Show the reviews behind this",
                                 key=f"sel_{tid}"):
                        st.session_state[SELECTED] = tid
                        st.rerun()
                with num:
                    design.html(
                        f'<div class="rnum">{int(r.reviews):,}</div>'
                        f'<div class="rlab">reviews</div>'
                        f'<div class="track"><div class="fill" style="width:'
                        f'{max(2, round(100 * int(r.reviews) / biggest))}%">'
                        "</div></div>")
                with trend:
                    design.html(
                        f'<div class="{cls}" style="font-size:18px;'
                        f'text-align:right">{move}</div>'
                        '<div class="rlab" style="text-align:right">vs 8 '
                        "weeks before</div>"
                        '<div style="display:flex;justify-content:flex-end;'
                        'margin-top:4px">'
                        + design.spark(weeks, width=6) + "</div>")

        design.html(
            f'<div style="padding:16px 48px;font-size:12px;color:{design.FAINT}">'
            "“vs the eight weeks before” is each problem against its own "
            "previous window.</div>")

    # ── the evidence rail ───────────────────────────────────────────────────
    with evidence:
        row = rows[rows.theme_id == chosen].iloc[0]
        quotes = sql("""
            SELECT r.score AS stars, r.reviewed_at::date AS day, r.content
              FROM review_themes rt
              JOIN reviews r ON r.app = rt.app AND r.review_id = rt.review_id
             WHERE rt.model = %s AND rt.theme_id = %s
               AND r.word_count BETWEEN 12 AND 60
             ORDER BY rt.strength DESC LIMIT 4""", (MODEL, int(chosen)))
        cards = "".join(
            f'<div class="evi"><div class="m">{int(q.stars)} star'
            f'{"" if q.stars == 1 else "s"} · {q.day:%d %b %Y}</div>'
            f'<div class="t">{clean(q.content, 260)}</div></div>'
            for q in quotes.itertuples())
        design.html(
            f'<div style="padding:26px 40px 44px 32px;border-left:1px solid '
            f'{design.DIVIDER}"><div class="kicker">The evidence</div>'
            f'<div style="font-weight:800;font-size:21px;line-height:1.2">'
            f'{clean(row["name"])}</div>'
            f'<div style="font-size:13px;color:{design.MUTED};margin:6px 0 18px">'
            f'{int(row.reviews):,} reviews in the window · averaging '
            f'{row.avg_rating:.1f} stars · {clean(row.area)}</div>{cards}</div>')
