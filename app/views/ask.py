"""Ask anything — search by meaning, or browse every problem.

The search box and the problem list sit together because they answer the same
question, "what are people saying about X", for an X that either is or is not
one of the problems the system found on its own.
"""

import html as _html

import design
import shared
import streamlit as st
from shared import MODEL, sql

K = 25              # how many reviews a search returns
EXAMPLES = ["refund never came back to my account",
            "driver was rude to me",
            "the app is unusable on my old phone"]


def clean(text, limit=None):
    out = " ".join(str(text).split())
    if limit and len(out) > limit:
        out = out[:limit].rstrip() + "…"
    return _html.escape(out)


def stars(n):
    n = int(n)
    return f"{n} star" if n == 1 else f"{n} stars"


def render() -> None:
    # Streamlit forbids assigning to a key an instantiated widget owns, so the
    # seed the example buttons write lives under its own key and is passed as
    # `value`. The input itself is given no key: its identity then follows its
    # arguments, so a new seed produces a widget carrying the new text, while
    # anything typed survives reruns where the seed has not changed.
    st.session_state.setdefault("seed", "")

    design.html(
        '<div style="padding:34px 48px 6px">'
        '<h1 style="font-size:38px;max-width:26ch;margin-bottom:10px !important">'
        "Ask it the way a customer would say it</h1>"
        '<p class="sub">You don’t have to guess the words in the review. '
        "“Driver was rude” also finds “rider shouted at me” — nothing in common "
        "except the meaning.</p></div>")

    with st.container(key="searchbox"):
        pad, box, _ = st.columns([0.055, 1.5, 1.2])
        with box:
            typed = st.text_input(
                "Search", value=st.session_state["seed"],
                label_visibility="collapsed",
                placeholder="e.g. refund never came back to my account")
            design.html(f'<div style="font-size:12px;color:{design.FAINT};'
                        'margin:10px 0 2px">Try:</div>')
            for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
                with col:
                    if st.button(example, key=f"ex_{example}"):
                        st.session_state["seed"] = example
                        st.rerun()

    query = (typed or "").strip()
    design.html('<div style="height:24px"></div>')
    design.rule()

    left, right = st.columns([1, 0.5], gap="small")

    with left:
        if not query:
            design.html(
                '<div style="padding:30px 40px 44px 48px"><p class="sub">'
                "Type a complaint above, or pick an example.</p></div>")
        else:
            shared.get_search()
            from src.search.query import search

            hits = search(query, k=K)
            meta = sql("""SELECT review_id, score AS stars,
                                 reviewed_at::date AS day
                            FROM reviews
                           WHERE app = 'swiggy' AND review_id = ANY(%s)""",
                       (list(hits.review_id),))
            topic = sql("""SELECT rt.review_id,
                                  coalesce(t.display_name, t.label) AS topic
                             FROM review_themes rt
                             JOIN themes t ON t.model = rt.model
                                          AND t.theme_id = rt.theme_id
                            WHERE rt.model = %s AND rt.review_id = ANY(%s)""",
                        (MODEL, list(hits.review_id)))
            hits = hits.merge(meta, on="review_id", how="left") \
                       .merge(topic, on="review_id", how="left")
            # The headline must not restate K: len(hits) is always 25 because
            # that is what was asked for. How many of the closest reviews came
            # from unhappy customers is a real answer.
            unhappy = int((hits.stars <= 2).sum())

            cards = "".join(
                '<div class="review"><div class="m">'
                f'<b style="color:{design.INK}">{stars(r.stars)}</b>'
                + (f'<span class="tag">{clean(r.topic)}</span>'
                   if isinstance(r.topic, str) else "")
                + f'<span style="margin-left:auto">{r.day:%d %b %Y}</span></div>'
                f'<div style="font-size:15px;line-height:1.5">'
                f'{clean(r.content)}</div></div>'
                for r in hits.itertuples())

            design.html(
                '<div style="padding:26px 40px 44px 48px">'
                '<div style="display:flex;align-items:baseline;gap:12px;'
                'flex-wrap:wrap;margin-bottom:10px">'
                f'<h2 style="font-size:22px">{unhappy} of the {len(hits)} '
                "closest reviews are unhappy customers</h2>"
                f'<span style="font-size:13px;color:{design.MUTED}">'
                f"averaging {hits.stars.mean():.1f} stars · closest in meaning "
                f"first</span></div>{cards}</div>")

    # ── browse every problem ────────────────────────────────────────────────
    with right:
        # Complaints only. Unfiltered, the nine largest groups are all praise
        # — "Praise: 'good'" alone is 3,402 reviews — so a rail headed
        # "problems" would have listed compliments.
        themes = sql("""
            SELECT theme_id, coalesce(display_name, label) AS name, n_rows
              FROM themes
             WHERE model = %s AND actionable AND avg_rating <= 2.5
             ORDER BY n_rows DESC LIMIT 14""", (MODEL,))
        total = int(sql("SELECT count(*) AS n FROM themes WHERE model = %s "
                        "AND actionable AND avg_rating <= 2.5",
                        (MODEL,)).n.iloc[0])
        history = sql("""
            SELECT theme_id, week_start, reviews FROM theme_weekly
             WHERE model = %s AND theme_id = ANY(%s)
             ORDER BY theme_id, week_start""",
            (MODEL, [int(i) for i in themes.theme_id]))

        rows = "".join(
            f'<div class="topic"><span class="lab">{clean(t.name)}</span>'
            + design.spark(
                history[history.theme_id == t.theme_id].reviews.tolist()[-8:])
            + f'<span class="n">{int(t.n_rows):,}</span></div>'
            for t in themes.itertuples())

        design.html(
            f'<div style="padding:26px 40px 44px 32px;border-left:1px solid '
            f'{design.DIVIDER}">'
            f'<h3 style="font-size:20px;margin-bottom:6px !important">Or browse '
            f"all {total} problems</h3>"
            f'<p style="font-size:12.5px;color:{design.MUTED};margin:0 0 14px">'
            "Grouped automatically — nobody wrote this list. Complaints only; "
            f"106 groups in all. Bars are the last eight weeks.</p>{rows}"
            "</div>")
