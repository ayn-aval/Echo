"""Dark results-board theme: tokens, page CSS, and the small HTML builders.

Three rules hold this file together, and breaking any of them is what made the
supplied reference render badly:

1. **A block that contains a Streamlit widget must sit in `st.container(key=...)`.**
   Streamlit closes every `<div>` at the end of the `st.markdown` call that opened
   it, so `html('<div class="band">')` … widgets … `html('</div>')` produces an
   *empty* band and leaves the widgets unpadded. A keyed container emits a real
   DOM node carrying `st-key-<key>`, which CSS below can pad and rule.

2. **A block that contains no widget is emitted as a single `html()` call**, with
   its own internal flex or grid — never as `st.columns`. That is what stops the
   gaps Streamlit puts between columns from tearing through a solid field.

3. Every colour here was contrast-checked against the ground before use. The
   alpha values are not free: `.48` is the floor for small text (4.58:1) and `.37`
   the floor for an unlabelled bar (3.43:1 on the ground, 3.01:1 on its track).
"""

from __future__ import annotations

import streamlit as st

MARK = "◼"  # the page icon: a filled square, matching the flat geometry

BG = "#0c0d10"
SURFACE = "#14161a"
TEXT = "#f2f2f0"
ACCENT = "#6ff2c0"

# Alpha tones, all composited over BG. See the docstring for why these numbers.
DIM = "rgba(255,255,255,0.37)"      # comparison bars and unflagged columns
STEP = "rgba(255,255,255,0.55)"     # borders and axis rules
TRACK = "rgba(255,255,255,0.06)"    # the empty part of a bar
LINE = "rgba(255,255,255,0.14)"     # band dividers
HAIR = "rgba(255,255,255,0.08)"     # row rules inside a table
MUTED = "rgba(242,242,240,0.66)"    # secondary prose
FAINT = "rgba(242,242,240,0.58)"    # tertiary prose
LABEL = "rgba(242,242,240,0.48)"    # the small uppercase labels

FONT = '"Archivo", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def _css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;800&display=swap');

:root {{
  --bg:{BG}; --surface:{SURFACE}; --text:{TEXT}; --accent:{ACCENT};
  --dim:{DIM}; --step:{STEP}; --track:{TRACK}; --line:{LINE}; --hair:{HAIR};
  --muted:{MUTED}; --faint:{FAINT}; --label:{LABEL};
}}

html, body, .stApp, [class*="st-"], button, input, select, textarea {{
  font-family: {FONT} !important;
}}
/* Material ligature icons must keep their own font or they render as words. */
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}

.stApp {{ background: var(--bg); color: var(--text); }}
* {{ border-radius: 0 !important; }}
::selection {{ background: rgba(111,242,192,0.30); }}

/* Full-bleed page: the bands supply their own margins, so the frame has none. */
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
[data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
[data-testid="stHorizontalBlock"] {{ gap: 0 !important; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}

h1, h2, h3 {{ font-weight: 800 !important; letter-spacing: -0.015em; color: var(--text);
              margin: 0 !important; }}
p {{ margin: 0; }}
.n {{ font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
.lab {{ font-size: 10.5px; letter-spacing: 0.16em; text-transform: uppercase;
        font-weight: 600; }}

/* ── bands ──────────────────────────────────────────────────────────────────
   Keyed containers. `band_*` is a full-width padded section; `padl_*`/`padr_*`
   are the two halves of a split section, the left one carrying the divider. */
[class*="st-key-band_"] {{ padding: 44px 56px 50px; border-bottom: 1px solid var(--line); }}
[class*="st-key-hero_"] {{ padding: 60px 56px 54px; border-bottom: 1px solid var(--line); }}
[class*="st-key-padl_"] {{ padding: 42px 44px 46px 56px; border-right: 1px solid var(--line); }}
[class*="st-key-padr_"] {{ padding: 42px 56px 46px 44px; }}
[class*="st-key-split_"] {{ border-bottom: 1px solid var(--line); }}

.topbar {{ display:flex; align-items:center; gap:24px; padding:20px 56px;
           border-bottom:1px solid var(--line); }}
.topbar .mark {{ font-weight:800; font-size:19px; letter-spacing:-0.02em; }}
.topbar .sep {{ width:1px; height:20px; background:rgba(255,255,255,.18); }}
.topbar .what {{ font-size:13px; color:var(--faint); }}

.figlab {{ display:flex; align-items:baseline; gap:16px; margin-bottom:6px; }}
.figlab .lab {{ color:var(--label); }}
.figlab h2 {{ font-size:22px; }}
.figsub {{ font-size:13px; color:var(--muted); margin:0 0 28px !important; max-width:74ch; }}

.kpi-n {{ font-size:76px; font-weight:800; line-height:.9;
          font-variant-numeric:tabular-nums; letter-spacing:-0.03em; }}
.chip {{ padding:5px 9px; background:var(--accent); color:var(--bg);
         font-size:12px; font-weight:800; white-space:nowrap; }}
.chip.ghost {{ background:transparent; border:1px solid var(--step); color:var(--text); }}
.kpi-note {{ font-size:13.5px; line-height:1.5; color:var(--muted); margin-top:14px;
             max-width:34ch; }}
.kpi-note b {{ color:var(--text); font-weight:600; }}

.card {{ background:var(--surface); padding:24px; }}
.foot {{ font-size:12px; line-height:1.6; color:var(--faint); }}
.foot b {{ color:var(--muted); font-weight:400; }}

.review {{ padding:15px 0; border-bottom:1px solid var(--hair); }}
.review .m {{ display:flex; gap:12px; align-items:center; font-size:11.5px;
              color:var(--faint); margin-bottom:7px; }}
.tag {{ background:var(--surface); color:var(--muted); font-size:11px; padding:3px 9px; }}

/* ── controls ───────────────────────────────────────────────────────────────
   Streamlit buttons default to full width, which made the reference's topic
   switcher three enormous slabs. These are chips: auto width, hairline border. */
[class*="st-key-chips_"] .stButton > button {{
  font-size:11.5px; font-weight:600; padding:6px 12px; width:auto;
  margin-right:8px; white-space:nowrap;
  border:1px solid var(--step); background:transparent; color:var(--muted);
  text-align:left; justify-content:flex-start; }}
[class*="st-key-chips_"] .stButton > button:hover {{
  border-color:var(--accent); color:var(--text); background:transparent; }}
[class*="st-key-chips_"] .stButton > button[kind="primary"] {{
  background:var(--accent); color:var(--bg); border-color:var(--accent); }}
.stButton > button:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}

/* `gap: 0` on the vertical block leaves no slack, so a control taller than the
   height Streamlit reserved for it overlaps whatever follows. The input below is
   45px against a reserved 38px, which buried the label under the chips by 16px.
   Both rules give the affected containers that slack back explicitly. */
[data-testid="stTextInput"] {{ margin-bottom: 12px; }}
[class*="st-key-chips_"] {{ padding-top: 18px; }}

.stTextInput input {{ background:var(--surface); border:1px solid var(--step);
  color:var(--text); font-size:15px; padding:12px 14px; }}
.stTextInput input:focus {{ border-color:var(--accent); box-shadow:none; }}
.stTextInput input::placeholder {{ color:var(--faint); }}
</style>
"""


def boot() -> None:
    st.markdown(_css(), unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def fig_label(number: str, title: str, sub: str) -> None:
    html(f'<div class="figlab"><span class="lab">{number}</span><h2>{title}</h2></div>'
         f'<p class="figsub">{sub}</p>')


def chip_row(options: list[str], state_key: str,
             widths: list[float] | None = None) -> int:
    """An exclusive row of chips. Returns the selected index.

    `widths` needs one entry per option plus a trailing spacer. Equal columns
    wrap the longest label onto three lines, so callers size the columns to
    their own text.

    The rerun matters: a button reports its click on the same run, by which point
    the chips above have already been drawn from the *old* selection, so without
    it the highlight trails the content by one click.
    """
    st.session_state.setdefault(state_key, 0)
    with st.container(key=f"chips_{state_key}"):
        cols = st.columns(widths or [1] * len(options) + [max(1, 7 - len(options))])
        for i, opt in enumerate(options):
            with cols[i]:
                if st.button(opt, key=f"{state_key}_{i}",
                             type="primary" if st.session_state[state_key] == i else "secondary"):
                    st.session_state[state_key] = i
                    st.rerun()
    return st.session_state[state_key]


def column_chart(values: list[int], flags: list[int], height: int = 200) -> str:
    """Flat column chart. Flagged columns take the accent, the rest the dim tone."""
    peak = max(values) or 1
    cols = ""
    for i, v in enumerate(values):
        on = i in flags
        fill = "var(--accent)" if on else "var(--dim)"
        col = "var(--accent)" if on else "var(--faint)"
        h = max(3, round(100 * v / peak))
        cols += ('<div style="flex:1;display:flex;flex-direction:column;'
                 'justify-content:flex-end;align-items:center;height:100%">'
                 f'<span class="n" style="font-size:9.5px;margin-bottom:5px;color:{col}">{v}</span>'
                 f'<div style="width:100%;height:{h}%;background:{fill}"></div></div>')
    return (f'<div style="display:flex;align-items:flex-end;gap:4px;height:{height}px;'
            f'border-bottom:1px solid var(--step)">{cols}</div>')


def hbar(pct: float, fill: str = "var(--accent)", height: int = 8) -> str:
    return (f'<div style="height:{height}px;background:var(--track)">'
            f'<div style="height:{height}px;width:{pct:.1f}%;background:{fill}"></div></div>')
