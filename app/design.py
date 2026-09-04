"""Design system for the Echo app: tokens, chrome, and chart defaults.

One place for every colour, font and spacing decision, so page files describe
content and never appearance.

The look is editorial rather than dashboard: Archivo at weight 800, square
corners everywhere, a warm off-white ground, one hot red accent, and rules
instead of cards and shadows. `border-radius: 0` is a hard rule of the system.
The shell is full-bleed — the content column is the whole window, and each
section supplies its own padding — so a dark header band and a red decision band
can run edge to edge.

## Two rules that come from what Streamlit will and will not do

**Streamlit cannot wrap its own widgets in arbitrary HTML.** Writing
`<div class="row">`, then a button, then `</div>` does not nest the button: the
div is auto-closed on the first markdown call and comes out empty. The reference
design this is built from does exactly that for its tab row, and its own DOM
reports one `.navrow` element containing zero buttons, so none of the tab styling
ever applied. Anything that must wrap a widget uses `st.container(key=...)`,
which emits a real element carrying `st-key-<key>` — verified reaching its
children with a probe before this file was written.

**Any block with no widget in it is one `st.markdown` call.** Splitting a
purely visual block across `st.columns` puts a gutter through it; that is what
tore a white gap across the reference's red poster. Columns are for widgets.

## The palette, and what was measured

Checked against the data-visualisation colour rules — OKLCH lightness and
chroma, OKLab Delta-E under simulated protanopia and deuteranopia, and WCAG
contrast against both surfaces. Results that changed a decision:

    accent      #ec3013   3.76:1 on the ground        ok
    accent-700  #ae1800   6.41:1                      ok
    neutral-700 #605d5d   5.83:1                      ok
    accent-400  #ff9783   1.88:1                      WARN
    neutral-400 #bab6b6   1.80:1                      FAIL — not used

`accent-400` is legitimate only for bar fills, because `bar()` always prints the
name and the number beside the fill; the colour never carries the value alone.
The reference also uses `#bab6b6` for the fix-list sparklines, where a 5px mark
has no label to fall back on — at 1.80:1 that is unreadable, so `spark()` uses
`neutral-700` instead.

Stars are drawn in two states, not three. A one-to-five scale reads as diverging
— bad, neutral, good — and a diverging scale needs two hues plus a grey
midpoint. This system supplies one hue, so the chart says the only thing that
matters: unhappy in accent, everything else in neutral.
"""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
MARK = str(ASSETS / "mark.svg")

# ── tokens ──────────────────────────────────────────────────────────────────
BG = "#f3f2f2"           # the page ground
SURFACE = "#eae9e9"      # inset blocks: quotes, evidence, bar tracks
INK = "#201e1d"
ACCENT = "#ec3013"
ACCENT_400 = "#ff9783"
ACCENT_600 = "#dd2b0f"
ACCENT_700 = "#ae1800"
NEUTRAL_700 = "#605d5d"
DIVIDER = "rgba(32, 30, 29, 0.40)"
MUTED = "rgba(32, 30, 29, 0.62)"
FAINT = "rgba(32, 30, 29, 0.45)"

FONT = '"Archivo", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
NUM = "font-variant-numeric:tabular-nums;font-feature-settings:'tnum';"

TABS = [("This week", "week"), ("The fix list", "fix"),
        ("Ask anything", "ask"), ("Can I trust it", "trust")]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800&display=swap');

  html, body, .stApp, [class*="st-"], button, input, textarea, select {{
      font-family: {FONT} !important; }}
  .stApp {{ background:{BG}; color:{INK}; }}
  * {{ border-radius: 0 !important; }}
  /* ...except the icon spans. Streamlit draws its icons as Material ligatures,
     so overriding their font renders the literal text "keyboard_double_arrow". */
  [data-testid="stIconMaterial"], span[class*="material"], .material-icons {{
      font-family:"Material Symbols Rounded","Material Icons" !important; }}

  #MainMenu, footer, [data-testid="stToolbarActions"],
  [data-testid="stAppDeployButton"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], [data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
  header[data-testid="stHeader"] {{ background:transparent !important;
      height:0 !important; }}

  /* Full bleed: the ink header and the red decision band run edge to edge, so
     the content column is the window and each section pads itself. */
  .block-container {{ padding:0 !important; max-width:100% !important; }}
  [data-testid="stVerticalBlock"] {{ gap:0 !important; }}
  :focus-visible {{ outline:2px solid {ACCENT}; outline-offset:2px; }}

  h1,h2,h3,h4 {{ font-family:{FONT} !important; font-weight:800 !important;
      letter-spacing:-0.015em; line-height:1.12 !important; color:{INK};
      margin:0 !important; }}
  p, li {{ color:{INK}; }}

  /* ── ink header ──────────────────────────────────────────────────────── */
  .ink {{ background:{INK}; color:{BG}; display:flex; align-items:center;
      gap:32px; padding:15px 48px; }}
  .ink .mark {{ font-weight:800; font-size:20px; letter-spacing:-0.02em;
      flex:none; }}
  .ink .bar {{ width:1px; height:22px; background:rgba(243,242,242,.35);
      flex:none; }}
  .ink .what {{ font-size:13.5px; line-height:1.4; max-width:78ch; }}
  .ink .meta {{ margin-left:auto; text-align:right; font-size:11px;
      letter-spacing:.08em; text-transform:uppercase; opacity:.75; flex:none;
      {NUM} }}

  /* ── tabs ────────────────────────────────────────────────────────────────
     Keyed container, so the CSS actually reaches the buttons. */
  .st-key-tabs {{ border-bottom:2px solid {DIVIDER}; padding:0 40px; }}
  .st-key-tabs .stButton > button {{ background:transparent !important;
      border:0 !important; border-bottom:3px solid transparent !important;
      font-weight:800; font-size:14px; padding:14px 10px; width:100%;
      color:{MUTED} !important; justify-content:flex-start; text-align:left; }}
  .st-key-tabs .stButton > button:hover {{ color:{INK} !important;
      background:transparent !important; }}
  .st-key-tabs .stButton > button[kind="primary"] {{ color:{INK} !important;
      border-bottom:3px solid {ACCENT} !important; }}
  .st-key-tabs .dateline {{ text-align:right; font-size:12px; padding-top:17px;
      color:{MUTED}; {NUM} }}

  /* ── generic widgets ─────────────────────────────────────────────────── */
  .stButton > button {{ font-weight:800; font-size:14px;
      border:1px solid {DIVIDER}; background:transparent; color:{INK};
      padding:9px 15px; text-align:left; justify-content:flex-start; }}
  .stButton > button:hover {{ background:rgba(32,30,29,.07); color:{INK};
      border-color:{DIVIDER}; }}
  .stButton > button[kind="primary"] {{ background:{ACCENT}; color:{BG};
      border-color:{ACCENT}; }}
  .stButton > button[kind="primary"]:hover {{ background:{ACCENT_600};
      color:{BG}; }}
  .st-key-searchbox {{ padding-top:16px; }}
  .st-key-searchbox .stButton {{ margin-top:10px; }}
  .st-key-tech {{ padding:0 48px 44px; }}
  /* The reference leaves its hero buttons hard against the window edge,
     because a markdown <div> cannot wrap them. A keyed container can. */
  .st-key-herobtns {{ padding:0 48px; }}
  .stTextInput input {{ background:{SURFACE}; border:1px solid {DIVIDER};
      font-size:16px; color:{INK}; min-height:46px; padding:11px 14px; }}
  .stTextInput input:focus {{ border-color:{ACCENT}; box-shadow:none; }}
  .stTextInput input::placeholder {{ color:{FAINT}; }}
  [data-testid="stExpander"] {{ border:1px solid {DIVIDER};
      background:transparent; }}
  [data-testid="stExpander"] summary {{ font-weight:800; font-size:14px; }}

  /* ── content blocks ──────────────────────────────────────────────────── */
  .pad {{ padding:36px 48px; }}
  .rule {{ height:2px; background:{DIVIDER}; }}
  .hair {{ height:1px; background:{DIVIDER}; }}
  .kicker {{ font-size:11px; letter-spacing:.12em; text-transform:uppercase;
      color:{ACCENT}; font-weight:800; margin-bottom:16px; }}
  .verdict {{ font-weight:800; font-size:50px; line-height:1.05;
      letter-spacing:-0.02em; max-width:26ch; margin:0 0 20px; }}
  .lede {{ font-size:18px; line-height:1.5; max-width:56ch; margin:0 0 20px; }}
  .sub {{ font-size:15px; line-height:1.55; max-width:64ch; color:{MUTED};
      margin:0; }}
  .big {{ font-weight:800; font-size:54px; line-height:1; {NUM} }}
  .bigl {{ font-size:13px; margin-top:7px; max-width:32ch; color:{MUTED}; }}

  .poster {{ background:{ACCENT}; color:{BG}; padding:34px 48px;
      display:flex; gap:44px; align-items:flex-start; }}
  .poster .k {{ font-size:11px; letter-spacing:.12em; text-transform:uppercase;
      font-weight:800; margin-bottom:14px; opacity:.9; }}
  .poster .h {{ font-weight:800; font-size:31px; line-height:1.15;
      max-width:32ch; margin-bottom:14px; }}
  .poster .b {{ font-size:15px; line-height:1.55; max-width:58ch; }}
  .poster .side {{ flex:none; width:34%; border:2px solid {BG};
      padding:18px 20px; }}
  .poster .side .t {{ font-size:14.5px; line-height:1.55; }}

  .cell {{ padding:22px 28px; border-top:1px solid {DIVIDER};
      border-right:1px solid {DIVIDER}; height:100%; }}
  .mover-n {{ font-weight:800; font-size:29px; line-height:1; {NUM} }}
  .mover-t {{ font-weight:800; font-size:16px; margin-top:10px;
      line-height:1.25; }}
  .mover-s {{ font-size:12.5px; margin-top:5px; color:{MUTED}; }}

  .quote {{ background:{SURFACE}; padding:12px 14px; font-size:14px;
      margin-bottom:9px; }}
  .evi {{ background:{SURFACE}; padding:14px 16px; margin-bottom:12px; }}
  .evi .m {{ font-size:11.5px; font-weight:800; margin-bottom:6px;
      color:{FAINT}; {NUM} }}
  .evi .t {{ font-size:14px; line-height:1.5; }}

  /* Rows are keyed containers, so they can be padded and ruled by class. The
     global gap:0 above is what makes the full-bleed bands sit flush, and it
     also removes the spacing *inside* a row — without this the rank, title and
     button of one row overlap the next. */
  [class*="st-key-row_"] {{ padding:20px 36px 20px 48px;
      border-bottom:1px solid {DIVIDER}; }}
  [class*="st-key-row_"] .stButton {{ margin-top:9px; }}
  [class*="st-key-row_"] .stButton > button {{ border:0; padding:0;
      font-size:12.5px; font-weight:800; color:{ACCENT} !important;
      background:transparent !important; }}
  [class*="st-key-row_"] .stButton > button:hover {{ text-decoration:underline; }}
  .st-key-chips {{ padding:0 48px; }}
  .st-key-chips .stButton > button {{ font-size:12.5px; padding:8px 12px;
      white-space:nowrap; }}

  .rank {{ font-weight:800; font-size:36px; line-height:.9;
      color:rgba(32,30,29,.25); {NUM} }}
  .rank.on {{ color:{ACCENT}; }}
  .rtitle {{ font-weight:800; font-size:19px; line-height:1.2; }}
  .rmeta {{ font-size:13px; margin-top:5px; color:{MUTED}; {NUM} }}
  .rnum {{ font-weight:800; font-size:21px; line-height:1; {NUM} }}
  .rlab {{ font-size:11.5px; color:{FAINT}; margin:6px 0 7px; }}
  .track {{ height:8px; background:{SURFACE}; }}
  .fill {{ height:8px; background:{ACCENT_400}; }}
  .up {{ color:{ACCENT_700}; font-weight:800; {NUM} }}
  .down {{ color:{NEUTRAL_700}; font-weight:800; {NUM} }}

  .review {{ padding:16px 0; border-bottom:1px solid {DIVIDER}; }}
  .review .m {{ display:flex; gap:12px; align-items:center; font-size:12px;
      color:{MUTED}; margin-bottom:6px; {NUM} }}
  .tag {{ background:{SURFACE}; color:{NEUTRAL_700}; font-size:11px;
      padding:3px 10px; }}
  .step {{ display:flex; gap:16px; padding:13px 0;
      border-bottom:1px solid {DIVIDER}; }}
  .step b {{ font-size:13px; color:{ACCENT}; width:34px; flex:none; {NUM} }}
  .step span {{ font-size:14.5px; line-height:1.5; }}
  .foot {{ padding:20px 48px 40px; display:flex; gap:32px; flex-wrap:wrap;
      font-size:12.5px; color:{MUTED}; }}
  .foot b {{ color:{INK}; }}
  .topic {{ display:flex; align-items:flex-end; gap:12px; padding:10px 0;
      border-bottom:1px solid {DIVIDER}; }}
  .topic .lab {{ font-weight:800; font-size:14px; flex:1; }}
  .topic .n {{ font-size:12px; width:52px; text-align:right; color:{MUTED};
      {NUM} }}
  .spark {{ display:flex; align-items:flex-end; gap:2px; height:24px; }}
  .spark i {{ display:block; }}
  .js-plotly-plot .plotly text {{ font-family:{FONT} !important; }}
</style>
"""


def boot() -> None:
    """Inject the stylesheet. Called once from the shell."""
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def ink_header(one_liner: str, meta: str) -> None:
    """The dark band. Its one line is the whole product, stated once, on every
    screen — so a visitor who lands anywhere knows what they are looking at."""
    html(f'<div class="ink"><div class="mark">ECHO</div><div class="bar"></div>'
         f'<div class="what">{one_liner}</div>'
         f'<div class="meta">{meta}</div></div>')


def tabs(active: str, dateline: str) -> str:
    """Top tab row. Returns the active key, which may have just changed.

    Inside `st.container(key="tabs")` so `.st-key-tabs .stButton` can restyle
    the buttons into flush tabs. Wrapping them in a plain markdown div does not
    work — see the module docstring.
    """
    with st.container(key="tabs"):
        cols = st.columns([1.1, 1.1, 1.2, 1.3, 4.5])
        for col, (label, key) in zip(cols, TABS):
            with col:
                if st.button(label, key=f"tab_{key}",
                             type="primary" if key == active else "secondary"):
                    # Rerun rather than fall through. A button reports its click
                    # on the same run, by which point the tab row has already
                    # been drawn from the OLD active key — so the content would
                    # switch while the underline stayed on the previous tab.
                    st.session_state["screen"] = key
                    st.rerun()
        with cols[-1]:
            html(f'<div class="dateline">{dateline}</div>')
    return active


def spark(weeks, colour=NEUTRAL_700, width=5, height=24) -> str:
    """An inline sparkline as bare HTML. Returns markup, does not render.

    Not drawn in the reference's #bab6b6: at 1.80:1 against the ground a 5px
    mark with no label beside it is not visible. neutral-700 clears 5.8:1.
    """
    weeks = list(weeks) or [1]
    top = max(weeks) or 1
    bars = "".join(f'<i style="width:{width}px;'
                   f'height:{max(2, round(height * v / top))}px;'
                   f'background:{colour}"></i>' for v in weeks)
    return f'<span class="spark">{bars}</span>'


def poster(kicker: str, head: str, body: str, side_kicker: str = "",
           side: str = "") -> None:
    """The one loud block on a screen, as a single markdown call.

    Rendering this as two `st.columns` puts the column gutter through the red
    field — a white stripe down the middle of the poster. One call, internal
    flex, no seam.
    """
    aside = (f'<div class="side"><div class="k">{side_kicker}</div>'
             f'<div class="t">{side}</div></div>') if side else ""
    html(f'<div class="poster"><div><div class="k">{kicker}</div>'
         f'<div class="h">{head}</div><div class="b">{body}</div></div>'
         f'{aside}</div>')


def rule() -> None:
    html('<div class="rule"></div>')


# ── charts ──────────────────────────────────────────────────────────────────
def style(fig, height=300, ylab="", xlab="", legend=False):
    """Flat chrome to match the page: square, unboxed, no grid but a baseline."""
    fig.update_layout(
        height=height, font=dict(family="Archivo, sans-serif", size=12.5,
                                 color=NEUTRAL_700),
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=4, r=4, t=8, b=4), showlegend=legend,
        hoverlabel=dict(bgcolor=INK, bordercolor=INK,
                        font=dict(family="Archivo, sans-serif", size=12,
                                  color=BG)))
    axis = dict(showgrid=False, zeroline=False, linecolor=DIVIDER, linewidth=1,
                tickfont=dict(color=MUTED, size=11.5),
                title_font=dict(color=MUTED, size=11.5))
    fig.update_xaxes(**axis, title_text=xlab)
    fig.update_yaxes(**axis, title_text=ylab)
    return fig


def line(x, y, hover, height=240, ylab="", fill=False):
    """A single series. No legend — the heading above it names the series."""
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=ACCENT, width=2),
        fill="tozeroy" if fill else None,
        fillcolor="rgba(236,48,19,0.08)", hovertemplate=hover))
    return style(fig, height=height, ylab=ylab)
