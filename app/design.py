"""Design system for the Echo app: tokens, chrome, and chart defaults.

One place for every colour, font and spacing decision, so page files describe
content and never appearance.

The look is editorial rather than dashboard: Archivo at weight 800, square
corners everywhere, a warm off-white ground, one hot red accent, and 2px rules
instead of cards and shadows. `border-radius: 0` is a hard rule of the system —
a single rounded control is enough to make the whole page look like a default
Streamlit app again.

## The palette, and what was measured

These tokens were checked against the data-visualisation checks — OKLCH
lightness and chroma, OKLab Delta-E under simulated protanopia and
deuteranopia, and WCAG contrast against both surfaces. Results that changed a
decision:

    accent      #ec3013   3.76:1 on the ground        ok
    accent-600  #dd2b0f   4.25:1                      ok
    accent-700  #ae1800   6.41:1                      ok
    neutral-700 #605d5d   5.83:1                      ok
    accent-400  #ff9783   1.88:1                      WARN

accent-400 is the bar fill, and 1.88:1 is well under the 3:1 floor. A contrast
warning is dischargeable only by a mandatory second cue, so it is legal here
precisely because bar() always prints the name and the number above the fill —
the colour is never the only thing carrying the value. Remove that label and the
bar becomes illegible, not merely ugly.

For the same reason accent-400 is *not* used for sparklines: a 5px mark with no
label beside it has nothing to fall back on, so spark() uses the full accent.
Every pair that shares a chart clears colourblind separation comfortably; the
closest, accent against neutral-700, is 9.9 under protanopia against a target
of 8.

Stars are drawn in two states, not three. A one-to-five scale reads as diverging
— bad, neutral, good — and a diverging scale needs two hues plus a grey
midpoint. This system supplies one hue, so rather than invent a second, the
chart says the only thing that matters: unhappy in accent, everything else in
neutral.
"""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
MARK = str(ASSETS / "mark.svg")

# ── tokens ──────────────────────────────────────────────────────────────────
BG = "#f3f2f2"           # the page ground
SURFACE = "#eae9e9"      # inset blocks: quote chips, bar tracks, limits
INK = "#201e1d"
ACCENT = "#ec3013"
ACCENT_400 = "#ff9783"
ACCENT_600 = "#dd2b0f"
ACCENT_700 = "#ae1800"
NEUTRAL_700 = "#605d5d"
DIVIDER = "rgba(32, 30, 29, 0.40)"
MUTED = "rgba(32, 30, 29, 0.60)"
FAINT = "rgba(32, 30, 29, 0.45)"

FONT = '"Archivo", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
NUM = "font-variant-numeric:tabular-nums;font-feature-settings:'tnum';"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800&display=swap');

  html, body, [class*="st-"], .stApp, button, input, textarea, select {{
      font-family: {FONT} !important; }}
  /* ...but not the icon spans. Streamlit draws its icons as Material ligatures,
     so overriding their font makes the control render the literal text
     "keyboard_double_arrow_left" instead of an arrow. */
  [data-testid="stIconMaterial"], span[class*="material"], .material-icons {{
      font-family: "Material Symbols Rounded", "Material Icons" !important; }}
  .stApp {{ background: {BG}; color: {INK}; }}
  * {{ border-radius: 0 !important; }}

  /* Hide the toolbar, NOT the header.
     `header {{ visibility: hidden }}` used to be here and it broke the app: when
     the sidebar is collapsed Streamlit renders the reopen control as
     [data-testid="stExpandSidebarButton"] *inside* <header data-testid="stHeader">,
     so hiding the header left no way at all to bring the menu back. The button
     is nested inside [data-testid="stToolbar"] too, so hiding the toolbar
     reproduces the same trap one level down. Hide only the toolbar's action
     group and leave both the header and the toolbar in place. */
  #MainMenu, footer, [data-testid="stToolbarActions"],
  [data-testid="stAppDeployButton"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], [data-testid="stHeaderLogo"]
      {{ display: none !important; }}
  header[data-testid="stHeader"] {{ background: transparent !important; }}
  [data-testid="stExpandSidebarButton"] {{ visibility: visible !important;
      background:{BG}; border:1px solid {DIVIDER}; color:{INK}; }}
  [data-testid="stExpandSidebarButton"]:hover {{ background:{SURFACE}; }}
  [data-testid="stSidebarCollapseButton"] {{ opacity:1 !important;
      visibility:visible !important; }}

  /* The reference this is built from clipped the kicker on every screen: its
     top padding was smaller than the first line's box. 3.2rem clears it. */
  .block-container {{ padding: 3.2rem 1rem 6rem; max-width: 1180px; }}
  :focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}

  h1, h2, h3, h4 {{ font-family: {FONT} !important; font-weight: 800 !important;
      letter-spacing: -0.015em; line-height: 1.12 !important; color: {INK};
      margin-top: 0 !important; }}
  h1 {{ font-size: 44px !important; margin-bottom: .55rem !important; }}
  h2 {{ font-size: 28px !important; margin: 2.2rem 0 .7rem !important; }}
  h3 {{ font-size: 21px !important; margin: 0 0 .55rem !important; }}
  p, li, .stMarkdown {{ color: {INK}; }}
  li {{ margin-bottom: .3rem; }}

  /* ── sidebar ─────────────────────────────────────────────────────────────
     The reference sets [data-testid="stSidebarNav"] to display:none, which
     makes every screen unreachable — verified in its own running DOM. The nav
     stays; it is only restyled. */
  [data-testid="stSidebar"] {{ background:{BG};
      border-right: 2px solid {DIVIDER}; width: 268px !important; }}
  [data-testid="stSidebar"] > div {{ padding-top: .3rem; }}
  [data-testid="stSidebarNav"] ul {{ gap: 0; }}
  [data-testid="stSidebarNav"] a {{ padding: 9px 14px; margin: 0 8px;
      font-size: 15px; font-weight: 600; color: {MUTED}; position: relative; }}
  [data-testid="stSidebarNav"] a:hover {{ background: rgba(32,30,29,.06);
      color: {INK}; }}
  [data-testid="stSidebarNav"] a[aria-current="page"] {{ color:{INK};
      font-weight: 800; background: transparent; }}
  [data-testid="stSidebarNav"] a[aria-current="page"]::before {{ content:'';
      position:absolute; left:-8px; top:6px; bottom:6px; width:3px;
      background:{ACCENT}; }}
  [data-testid="stSidebarNav"] span {{ font-weight: inherit; }}
  /* The wordmark is drawn into the sidebar header rather than written with
     st.sidebar, because Streamlit fixes the sidebar's child order — header,
     nav, user content — so markdown always lands *below* the menu, and this
     design puts the brand above it. Presentational only: the app's actual name
     is in the page title and the headings. */
  /* The header is a flex row by default, which puts the subtitle beside the
     wordmark instead of under it. Block layout stacks them; the collapse
     control is pinned to the corner so it keeps its place. */
  [data-testid="stSidebarHeader"] {{ display:block !important;
      position:relative; height:auto !important;
      padding:14px 14px 0 !important; }}
  [data-testid="stSidebarCollapseButton"] {{ position:absolute; top:10px;
      right:8px; }}
  /* Two pseudo-elements, because the wordmark and its line of explanation are
     set at different sizes and a single `content` cannot carry both. */
  [data-testid="stSidebarHeader"]::before {{ content:"Echo"; display:block;
      font-weight:800; font-size:22px; letter-spacing:-0.02em; color:{INK}; }}
  [data-testid="stSidebarHeader"]::after {{
      content:"What Swiggy customers\A are complaining about";
      white-space:pre; display:block; font-size:12px; line-height:1.4;
      color:{MUTED}; margin-top:4px; padding-bottom:12px;
      border-bottom:2px solid {DIVIDER}; }}
  .side-rule {{ height:2px; background:{DIVIDER}; margin:12px 14px; }}
  .side-note {{ font-size:11px; line-height:1.55; color:{MUTED};
      padding:0 14px; {NUM} }}

  /* ── blocks ──────────────────────────────────────────────────────────── */
  .kicker {{ font-size:11px; letter-spacing:.1em; text-transform:uppercase;
      color:{ACCENT}; font-weight:800; margin-bottom:10px; }}
  .lede {{ font-size:19px; line-height:1.5; max-width:62ch; }}
  .sub {{ font-size:15px; line-height:1.55; max-width:64ch; color:{MUTED}; }}
  .stat-n {{ font-weight:800; font-size:34px; line-height:1; {NUM} }}
  .stat-l {{ font-size:12px; color:{MUTED}; margin-top:5px; }}
  .quote-chip {{ background:{SURFACE}; padding:12px 14px; font-size:14px;
      margin-bottom:10px; }}
  .problem-box {{ border:2px solid {ACCENT}; padding:18px 20px; }}
  .poster {{ background:{ACCENT}; color:{BG}; padding:24px 28px;
      font-weight:800; font-size:22px; line-height:1.25; }}
  .poster small {{ display:block; font-size:12px; letter-spacing:.1em;
      text-transform:uppercase; margin-bottom:8px; opacity:.85;
      font-weight:800; }}
  .limit {{ background:{SURFACE}; padding:14px 16px; margin-bottom:12px; }}
  .limit b {{ font-size:15px; }}

  /* Bars carry their value as text above the fill, always. The fill colour
     alone is 1.88:1 against the ground and cannot be read as a quantity. */
  .bar-head {{ display:flex; justify-content:space-between; font-size:13px;
      margin-bottom:4px; {NUM} }}
  .bar-head b {{ font-weight:800; }}
  .bar-head span {{ color:{MUTED}; }}
  .bar-track {{ height:14px; background:{SURFACE}; }}
  .bar-fill {{ height:14px; background:{ACCENT_400}; }}
  .bar-fill.on {{ background:{ACCENT}; }}

  .review {{ padding:16px 0; border-bottom:1px solid {DIVIDER}; }}
  .review-meta {{ display:flex; gap:12px; align-items:center; font-size:12px;
      color:{MUTED}; margin-bottom:6px; {NUM} }}
  .tag {{ background:{SURFACE}; color:{NEUTRAL_700}; font-size:11px;
      padding:3px 10px; }}
  .up {{ color:{ACCENT_700}; font-weight:600; {NUM} }}
  .down {{ color:{NEUTRAL_700}; font-weight:600; {NUM} }}

  .ranked {{ width:100%; border-collapse:collapse; font-size:14px; }}
  .ranked th {{ text-align:left; font-size:11px; letter-spacing:.08em;
      text-transform:uppercase; color:{MUTED}; padding:8px;
      border-bottom:2px solid {DIVIDER}; font-weight:800; }}
  .ranked td {{ padding:10px 8px; border-bottom:1px solid {DIVIDER};
      vertical-align:top; }}
  .ranked .n {{ font-weight:800; color:{FAINT}; width:30px; {NUM} }}
  .ranked .name {{ font-weight:800; font-size:15px; }}
  .ranked .said {{ font-size:12.5px; color:{MUTED}; margin-top:3px; }}
  .ranked .num {{ font-weight:800; white-space:nowrap; {NUM} }}

  .topic-row {{ display:flex; align-items:flex-end; gap:12px; padding:9px 0;
      border-bottom:1px solid {DIVIDER}; }}
  .topic-row .lab {{ font-weight:800; font-size:14px; flex:1; }}
  .topic-row .n {{ font-size:12px; color:{MUTED}; width:52px;
      text-align:right; {NUM} }}
  .spark {{ display:flex; align-items:flex-end; gap:2px; height:22px; }}
  .spark i {{ display:block; width:5px; background:{ACCENT}; }}

  /* ── widgets ─────────────────────────────────────────────────────────── */
  hr, [data-testid="stDivider"] hr {{ border:0; height:2px;
      background:{DIVIDER}; }}
  .stButton > button, .stDownloadButton > button {{ font-weight:800;
      font-size:14px; border:1px solid {DIVIDER}; background:transparent;
      color:{INK}; padding:8px 14px; text-align:left;
      justify-content:flex-start; width:100%; }}
  .stButton > button:hover {{ background:rgba(32,30,29,.07);
      border-color:{DIVIDER}; color:{INK}; }}
  .stButton > button[kind="primary"] {{ background:{ACCENT}; color:{BG};
      border-color:{ACCENT}; }}
  .stButton > button[kind="primary"]:hover {{ background:{ACCENT_600};
      color:{BG}; }}
  .stTextInput input {{ background:{SURFACE}; border:1px solid {DIVIDER};
      font-size:16px; color:{INK}; padding:11px 13px; }}
  .stTextInput input:focus {{ border-color:{ACCENT}; box-shadow:none; }}
  .stTextInput input::placeholder {{ color:{FAINT}; }}
  div[data-testid="stExpander"] {{ border:1px solid {DIVIDER};
      background:transparent; }}
  div[data-testid="stExpander"] summary {{ font-size:14px; font-weight:800; }}
  div[data-testid="stDataFrame"] {{ border:1px solid {DIVIDER}; }}
  [data-testid="stTable"] th, .stDataFrame th {{ text-transform:uppercase;
      font-size:11px; letter-spacing:.08em; }}
  a[data-testid="stPageLink-NavLink"] {{ font-weight:800; font-size:14px;
      color:{ACCENT} !important; }}
  a[data-testid="stPageLink-NavLink"] span {{ color:{ACCENT} !important; }}
  .js-plotly-plot .plotly text {{ font-family:{FONT} !important; }}
</style>
"""


def boot() -> None:
    """Inject the stylesheet. Called once from the navigation shell."""
    st.markdown(CSS, unsafe_allow_html=True)


# ── text blocks ─────────────────────────────────────────────────────────────
def kicker(text: str) -> None:
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


def lede(text: str) -> None:
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def sub(text: str) -> None:
    st.markdown(f'<p class="sub">{text}</p>', unsafe_allow_html=True)


def rule() -> None:
    st.markdown("<hr>", unsafe_allow_html=True)


def stat(number: str, label: str) -> None:
    st.markdown(f'<div class="stat-n">{number}</div>'
                f'<div class="stat-l">{label}</div>', unsafe_allow_html=True)


def poster(label: str, text: str) -> None:
    """The one loud block on a screen. Its text is a sentence, not a metric."""
    st.markdown(f'<div class="poster"><small>{label}</small>{text}</div>',
                unsafe_allow_html=True)


def quote_chip(text: str) -> None:
    st.markdown(f'<div class="quote-chip">“{text}”</div>',
                unsafe_allow_html=True)


def limit(title: str, body: str) -> None:
    st.markdown(f'<div class="limit"><b>{title}</b>'
                f'<div style="font-size:13.5px;margin-top:4px">{body}</div>'
                f'</div>', unsafe_allow_html=True)


def bar(name: str, value: str, pct: float, selected: bool = False) -> None:
    """A magnitude bar that always prints its own name and number.

    The label is not decoration. The fill is #ff9783, which sits at 1.88:1
    against the ground — under the 3:1 floor — and is only legitimate because
    the quantity is written out beside it. Never call this without a real value.
    """
    on = " on" if selected else ""
    st.markdown(f'<div class="bar-head"><b>{name}</b><span>{value}</span></div>'
                f'<div class="bar-track"><div class="bar-fill{on}" '
                f'style="width:{max(pct, 1.5):.1f}%"></div></div>',
                unsafe_allow_html=True)


def spark(values, height: int = 22) -> str:
    """An inline sparkline as bare HTML. Returns markup, does not render.

    Drawn in the full accent rather than the lighter step the bars use: a 5px
    mark carries no label of its own, so it has nothing to fall back on if the
    colour is too faint to see.
    """
    top = max(values) if len(values) and max(values) else 1
    bars = "".join(f'<i style="height:{max(2, round(height * v / top))}px"></i>'
                   for v in values)
    return f'<div class="spark">{bars}</div>'


def topic_row(name: str, values, count: str) -> None:
    st.markdown(f'<div class="topic-row"><span class="lab">{name}</span>'
                f'{spark(values)}<span class="n">{count}</span></div>',
                unsafe_allow_html=True)


# ── charts ──────────────────────────────────────────────────────────────────
def style(fig, height=300, ylab="", xlab="", legend=False):
    """Flat chrome to match the page: square, unboxed, no grid but a baseline."""
    fig.update_layout(
        height=height, font=dict(family="Archivo, sans-serif", size=12.5,
                                 color=NEUTRAL_700),
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=4, r=4, t=8, b=4), showlegend=legend,
        legend=dict(orientation="h", y=-0.2, x=0, title_text=""),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK,
                        font=dict(family="Archivo, sans-serif", size=12,
                                  color=BG)))
    axis = dict(showgrid=False, zeroline=False, linecolor=DIVIDER, linewidth=1,
                tickfont=dict(color=MUTED, size=11.5),
                title_font=dict(color=MUTED, size=11.5))
    fig.update_xaxes(**axis, title_text=xlab)
    fig.update_yaxes(**axis, title_text=ylab)
    return fig


def line(x, y, hover, height=260, ylab="", fill=False):
    """A single series. No legend — the heading above it names the series."""
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=ACCENT, width=2),
        fill="tozeroy" if fill else None,
        fillcolor="rgba(236,48,19,0.08)", hovertemplate=hover))
    return style(fig, height=height, ylab=ylab)
