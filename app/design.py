"""Design tokens, page styling and chart defaults for the dashboard.

One place for every colour, font and spacing decision, so pages describe content
and never appearance.

Palette values come from a validated reference palette rather than being chosen
by eye. Slots 1-3 clear colourblind separation and contrast checks in both light
and dark modes. Ratings use the diverging pair (red-grey-blue) because a star
rating has a genuine neutral middle; magnitude uses a single hue light-to-dark;
identity uses the fixed categorical order, never cycled.
"""

import plotly.graph_objects as go
import streamlit as st

# ── ink and surfaces ────────────────────────────────────────────────────────
PLANE = "#f9f9f7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HAIRLINE = "rgba(11,11,11,0.10)"

# ── categorical: identity, assigned in fixed order and never cycled ─────────
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]

# ── diverging: polarity, warm and cool poles with a neutral middle ──────────
NEGATIVE = "#d03b3b"
NEUTRAL = "#f0efec"
POSITIVE = "#2a78d6"
RATING_SCALE = [(0.0, NEGATIVE), (0.5, NEUTRAL), (1.0, POSITIVE)]

# ── sequential: magnitude, one hue light to dark ────────────────────────────
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CSS = f"""
<style>
  .stApp {{ background: {PLANE}; }}
  html, body, [class*="css"] {{ font-family: {FONT}; }}

  /* Streamlit's own chrome adds nothing for a reader. */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 2.6rem 3rem 4rem; max-width: 1400px; }}

  h1 {{ font-size: 1.9rem !important; font-weight: 620 !important;
       letter-spacing: -0.02em; color: {INK}; margin-bottom: .2rem !important; }}
  h2 {{ font-size: 1.18rem !important; font-weight: 600 !important;
       letter-spacing: -0.01em; color: {INK}; margin: 2.2rem 0 .1rem !important; }}
  h3 {{ font-size: .97rem !important; font-weight: 600 !important; color: {INK}; }}

  .lede {{ color: {INK_2}; font-size: 1.02rem; line-height: 1.6;
          max-width: 62ch; margin: .1rem 0 1.6rem; }}
  .sub {{ color: {MUTED}; font-size: .85rem; margin: -.1rem 0 1.1rem; }}

  /* Stat tiles: the number is the chart. */
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
           gap: 14px; margin: .4rem 0 1.6rem; }}
  .tile {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 10px;
          padding: 18px 20px 16px; }}
  .tile .k {{ color: {MUTED}; font-size: .75rem; font-weight: 500;
             text-transform: uppercase; letter-spacing: .06em; }}
  .tile .v {{ color: {INK}; font-size: 2rem; font-weight: 640;
             letter-spacing: -0.03em; line-height: 1.15; margin-top: 6px; }}
  .tile .n {{ color: {INK_2}; font-size: .82rem; margin-top: 5px; line-height: 1.45; }}

  .card {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 10px;
          padding: 20px 22px; margin-bottom: 14px; }}

  .pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px;
          font-size: .74rem; font-weight: 600; letter-spacing: .02em; }}
  .pill-neg {{ background: rgba(208,59,59,.10); color: #a32f2f; }}
  .pill-pos {{ background: rgba(42,120,214,.10); color: #1c5cab; }}

  section[data-testid="stSidebar"] {{ background: {SURFACE};
                                      border-right: 1px solid {HAIRLINE}; }}
  div[data-testid="stDataFrame"] {{ border: 1px solid {HAIRLINE}; border-radius: 10px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {HAIRLINE}; }}
  .stTabs [data-baseweb="tab"] {{ font-size: .92rem; font-weight: 550; }}
  div[data-testid="stExpander"] {{ border: 1px solid {HAIRLINE}; border-radius: 10px;
                                   background: {SURFACE}; }}
</style>
"""


def setup(title: str) -> None:
    """Page config plus the stylesheet. Called once at the top of every page."""
    st.set_page_config(page_title=f"Echo — {title}", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, lede: str = "") -> None:
    st.markdown(f"# {title}", unsafe_allow_html=True)
    if lede:
        st.markdown(f"<div class='lede'>{lede}</div>", unsafe_allow_html=True)


def tiles(items) -> None:
    """items: (label, value, note) — the headline numbers, as tiles not charts."""
    cells = "".join(
        f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div>"
        f"<div class='n'>{n}</div></div>" for k, v, n in items)
    st.markdown(f"<div class='tiles'>{cells}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='sub'>{text}</div>", unsafe_allow_html=True)


def style(fig, height=340, legend=False, ylab="", xlab=""):
    """Recessive chrome: hairline grid, no chart junk, generous breathing room."""
    fig.update_layout(
        height=height, font=dict(family=FONT, size=13, color=INK_2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=8, r=8, t=14, b=8),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.16, x=0, title_text="",
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS,
                        font=dict(family=FONT, size=12, color=INK)),
    )
    axis = dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS,
                tickfont=dict(color=MUTED, size=12),
                title_font=dict(color=MUTED, size=12))
    # Horizontal gridlines only. Vertical ones add noise without helping anyone
    # read a value off the chart.
    fig.update_xaxes(**axis, showgrid=False, title_text=xlab)
    fig.update_yaxes(**axis, showgrid=True, title_text=ylab)
    return fig
