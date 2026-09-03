"""Design system for the Echo app: tokens, chrome, and chart defaults.

One place for every colour, font and spacing decision, so page files describe
content and never appearance.

Palette values are taken unchanged from a validated reference palette rather than
picked by eye. The categorical slots clear colourblind separation and contrast
checks in both modes; ratings use the diverging pair because a star rating has a
genuine neutral middle; magnitude uses a single hue light to dark.
"""

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = str(ASSETS / "logo.svg")
MARK = str(ASSETS / "mark.svg")

PLANE = "#f7f7f5"
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
GRID = "#ececE6"
AXIS = "#d5d4cd"
LINE = "#e6e5df"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]

NEGATIVE = "#d03b3b"
NEUTRAL = "#e9e8e3"
POSITIVE = "#2a78d6"

CRITICAL = "#d03b3b"
WARNING = "#e08a2e"
GOOD = "#0ca30c"

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

CSS = f"""
<style>
  .stApp {{ background: {PLANE}; }}
  html, body, [class*="css"] {{ font-family: {FONT}; -webkit-font-smoothing: antialiased; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.6rem 2.6rem 4rem; max-width: 1460px; }}

  h1 {{ font-size: 1.75rem !important; font-weight: 660 !important;
       letter-spacing: -0.025em; color: {INK}; margin: 0 0 .25rem !important; }}
  h2 {{ font-size: 1.05rem !important; font-weight: 640 !important;
       letter-spacing: -0.01em; color: {INK}; margin: 2rem 0 .5rem !important; }}
  h3 {{ font-size: .9rem !important; font-weight: 620 !important; color: {INK_2};
       margin: 0 0 .5rem !important; }}
  p, li {{ color: {INK_2}; }}

  /* ── app bar ─────────────────────────────────────────────────────────── */
  .appbar {{ display:flex; align-items:center; justify-content:space-between;
            gap:20px; padding:0 0 16px; border-bottom:1px solid {LINE};
            margin-bottom:22px; }}
  .appbar .who {{ color:{MUTED}; font-size:.8rem; }}
  .appbar .who b {{ color:{INK_2}; font-weight:600; }}
  .crumb {{ color:{MUTED}; font-size:.74rem; font-weight:600;
           letter-spacing:.09em; text-transform:uppercase; }}
  .lede {{ color:{INK_2}; font-size:1rem; line-height:1.6; max-width:70ch;
          margin:.35rem 0 0; }}

  /* ── brand ──────────────────────────────────────────────────────────── */
  .brand {{ display:flex; align-items:center; gap:12px; padding:6px 6px 16px;
           margin:0 8px 6px; border-bottom:1px solid {LINE}; }}
  .brand-name {{ font-size:1.6rem; font-weight:720; letter-spacing:-0.035em;
                color:{INK}; line-height:1; }}
  .brand-sub {{ font-size:.73rem; font-weight:600; color:{MUTED};
               letter-spacing:.05em; text-transform:uppercase; margin-top:4px; }}
  section[data-testid="stSidebar"] [data-testid="stLogo"] {{ display:none; }}

  /* ── sidebar ─────────────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {{ background:{SURFACE};
      border-right:1px solid {LINE}; width:270px !important; }}
  section[data-testid="stSidebar"] > div {{ padding-top:.4rem; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ padding-top:.2rem; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {{ gap:2px; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
      border-radius:8px; padding:8px 12px; margin:0 8px;
      font-size:.92rem; font-weight:530; color:{INK_2}; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
      background:{PLANE}; color:{INK}; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
      background:rgba(42,120,214,.09); color:{BLUE}; font-weight:640; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span {{ font-weight:inherit; }}
  .sidenote {{ margin:14px 14px 6px; padding-top:14px; border-top:1px solid {LINE};
              color:{MUTED}; font-size:.74rem; line-height:1.5; }}

  /* ── tiles ───────────────────────────────────────────────────────────── */
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
           gap:12px; margin:.2rem 0 1.4rem; }}
  .tile {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:12px;
          padding:16px 18px 14px; }}
  .tile .k {{ color:{MUTED}; font-size:.72rem; font-weight:600;
             text-transform:uppercase; letter-spacing:.07em; }}
  .tile .v {{ color:{INK}; font-size:1.85rem; font-weight:670;
             letter-spacing:-0.03em; line-height:1.2; margin-top:7px; }}
  .tile .n {{ color:{INK_2}; font-size:.8rem; margin-top:4px; line-height:1.45; }}

  .card {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:12px;
          padding:18px 20px; margin-bottom:12px; }}
  .card h4 {{ margin:0 0 6px; font-size:1rem; font-weight:640; color:{INK}; }}

  /* ── priority rows ───────────────────────────────────────────────────── */
  .issue {{ display:flex; gap:16px; align-items:flex-start; background:{SURFACE};
           border:1px solid {LINE}; border-left:3px solid var(--accent);
           border-radius:10px; padding:16px 18px; margin-bottom:10px; }}
  .issue .rank {{ color:{MUTED}; font-size:1.5rem; font-weight:700;
                 line-height:1; min-width:28px; }}
  .issue .body {{ flex:1; }}
  .issue .name {{ font-size:1.02rem; font-weight:640; color:{INK}; }}
  .issue .meta {{ color:{INK_2}; font-size:.85rem; margin-top:5px; line-height:1.5; }}
  .issue .quote {{ color:{MUTED}; font-size:.83rem; margin-top:9px;
                  padding-left:11px; border-left:2px solid {AXIS};
                  line-height:1.5; font-style:italic; }}

  .badge {{ display:inline-block; padding:3px 9px; border-radius:6px;
           font-size:.7rem; font-weight:700; letter-spacing:.04em;
           text-transform:uppercase; }}
  .b-crit {{ background:rgba(208,59,59,.10); color:#a82f2f; }}
  .b-warn {{ background:rgba(224,138,46,.13); color:#8f5613; }}
  .b-good {{ background:rgba(12,163,12,.10); color:#0a6f0a; }}
  .b-flat {{ background:rgba(11,11,11,.06); color:{INK_2}; }}

  div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:10px; }}
  div[data-testid="stExpander"] {{ border:1px solid {LINE}; border-radius:10px;
                                   background:{SURFACE}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{ font-size:.9rem; font-weight:570; }}
  div[data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {LINE};
                                 border-radius:12px; padding:14px 16px; }}
  .stButton button {{ border-radius:8px; font-weight:570; }}
</style>
"""


BRAND_MARK = """
<svg width="38" height="38" viewBox="0 0 32 32" aria-hidden="true">
  <rect width="32" height="32" rx="9" fill="#2a78d6"/>
  <circle cx="11.5" cy="16" r="2.6" fill="#fff"/>
  <path d="M17 9.6a9 9 0 0 1 0 12.8" stroke="#fff" stroke-width="2.3"
        stroke-linecap="round" fill="none"/>
  <path d="M21.6 6.2a14 14 0 0 1 0 19.6" stroke="#fff" stroke-width="2.3"
        stroke-linecap="round" fill="none" opacity=".62"/>
  <path d="M26.2 3a19 19 0 0 1 0 26" stroke="#fff" stroke-width="2.3"
        stroke-linecap="round" fill="none" opacity=".32"/>
</svg>"""


def boot() -> None:
    """Stylesheet and brand. Called once from the navigation shell."""
    st.markdown(CSS, unsafe_allow_html=True)
    # st.logo places the mark above the nav; the block below is the wordmark the
    # reader actually registers, and is inline SVG plus text so it cannot fail to
    # render the way an external asset can.
    st.logo(LOGO, icon_image=MARK, size="large")
    with st.sidebar:
        st.markdown(
            f"<div class='brand'>{BRAND_MARK}"
            f"<div><div class='brand-name'>Echo</div>"
            f"<div class='brand-sub'>Feedback intelligence</div></div></div>",
            unsafe_allow_html=True)


def appbar(section: str, title: str, lede: str = "", right: str = "") -> None:
    """The bar at the top of every page: where you are, and what this page is."""
    st.markdown(
        f"<div class='appbar'><div>"
        f"<div class='crumb'>{section}</div>"
        f"<h1>{title}</h1>"
        + (f"<div class='lede'>{lede}</div>" if lede else "")
        + f"</div><div class='who'>{right}</div></div>",
        unsafe_allow_html=True)


def tiles(items) -> None:
    """items: (label, value, note). The number is the chart."""
    cells = "".join(
        f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div>"
        f"<div class='n'>{n}</div></div>" for k, v, n in items)
    st.markdown(f"<div class='tiles'>{cells}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div style='color:{MUTED};font-size:.83rem;line-height:1.5;"
                f"margin:-.2rem 0 1rem;'>{text}</div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "flat") -> str:
    return f"<span class='badge b-{kind}'>{text}</span>"


def style(fig, height=320, legend=False, ylab="", xlab=""):
    """Recessive chrome: hairline horizontal grid, no chart junk."""
    fig.update_layout(
        height=height, font=dict(family=FONT, size=12.5, color=INK_2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=6, r=6, t=10, b=6), showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0, title_text="",
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS,
                        font=dict(family=FONT, size=12, color=INK)))
    axis = dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS,
                tickfont=dict(color=MUTED, size=11.5),
                title_font=dict(color=MUTED, size=11.5))
    fig.update_xaxes(**axis, showgrid=False, title_text=xlab)
    fig.update_yaxes(**axis, showgrid=True, title_text=ylab)
    return fig
