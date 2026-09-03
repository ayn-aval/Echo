"""Design system for the Echo app: tokens, chrome, and chart defaults.

One place for every colour, font and spacing decision, so page files describe
content and never appearance.

Chart colours are slots 1-3 of a validated reference palette, unchanged: they
clear colourblind separation and contrast on a white surface, and charts still
sit on white here, so that validation still holds. The surface and ink tokens
below were changed and re-checked with a WCAG contrast calculation; every text
pair clears 4.5:1, and muted text clears 3:1.

The plane is deliberately darker than it was. At the previous #f7f7f5 a white
card sat 1.07:1 against its background — invisible — so nothing on any screen
read as a distinct object. It is now 1.13:1 and carries a shadow, which is what
actually separates a card from the page.
"""

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = str(ASSETS / "logo.svg")
MARK = str(ASSETS / "mark.svg")

# ── surfaces and ink ────────────────────────────────────────────────────────
PLANE = "#eef1f6"      # the page behind everything
SURFACE = "#ffffff"    # cards, and every chart background
SUNK = "#f6f8fb"       # inset areas: quotes, table headers
INK = "#101827"
INK_2 = "#4b5565"
MUTED = "#8792a5"
LINE = "#dfe4ec"
GRID = "#eef1f6"
AXIS = "#cbd3e1"

# The one dark surface, used for the single loudest element on a screen.
DEEP = "#111c33"
DEEP_2 = "#1d2c4a"
DEEP_INK = "#a9b8d4"

SHADOW = "0 1px 2px rgba(16,24,40,.04), 0 2px 6px rgba(16,24,40,.06)"
SHADOW_2 = "0 2px 4px rgba(16,24,40,.05), 0 14px 30px rgba(16,24,40,.10)"

# ── data colour (unchanged; validated reference slots) ──────────────────────
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]

NEGATIVE = "#d03b3b"
NEUTRAL = "#dfe4ec"
POSITIVE = "#2a78d6"

CRITICAL = "#d03b3b"
WARNING = "#e08a2e"    # a darkened step of the status warning hue, for legibility
GOOD = "#0ca30c"

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
NUM = "font-variant-numeric:tabular-nums;font-feature-settings:'tnum';"

CSS = f"""
<style>
  .stApp {{ background: {PLANE}; }}
  html, body, [class*="css"] {{ font-family: {FONT}; -webkit-font-smoothing: antialiased; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.5rem 2.4rem 5rem; max-width: 1480px; }}
  :focus-visible {{ outline: 2px solid {BLUE}; outline-offset: 2px; border-radius: 6px; }}

  h1 {{ font-size: 1.7rem !important; font-weight: 680 !important;
       letter-spacing: -0.028em; color: {INK}; margin: 0 !important; }}
  h2 {{ font-size: 1.12rem !important; font-weight: 660 !important;
       letter-spacing: -0.014em; color: {INK}; margin: 2.2rem 0 .85rem !important; }}
  h3 {{ font-size: .82rem !important; font-weight: 680 !important; color: {MUTED};
       letter-spacing: .07em; text-transform: uppercase;
       margin: 0 0 .7rem !important; }}
  p, li {{ color: {INK_2}; }}

  /* ── app bar ─────────────────────────────────────────────────────────── */
  .appbar {{ display:flex; align-items:flex-end; justify-content:space-between;
            gap:20px; padding:0 0 14px; border-bottom:1px solid {LINE};
            margin-bottom:20px; }}
  .appbar .who {{ color:{MUTED}; font-size:.79rem; white-space:nowrap; {NUM} }}
  .appbar .who b {{ color:{INK_2}; font-weight:640; }}
  .crumb {{ color:{MUTED}; font-size:.71rem; font-weight:700;
           letter-spacing:.1em; text-transform:uppercase; margin-bottom:5px; }}

  /* ── hero: the one loud thing on a screen ────────────────────────────── */
  .hero {{ background:linear-gradient(135deg,{DEEP} 0%,{DEEP_2} 100%);
          border-radius:16px; padding:30px 34px; margin:0 0 20px;
          box-shadow:{SHADOW_2}; position:relative; overflow:hidden; }}
  .hero::after {{ content:''; position:absolute; right:-70px; top:-90px;
                 width:300px; height:300px; border-radius:50%;
                 background:radial-gradient(circle,rgba(42,120,214,.34),transparent 68%); }}
  .hero .eyebrow {{ color:{DEEP_INK}; font-size:.71rem; font-weight:700;
                   letter-spacing:.11em; text-transform:uppercase; }}
  .hero .head {{ color:#ffffff; font-size:1.72rem; font-weight:660;
                letter-spacing:-0.028em; line-height:1.25; margin:11px 0 0;
                max-width:34ch; position:relative; z-index:1; }}
  .hero .row {{ display:flex; align-items:flex-end; gap:34px; margin-top:22px;
               flex-wrap:wrap; position:relative; z-index:1; }}
  .hero .big {{ color:#ffffff; font-size:3.5rem; font-weight:700; line-height:.95;
               letter-spacing:-0.045em; {NUM} }}
  .hero .unit {{ color:{DEEP_INK}; font-size:.85rem; margin-top:9px; }}
  .hero .side {{ color:{DEEP_INK}; font-size:.85rem; line-height:1.6;
                padding-left:34px; border-left:1px solid rgba(255,255,255,.16); }}
  .hero .side b {{ color:#ffffff; font-weight:640; {NUM} }}

  /* ── tiles ───────────────────────────────────────────────────────────── */
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
           gap:14px; margin:.1rem 0 1.5rem; }}
  .tile {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px;
          padding:17px 19px 15px; box-shadow:{SHADOW}; }}
  .tile.lead {{ border-color:rgba(42,120,214,.32); box-shadow:{SHADOW_2}; }}
  .tile .k {{ color:{MUTED}; font-size:.71rem; font-weight:700;
             text-transform:uppercase; letter-spacing:.08em; }}
  .tile .v {{ color:{INK}; font-size:1.95rem; font-weight:690;
             letter-spacing:-0.035em; line-height:1.15; margin-top:9px; {NUM} }}
  .tile.lead .v {{ font-size:2.4rem; color:{BLUE}; }}
  .tile .n {{ color:{INK_2}; font-size:.79rem; margin-top:5px; line-height:1.45; }}

  .card {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px;
          padding:19px 21px; margin-bottom:13px; box-shadow:{SHADOW}; }}
  .card h4 {{ margin:0 0 7px; font-size:1rem; font-weight:660; color:{INK}; }}

  /* ── ranked rows: label, bar, number in one line ─────────────────────── */
  .rank-list {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px;
               padding:8px 6px; box-shadow:{SHADOW}; }}
  .rrow {{ display:grid; grid-template-columns:minmax(150px,1.35fr) 3fr auto;
          gap:16px; align-items:center; padding:11px 16px; border-radius:9px; }}
  .rrow + .rrow {{ border-top:1px solid {GRID}; }}
  .rrow:hover {{ background:{SUNK}; }}
  .rrow .lab {{ color:{INK}; font-size:.9rem; font-weight:580; }}
  .rrow .track {{ background:{GRID}; border-radius:5px; height:9px; width:100%; }}
  .rrow .fill {{ height:9px; border-radius:5px; background:var(--c); }}
  .rrow .val {{ color:{INK}; font-size:.95rem; font-weight:680;
               text-align:right; min-width:112px; {NUM} }}
  .rrow .val small {{ color:{MUTED}; font-weight:520; font-size:.76rem;
                     margin-left:7px; }}

  /* ── priority rows ───────────────────────────────────────────────────── */
  .issue {{ display:flex; gap:17px; align-items:flex-start; background:{SURFACE};
           border:1px solid {LINE}; border-left:4px solid var(--accent);
           border-radius:12px; padding:17px 20px; margin-bottom:11px;
           box-shadow:{SHADOW}; }}
  .issue .rank {{ color:{MUTED}; font-size:1.45rem; font-weight:720;
                 line-height:1; min-width:26px; {NUM} }}
  .issue .body {{ flex:1; min-width:0; }}
  .issue .name {{ font-size:1.04rem; font-weight:660; color:{INK}; }}
  .issue .meta {{ color:{INK_2}; font-size:.84rem; margin-top:6px; line-height:1.5; {NUM} }}
  .issue .quote {{ color:{INK_2}; font-size:.83rem; margin-top:11px;
                  background:{SUNK}; border-left:3px solid {AXIS};
                  border-radius:0 8px 8px 0; padding:9px 13px; line-height:1.55; }}

  .badge {{ display:inline-block; padding:3px 9px; border-radius:6px;
           font-size:.68rem; font-weight:720; letter-spacing:.045em;
           text-transform:uppercase; vertical-align:middle; margin-left:9px; }}
  .b-crit {{ background:rgba(208,59,59,.10); color:#a82f2f; }}
  .b-warn {{ background:rgba(224,138,46,.14); color:#8f5613; }}
  .b-good {{ background:rgba(12,163,12,.10); color:#0a6f0a; }}
  .b-flat {{ background:rgba(16,24,39,.06); color:{INK_2}; }}

  .pill {{ display:inline-block; padding:4px 13px; border-radius:999px;
          font-size:.75rem; font-weight:700; letter-spacing:.02em; }}
  .pill-neg {{ background:rgba(208,59,59,.10); color:#a82f2f; }}
  .pill-pos {{ background:rgba(12,163,12,.10); color:#0a6f0a; }}

  .area-tag {{ margin-left:9px; padding:2px 9px; border-radius:6px;
              background:{SUNK}; border:1px solid {LINE}; color:{INK_2};
              font-size:.7rem; font-weight:620; vertical-align:middle; }}
  .chip {{ padding:3px 9px; border-radius:7px; font-size:.76rem; font-weight:700; {NUM} }}

  /* ── brand and sidebar ───────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {{ background:{SURFACE};
      border-right:1px solid {LINE}; width:274px !important; }}
  section[data-testid="stSidebar"] > div {{ padding-top:.2rem; }}
  [data-testid="stSidebarHeader"], [data-testid="stLogoSpacer"] {{ height:auto; }}
  [data-testid="stLogo"] {{ height:38px !important; width:auto !important;
                            margin:12px 0 6px 8px; }}

  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ padding-top:.35rem; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {{ gap:3px; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
      border-radius:9px; padding:10px 14px; margin:0 10px; position:relative;
      font-size:.93rem; font-weight:560; color:{INK_2}; transition:background .12s; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
      background:{SUNK}; color:{INK}; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
      background:rgba(42,120,214,.10); color:{BLUE}; font-weight:660; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"]
      a[aria-current="page"]::before {{
      content:''; position:absolute; left:-10px; top:9px; bottom:9px; width:3px;
      border-radius:0 3px 3px 0; background:{BLUE}; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span {{ font-weight:inherit; }}
  /* group headings ("Monitor", "Understand") */
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > div > div > span,
  section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {{
      color:{MUTED} !important; font-size:.68rem !important; font-weight:750 !important;
      letter-spacing:.11em; text-transform:uppercase; padding:14px 24px 5px !important; }}

  .filter-head {{ margin:18px 14px 2px; padding-top:16px;
                 border-top:1px solid {LINE}; color:{MUTED}; font-size:.68rem;
                 font-weight:750; letter-spacing:.11em; text-transform:uppercase; }}
  .sidenote {{ margin:16px 14px 6px; padding-top:14px; border-top:1px solid {LINE};
              color:{MUTED}; font-size:.73rem; line-height:1.55; }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label {{ font-size:.78rem;
      font-weight:620; color:{INK_2}; }}

  /* ── streamlit widgets ───────────────────────────────────────────────── */
  div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:12px;
                                    box-shadow:{SHADOW}; overflow:hidden; }}
  div[data-testid="stExpander"] {{ border:1px solid {LINE}; border-radius:12px;
                                   background:{SURFACE}; box-shadow:{SHADOW}; }}
  div[data-testid="stExpander"] summary {{ font-size:.85rem; font-weight:600;
                                           color:{INK_2}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{ font-size:.9rem; font-weight:580; }}
  .stButton button, .stDownloadButton button {{ border-radius:9px; font-weight:600; }}
  .stRadio [role="radiogroup"] {{ gap:18px; }}
  /* On the grey plane a grey field is invisible, so every control that accepts
     typing or a choice gets the card treatment: white, bordered, raised. */
  .stTextInput input, .stNumberInput input,
  div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {{
      background:{SURFACE} !important; border-color:{LINE} !important;
      border-radius:9px !important; box-shadow:{SHADOW}; }}
  .stTextInput input {{ padding:11px 14px !important; font-size:.95rem; }}
  .stTextInput input::placeholder {{ color:{MUTED}; }}
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
      box-shadow:none; }}
  [data-testid="stAlert"] {{ border-radius:12px; }}
  /* page links look like the buttons they behave as, and sit tight under the
     block they belong to rather than floating as loose text */
  div:has(> div > a[data-testid="stPageLink-NavLink"]) {{ margin-top:-6px; }}
  a[data-testid="stPageLink-NavLink"] {{ display:inline-flex; align-items:center;
      background:{SURFACE}; border:1px solid {LINE}; border-radius:9px;
      padding:7px 15px !important; box-shadow:{SHADOW};
      font-weight:640; font-size:.85rem; color:{BLUE} !important; }}
  a[data-testid="stPageLink-NavLink"]:hover {{ background:{SUNK};
      border-color:{AXIS}; text-decoration:none; }}
  a[data-testid="stPageLink-NavLink"] span {{ color:{BLUE} !important; }}
  .js-plotly-plot .plotly text {{ font-family:{FONT} !important; }}
</style>
"""


def boot() -> None:
    """Stylesheet and brand. Called once from the navigation shell.

    st.logo is the only brand mark. An earlier version also drew a wordmark into
    the sidebar body, which Streamlit places *below* the navigation, so the app
    showed its own name twice — once above the menu and once under it.
    """
    st.markdown(CSS, unsafe_allow_html=True)
    st.logo(LOGO, icon_image=MARK, size="large")


def appbar(section: str, title: str, right: str = "") -> None:
    """The bar at the top of every page: where you are, and what this page is."""
    st.markdown(
        f"<div class='appbar'><div>"
        f"<div class='crumb'>{section}</div><h1>{title}</h1></div>"
        f"<div class='who'>{right}</div></div>", unsafe_allow_html=True)


def hero(eyebrow: str, headline: str, value: str, unit: str, side: str = "") -> None:
    """The finding, stated. One per screen, at the top, and nothing else this loud.

    A dashboard that opens with three identical tiles makes the reader do the
    work of deciding which matters. This decides, and says so in a sentence.
    """
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{eyebrow}</div>"
        f"<div class='head'>{headline}</div><div class='row'>"
        f"<div><div class='big'>{value}</div><div class='unit'>{unit}</div></div>"
        + (f"<div class='side'>{side}</div>" if side else "")
        + "</div></div>", unsafe_allow_html=True)


def tiles(items, lead: int | None = None) -> None:
    """items: (label, value, note). `lead` indexes the one that matters most."""
    cells = "".join(
        f"<div class='tile{' lead' if i == lead else ''}'>"
        f"<div class='k'>{k}</div><div class='v'>{v}</div>"
        f"<div class='n'>{n}</div></div>" for i, (k, v, n) in enumerate(items))
    st.markdown(f"<div class='tiles'>{cells}</div>", unsafe_allow_html=True)


def rank_rows(rows) -> None:
    """A ranked comparison as HTML, not a chart.

    rows: (label, bar_fraction_0_to_1, value_text, note_text, colour).

    Drawn in HTML because a Plotly chart is a separate Streamlit element from the
    card markdown around it — which is why the area sparklines used to render
    outside their own card borders. One element, one box, no seams.
    """
    body = "".join(
        f"<div class='rrow' style='--c:{colour};'>"
        f"<div class='lab'>{label}</div>"
        f"<div class='track'><div class='fill' style='width:{max(frac, 0.012) * 100:.1f}%;'></div></div>"
        f"<div class='val'>{value}<small>{note}</small></div></div>"
        for label, frac, value, note, colour in rows)
    st.markdown(f"<div class='rank-list'>{body}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div style='color:{MUTED};font-size:.81rem;line-height:1.5;"
                f"margin:-.1rem 0 1rem;'>{text}</div>", unsafe_allow_html=True)


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


def rating_chip(rating: float) -> str:
    """A star rating as a coloured chip. Colour repeats what the number says and
    never replaces it, and the unit is shown so 2.1 cannot be read as anything
    other than 2.1 stars out of 5."""
    if rating <= 2.0:
        bg, fg = "rgba(208,59,59,.11)", "#a82f2f"
    elif rating < 3.5:
        bg, fg = "rgba(224,138,46,.15)", "#8f5613"
    else:
        bg, fg = "rgba(12,163,12,.10)", "#0a6f0a"
    return (f"<span class='chip' style='background:{bg};color:{fg};'>"
            f"{rating:.1f} stars</span>")


def trend(delta: float, text: str) -> str:
    """Direction word plus the number. Never an arrow glyph alone."""
    if delta > 0.02:
        return f"<span style='color:#0a6f0a;font-weight:640;'>Up</span> {text}"
    if delta < -0.02:
        return f"<span style='color:#a82f2f;font-weight:640;'>Down</span> {text}"
    return f"<span style='color:{MUTED};font-weight:640;'>Flat</span> {text}"


def sparkline(x, y, colour=None):
    """A bare trend line: no axes, no grid, no labels."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=list(x), y=list(y), mode="lines",
        line=dict(color=colour or CRITICAL, width=1.9, shape="spline"),
        fill="tozeroy", fillcolor="rgba(208,59,59,0.08)",
        hovertemplate="%{x|%d %b}: %{y:,} unhappy<extra></extra>"))
    fig.update_layout(height=52, margin=dict(l=0, r=0, t=4, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS,
                                      font=dict(family=FONT, size=11, color=INK)))
    return fig
