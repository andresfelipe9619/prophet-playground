"""Phone layout for the dashboard: one stylesheet, one Plotly layout, one place.

The dashboard is the end-user product and it is read on a phone, so a 375px
viewport is a first-class target rather than a degraded view of the desktop one.
Everything that makes that true lives here, for the same reason every piece of
explanatory copy lives in `ui.py`: three page modules each patching their own
CSS cannot be reviewed as a whole, and the rule "no surface is unreadable on a
phone" is only checkable if there is one surface to check.

Two mechanisms, both deliberately narrow:

* `apply()` injects a stylesheet whose rules are **entirely inside a
  `max-width: 640px` media query**. Nothing here changes the desktop layout, so
  a rule that turns out to be wrong is wrong only on phones and can be read as
  such. It is called once, from the shell, before any page renders.
* `responsive(fig)` is applied by `ui.chart` to every figure, so no call site
  changes and no chart can be added that misses it. It moves the legend off the
  right-hand side — a vertical legend costs a third of the width at 375px, which
  is the single largest reason these charts were unreadable on a phone — and
  turns off drag, so a finger on a chart scrolls the page instead of panning the
  axes.

The one layout decision worth stating: columns collapse to full width **except
when they hold metrics**, which go two-up. Streamlit's `st.columns(4)` of metric
tiles is the most common row in this dashboard; stacking it one-up turns four
numbers into four screens of scrolling, and the numbers are short enough to sit
side by side. `:has()` is what makes that distinction expressible in CSS; it is
available in Safari from 16.4, and a phone too old for it simply gets the
one-up stack, which is the safe direction to fail in.
"""

import streamlit as st

# The phone breakpoint. 640px is Streamlit's own — it is where the app already
# switches the sidebar to an overlay — so using anything else would put our
# layout change and the framework's on different sides of some screen width.
BREAKPOINT_PX = 640

# A plain string with a placeholder rather than an f-string: CSS is mostly braces,
# and an f-string would double every one of them, which is exactly the kind of
# noise that hides a misplaced rule. A test reads this literal and checks that
# every rule really is inside the media query.
_CSS_TEMPLATE = """
<style>
/* Shown only on a phone; `.desktop-only` is its complement. Both are outside
   the media query below because they must be *hidden* on the other side of it. */
.phone-only { display: none; }

@media (max-width: __BREAKPOINT__px) {
  .phone-only { display: block; }
  .desktop-only { display: none; }

  /* The wide layout's side padding is most of a phone's width. */
  [data-testid="stMainBlockContainer"] {
    padding: 1rem 0.75rem 4rem 0.75rem;
  }

  /* Columns stack, so a two-column comparison reads top to bottom instead of
     being squeezed into two unreadable halves. */
  [data-testid="stHorizontalBlock"] { gap: 0.5rem; }
  [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  /* ...except a row of metric tiles, which goes two-up: four short numbers
     stacked one-up is four screens of scrolling for one glance of information. */
  [data-testid="stColumn"]:has([data-testid="stMetric"]) {
    flex: 1 1 calc(50% - 0.25rem) !important;
    min-width: calc(50% - 0.25rem) !important;
  }

  /* Metric labels here are sentences ("Premio mayor necesario para que..."),
     not words: let them wrap rather than truncate to an ellipsis. */
  [data-testid="stMetricLabel"] p {
    white-space: normal !important;
    overflow: visible !important;
    font-size: 0.78rem !important;
    line-height: 1.25 !important;
  }
  [data-testid="stMetricValue"] { font-size: 1.3rem !important; }

  /* Ten numbered tabs do not fit; make the strip swipeable and drop the
     scrollbar that would otherwise sit on top of the labels. */
  [data-baseweb="tab-list"] {
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }
  [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
  [data-baseweb="tab"] {
    white-space: nowrap !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
  }
  [data-baseweb="tab"] p { font-size: 0.82rem !important; }

  /* Headings sized for a phone rather than scaled down from a monitor. */
  h1 { font-size: 1.45rem !important; }
  h2 { font-size: 1.2rem !important; }
  h3 { font-size: 1.05rem !important; }

  /* iOS zooms the whole page when a focused input is under 16px, and never
     zooms back out. Every text input, number input and select gets 16px. */
  input, textarea, select,
  [data-baseweb="input"] input,
  [data-baseweb="select"] input { font-size: 16px !important; }

  /* Thumb-sized targets: full-width buttons, roomier checkboxes and radios. */
  .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    width: 100% !important;
  }
  [data-testid="stCheckbox"] label, [data-testid="stRadio"] label {
    min-height: 2rem;
  }

  /* Nothing here sets the sidebar's width, and that is deliberate: Streamlit
     collapses it with a translate of a fixed number of pixels, not by its own
     width, so widening it leaves a strip of sidebar parked on top of the ☰
     button and the tap that would open it lands on the sidebar instead. Every
     control on a phone lives behind that button. Measured, not guessed — at
     85vw the collapsed sidebar covered x=0..31px and the button sits at 18px.

     The drag-to-resize handle is hidden because it is a mouse gesture that on a
     touch screen only adds a 8px-wide target next to the sidebar's edge. */
  [data-testid="stSidebarResizeHandle"] { display: none !important; }

  /* Wide frames scroll inside their own box instead of widening the page —
     a horizontally scrolling *page* makes every other column unreachable. */
  [data-testid="stDataFrame"], [data-testid="stTable"] {
    max-width: 100% !important;
    overflow-x: auto !important;
  }
  [data-testid="stMainBlockContainer"] { overflow-x: hidden; }
}
</style>
"""

_CSS = _CSS_TEMPLATE.replace("__BREAKPOINT__", str(BREAKPOINT_PX))

# Passed to every `st.plotly_chart`. `scrollZoom` off and the mode bar hidden
# are both about the same thing: on a touch screen the gestures that drive a
# Plotly chart are the gestures that scroll the page, and the page has to win.
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "responsive": True,
    "doubleClick": False,
}


def apply():
    """Inject the phone stylesheet. Called once, by the shell, before any page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def responsive(fig):
    """Make one Plotly figure legible at 375px. Applied by `ui.chart` to all of them.

    The legend is the whole point. Plotly's default puts it in a vertical column
    on the right, which on a phone leaves the plot itself about two thirds of an
    already narrow screen; moved above the plot it costs one line of height,
    which is the cheap direction. The top margin only grows when there is
    actually a legend to make room for — a single-trace figure keeps its tight
    crop.
    """
    has_legend = len(fig.data) > 1 if fig.layout.showlegend is None else bool(fig.layout.showlegend)
    fig.update_layout(
        autosize=True,
        dragmode=False,
        margin=dict(l=10, r=10, t=44 if has_legend else 10, b=40),
        font=dict(size=12),
    )
    if has_legend:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11),
        ))
    return fig


def phone_only(markdown):
    """Render `markdown` on phones only.

    Used for the one thing a phone reader needs and a desktop reader does not:
    knowing that the controls are behind the ☰ button rather than absent.
    """
    st.markdown(f'<div class="phone-only">{markdown}</div>', unsafe_allow_html=True)
