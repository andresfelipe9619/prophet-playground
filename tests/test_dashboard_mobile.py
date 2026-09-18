"""The phone layout's invariants, checked without launching Streamlit.

The dashboard is read on a phone, and a layout rule breaks in a way no unit test
of a model would ever see: the page renders, every number is right, and a control
is unreachable. These pin the three properties that a screenshot does not show.

Like `test_dashboard_help.py`, this reads the source rather than importing it —
`dashboard/mobile.py` imports streamlit, which is deliberately absent from
requirements-test.txt.
"""

import ast
import os
import re

DASHBOARD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")


def source(name):
    with open(os.path.join(DASHBOARD, name), encoding="utf-8") as handle:
        return handle.read()


def css():
    """The stylesheet literal from dashboard/mobile.py, without importing it.

    `_CSS_TEMPLATE` is a plain string rather than an f-string precisely so this
    works: `literal_eval` can read it, and CSS full of doubled braces is CSS
    nobody proofreads.
    """
    tree = ast.parse(source("mobile.py"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "_CSS_TEMPLATE" for t in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError("dashboard/mobile.py no longer defines _CSS_TEMPLATE")


def strip_comments(text):
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def test_every_rule_is_behind_the_phone_breakpoint():
    # The point of the module: a rule that turns out to be wrong is wrong only on
    # phones. One rule outside the media query silently restyles the desktop app,
    # and nobody looking at a phone would ever see it.
    before_media, _, _ = css().partition("@media")
    body = strip_comments(before_media).replace("<style>", "")
    selectors = [rule.split("{")[0].strip() for rule in body.split("}") if "{" in rule]
    assert selectors == [".phone-only"], (
        "only the .phone-only utility may sit outside the media query — it has to be "
        f"hidden on the desktop side of it. Found: {selectors}"
    )


def test_the_sidebar_width_is_never_set():
    # Measured, not theoretical. Streamlit collapses the sidebar by translating it
    # a fixed number of pixels, not by its own width, so setting a width leaves a
    # strip of collapsed sidebar sitting on top of the ☰ button — and on a phone
    # every control in the app is behind that button. The app looks perfect in a
    # screenshot and cannot be used.
    rules = strip_comments(css())
    for match in re.finditer(r'\[data-testid="stSidebar"\][^{}]*\{([^}]*)\}', rules):
        assert not re.search(r"\bwidth\b", match.group(1)), (
            "dashboard/mobile.py sets a width on the sidebar. That parks the collapsed "
            "sidebar over the ☰ button; see docs/local-setup.md."
        )


def test_charts_go_through_the_responsive_layout():
    # Same reasoning as ui.chart owning the HELP key: applying this at the call
    # sites means the next chart added is the one that forgets.
    ui = source("ui.py")
    plot_call = re.search(r"st\.plotly_chart\((.*?)\)\n", ui, flags=re.S)
    assert plot_call, "dashboard/ui.py no longer renders the chart"
    assert "mobile.responsive(" in plot_call.group(1), \
        "ui.chart must pass the figure through mobile.responsive()"
    assert "mobile.PLOTLY_CONFIG" in plot_call.group(1), \
        "ui.chart must pass mobile.PLOTLY_CONFIG, or touch gestures fight page scrolling"


def test_the_shell_installs_the_stylesheet():
    # Injected once, before any page renders. A page doing it itself would be the
    # start of three page modules with their own CSS.
    app = source("app.py")
    assert "mobile.apply()" in app, "dashboard/app.py must call mobile.apply()"
    assert "mobile.apply()" not in source("baloto_page.py"), \
        "the stylesheet belongs to the shell, not to a page"
