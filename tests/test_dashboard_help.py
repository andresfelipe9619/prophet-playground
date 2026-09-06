"""The dashboard's explain-on-hover rule, checked without launching Streamlit.

Two invariants, both of which only fail at runtime and only on a surface someone
actually opens — which on a three-domain dashboard can be a tab nobody visits for
months:

**Every HELP key a page asks for exists.** A missing one is a `KeyError` raised
from inside a tab body, so the whole page renders as an error.

**No page calls `st.plotly_chart` directly.** The house rule is that no chart
appears without saying what it does *not* mean, and `ui.chart()` is what enforces
it by requiring a HELP key. A bare `st.plotly_chart` is a chart that slipped past.

The dashboard imports streamlit and plotly, which are deliberately not in
requirements-test.txt, so this reads the source with `ast` and regexes instead of
importing it. That is also why it can check `dashboard/app.py` — the one file the
suite could never import.
"""

import ast
import os
import re

import pytest

DASHBOARD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
PAGES = ("app.py", "baloto_page.py", "football_page.py", "cycling_page.py")


def source(name):
    with open(os.path.join(DASHBOARD, name), encoding="utf-8") as handle:
        return handle.read()


def help_keys():
    """The keys of the single HELP dict in dashboard/ui.py."""
    tree = ast.parse(source("ui.py"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "HELP" for t in node.targets)):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("dashboard/ui.py no longer defines a HELP dict")


def keys_used(name):
    """Every HELP key a page asks for, whether directly or through the helpers."""
    text = source(name)
    used = set(re.findall(r'HELP\["([^"]+)"\]', text))
    for call in re.finditer(r"\b(?:section|chart)\(", text):
        # The helpers take their key as the last string argument of the call, and
        # calls wrap across lines, so scan a window rather than one line.
        window = text[call.end():call.end() + 400]
        strings = re.findall(r'"([a-z0-9_]+)"', window.split(")\n")[0])
        used.update(strings[-1:])
    return used


def test_the_help_dict_is_the_single_one():
    # Three page modules with their own inline strings could not be reviewed as a
    # set, which is the only way the "say what it does not mean" rule is checkable.
    for name in PAGES:
        assert "HELP = {" not in source(name), f"{name} declares a second HELP dict"


@pytest.mark.parametrize("name", PAGES)
def test_every_help_key_a_page_asks_for_exists(name):
    missing = sorted(keys_used(name) - help_keys())
    assert not missing, f"{name} asks for HELP keys that do not exist: {missing}"


@pytest.mark.parametrize("name", PAGES)
def test_no_page_renders_a_chart_outside_the_helper(name):
    assert "st.plotly_chart" not in source(name), (
        f"{name} calls st.plotly_chart directly. Route it through ui.chart(), which "
        "requires a HELP key — that is what stops a chart appearing with no explanation."
    )


def test_every_domain_in_the_selector_has_a_page_module():
    tree = ast.parse(source("app.py"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "DOMAINS" for t in node.targets)):
            modules = [v.elts[0].value for v in node.value.values]
            break
    else:
        raise AssertionError("dashboard/app.py no longer defines DOMAINS")

    for module in modules:
        path = os.path.join(DASHBOARD, module.split(".")[-1] + ".py")
        assert os.path.exists(path), f"{module} is in DOMAINS but {path} does not exist"
        # The suffix is not decoration: Streamlit puts the script's directory on
        # sys.path, so dashboard/football.py would shadow the football/ package.
        assert module.endswith("_page"), f"{module} needs the _page suffix to avoid shadowing"


def test_each_page_module_exposes_render():
    for name in PAGES[1:]:
        assert re.search(r"^def render\(\):", source(name), flags=re.M), \
            f"{name} does not define render(), which is what app.py dispatches to"
