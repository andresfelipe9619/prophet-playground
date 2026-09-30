"""docs/data-sources.md, held to the code it describes.

A recipe in a document is code nobody runs, which is how documentation goes
wrong without anyone noticing: a contract gains a column, the recipe keeps
showing the old shape, and the first reader to try it finds out. So every
```python block on that page is executed here, from a scratch directory, and
each one ends by loading its own output back through the module that owns the
contract.

The league tables are pinned the same way. The page previously described a
Colombian football-data file that football-data does not publish; a table that
must equal `LEAGUES` / `EXTRA_LEAGUES` cannot drift from the downloader again
without a failing test.
"""

import os
import re
import warnings

import pytest

from football.downloader import EXTRA_LEAGUES, LEAGUES

PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "docs", "data-sources.md")


def page():
    with open(PAGE, encoding="utf-8") as handle:
        return handle.read()


def python_blocks():
    return re.findall(r"```python\n(.*?)```", page(), flags=re.S)


def section(heading):
    """The text of one `###` section, up to the next heading of any level."""
    text = page()
    start = text.index(heading)
    following = re.search(r"\n#{2,3} ", text[start + len(heading):])
    return text[start:start + len(heading) + (following.start() if following else len(text))]


def codes_in(text):
    return set(re.findall(r"\| `([A-Z0-9]{2,3})` \|", text))


def test_the_page_carries_the_recipes_it_promises():
    assert len(python_blocks()) == 4


@pytest.mark.parametrize("index", range(4))
def test_every_recipe_runs_and_loads_back_through_its_contract(index, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        # The Colombia recipe triggers the soft-baseline warning by design.
        warnings.simplefilter("ignore", UserWarning)
        exec(compile(python_blocks()[index], f"data-sources.md recipe {index + 1}", "exec"), {})
    assert os.listdir(tmp_path / "exported_data")


def test_the_main_league_table_is_exactly_what_the_downloader_accepts():
    assert codes_in(section("### 1.1 The main leagues")) == set(LEAGUES)


def test_the_extra_league_table_is_exactly_what_the_downloader_accepts():
    table = codes_in(section("### 1.2 The sixteen"))
    assert table == set(EXTRA_LEAGUES)
    assert "COL" not in table


def test_the_catalogue_names_at_least_twenty_sources():
    rows = re.findall(r"^\| (\d+) \| ", page(), flags=re.M)
    numbers = [int(n) for n in rows]
    assert numbers == list(range(1, len(numbers) + 1))
    assert len(numbers) >= 20
