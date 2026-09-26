"""The loterias.com parser — lottery/utils/scraper.py.

`parse_results_page` does no I/O, which is the whole reason it is a separate
function, so it is tested here against saved-style HTML. As with
`tests/test_cycling_scraper.py`, that is the only verification available: the
sandbox this was written in cannot reach the site, so these fixtures are built
from the layout the parser documents rather than captured from it. They pin the
parser's *shape contract*, not the site's current markup.

What the tests pin is the behaviour that would otherwise fail silently:

- **A page that yields nothing raises.** A scraper that writes an empty CSV when
  the markup changes is the failure mode this module is shaped against — you
  find out months later, with the analysis already running on stale data.
- **A wrong ball count raises.** Five balls where six are expected is the
  fingerprint of a changed page, and a five-ball `Ball` string would flow
  straight into `preprocess_draws` as a malformed draw.
- **An unknown month raises rather than becoming "00".** A silently wrong month
  produces a date that parses, sorts wrongly, and looks entirely ordinary.
- **`scrape_years` tolerates an empty year only because another year worked.**
  That asymmetry is the one thing separating "this year has no results" from
  "the parser is broken", and it is invisible in the return value.
- **`merge_into` lets the existing rows win.** A hand-corrected local file must
  not be silently overwritten by a re-scrape.

Note `requests` is deliberately *not* imported by this module at import time
(see `fetch_year`), which is what lets this file run without it installed.
`fetch_year` itself is network code and is stubbed out rather than exercised.
"""

import pandas as pd
import pytest

from lottery.models.common import MAIN_BALLS_DRAWN
from lottery.utils.processor import preprocess_draws
from lottery.utils.scraper import (
    BALLS_PER_DRAW,
    ScrapeError,
    merge_into,
    parse_results_page,
    parse_spanish_date,
    parse_years,
    scrape_years,
)


def balls(numbers):
    items = "".join(f'<li class="ball">{n}</li>' for n in numbers)
    return f'<ul class="balls">{items}</ul>'


def row(date, main, revancha=None):
    """One results row in the site's shape: a linked date cell and a ball cell."""
    lists = balls(main) + (balls(revancha) if revancha else "")
    return (
        f'<tr><td class="centred"><a href="/baloto/resultados/x">{date}</a></td>'
        f'<td class="baloto">{lists}</td></tr>'
    )


def page(rows):
    return f"<html><body><table><tbody>{''.join(rows)}</tbody></table></body></html>"


DRAW = [3, 12, 19, 27, 41, 8]


# --- parse_spanish_date -------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("12 oct 2024", "12/10/2024"),
        ("12 octubre 2024", "12/10/2024"),      # keyed on the first three letters
        ("5 ene. 2021", "05/01/2021"),          # abbreviating full stop, day zero-padded
        ("sábado 7 sep 2019", "07/09/2019"),    # surrounding words are ignored
        ("1 septiembre 2019", "01/09/2019"),
    ],
)
def test_a_spanish_date_becomes_the_contract_s_dd_mm_yyyy(text, expected):
    assert parse_spanish_date(text) == expected


def test_an_unknown_month_raises_rather_than_becoming_month_zero():
    # The failure this guards: a month silently read as "00" produces a date
    # that still parses and still sorts -- into the wrong place, forever.
    with pytest.raises(ScrapeError, match="Unknown month"):
        parse_spanish_date("12 smarch 2024")


def test_a_date_that_cannot_be_read_at_all_raises():
    with pytest.raises(ScrapeError, match="Could not read a date"):
        parse_spanish_date("resultados recientes")


# --- parse_results_page -------------------------------------------------------

def test_a_draw_is_read_as_five_main_balls_then_the_superbalota():
    rows = parse_results_page(page([row("12 oct 2024", DRAW)]))

    assert rows == [{"Date": "12/10/2024", "Ball": "3-12-19-27-41-8", "Revancha": ""}]


def test_the_parsed_ball_string_is_what_the_data_contract_reads():
    """The point of the dash-separated string: it goes straight into the loader."""
    rows = parse_results_page(page([row("12 oct 2024", DRAW), row("14 oct 2024", [1, 2, 3, 4, 5, 6])]))

    df, balls_expanded = preprocess_draws(pd.DataFrame(rows)[["Date", "Ball"]], validate=False)

    assert balls_expanded.shape == (2, BALLS_PER_DRAW)
    assert list(balls_expanded.iloc[0]) == DRAW
    # The last column is the superbalota, which is what makes the site's order
    # load-bearing rather than incidental.
    assert balls_expanded.shape[1] == MAIN_BALLS_DRAWN + 1


def test_a_second_ball_list_is_kept_as_the_revancha():
    rows = parse_results_page(page([row("12 oct 2024", DRAW, revancha=[2, 4, 6, 8, 10, 1])]))

    assert rows[0]["Revancha"] == "2-4-6-8-10-1"


def test_rows_without_both_cells_are_not_draws_and_are_skipped():
    header = '<tr><th class="centred">Fecha</th><th>Resultado</th></tr>'
    unrelated = '<tr><td class="centred">publicidad</td></tr>'

    rows = parse_results_page(page([header, unrelated, row("12 oct 2024", DRAW)]))

    assert len(rows) == 1


def test_a_page_that_yields_no_draws_raises():
    # The whole shape of this module: an empty parse is a loud failure, because
    # writing an empty CSV when the markup changed is discovered months later.
    with pytest.raises(ScrapeError, match="Parsed 0 draws"):
        parse_results_page("<html><body><p>Sin resultados</p></body></html>", year=2024)


def test_an_empty_page_is_tolerated_only_when_the_caller_asks_for_it():
    assert parse_results_page("<html></html>", allow_empty=True) == []


def test_a_draw_with_the_wrong_number_of_balls_raises():
    with pytest.raises(ScrapeError, match="has 5 balls"):
        parse_results_page(page([row("12 oct 2024", [3, 12, 19, 27, 41])]))


def test_a_draw_with_too_many_balls_raises_too():
    with pytest.raises(ScrapeError, match="has 7 balls"):
        parse_results_page(page([row("12 oct 2024", [*DRAW, 9])]))


def test_a_ball_cell_with_no_ball_list_raises():
    html = page(['<tr><td class="centred"><a>12 oct 2024</a></td><td class="baloto"></td></tr>'])

    with pytest.raises(ScrapeError, match="No <ul class='balls'>"):
        parse_results_page(html)


def test_a_structural_failure_names_the_draw_it_happened_on():
    """The message has to identify the row, or a 300-draw page says only 'something is wrong'."""
    with pytest.raises(ScrapeError, match="12/10/2024"):
        parse_results_page(page([row("12 oct 2024", [1, 2, 3])]))


# --- scrape_years -------------------------------------------------------------

class FakeSession:
    """Stands in for requests.Session, which nothing here is allowed to need."""


def stub_fetch(monkeypatch, pages):
    """Make fetch_year serve `pages` (year -> html, or None for 'no such page')."""
    monkeypatch.setattr("lottery.utils.scraper.fetch_year", lambda year, session=None, **kw: pages[year])
    monkeypatch.setattr("lottery.utils.scraper.time.sleep", lambda seconds: None)


def test_years_with_no_results_are_skipped_when_another_year_parsed(monkeypatch, capsys):
    stub_fetch(monkeypatch, {
        2019: "<html></html>",                        # published nothing
        2020: None,                                   # no page at all
        2021: page([row("12 oct 2021", DRAW)]),
    })

    draws = scrape_years([2019, 2020, 2021], session=FakeSession())

    assert list(draws["Date"]) == ["12/10/2021"]
    assert "No results for: 2019, 2020" in capsys.readouterr().out


def test_no_draws_from_any_year_raises_because_that_is_a_broken_parser(monkeypatch):
    # The asymmetry that keeps the scraper honest: an empty year is tolerated
    # only because some other year worked. Nothing working means the parser is.
    stub_fetch(monkeypatch, {2019: "<html></html>", 2020: None})

    with pytest.raises(ScrapeError, match="No draws parsed from any of"):
        scrape_years([2019, 2020], session=FakeSession())


def test_a_structural_problem_raises_from_any_year_even_when_others_worked(monkeypatch):
    stub_fetch(monkeypatch, {
        2020: page([row("12 oct 2020", DRAW)]),
        2021: page([row("12 oct 2021", [1, 2, 3])]),
    })

    with pytest.raises(ScrapeError, match="has 3 balls"):
        scrape_years([2020, 2021], session=FakeSession())


# --- merge_into ---------------------------------------------------------------

def test_merging_into_a_missing_file_just_writes_the_scrape(tmp_path):
    path = tmp_path / "nested" / "final-final.csv"

    combined = merge_into(pd.DataFrame([{"Date": "12/10/2024", "Ball": "3-12-19-27-41-8"}]), str(path))

    assert path.exists()
    assert len(combined) == 1


def test_a_re_scrape_does_not_overwrite_a_hand_corrected_row(tmp_path):
    path = tmp_path / "final-final.csv"
    pd.DataFrame([{"Date": "12/10/2024", "Ball": "1-2-3-4-5-6"}]).to_csv(path, index=False)

    combined = merge_into(pd.DataFrame([{"Date": "12/10/2024", "Ball": "3-12-19-27-41-8"}]), str(path))

    assert list(combined["Ball"]) == ["1-2-3-4-5-6"]


def test_merged_draws_come_out_in_chronological_order(tmp_path):
    path = tmp_path / "final-final.csv"
    pd.DataFrame([{"Date": "12/10/2024", "Ball": "1-2-3-4-5-6"}]).to_csv(path, index=False)

    combined = merge_into(
        pd.DataFrame([
            {"Date": "03/01/2024", "Ball": "7-8-9-10-11-12"},
            {"Date": "31/12/2024", "Ball": "13-14-15-16-17-1"},
        ]),
        str(path),
    )

    # dd/mm/yyyy sorts wrongly as a string -- 03/01 after 31/12 -- so this is a
    # real property of merge_into, not a restatement of the input order.
    assert list(combined["Date"]) == ["03/01/2024", "12/10/2024", "31/12/2024"]


# --- parse_years --------------------------------------------------------------

@pytest.mark.parametrize(
    "spec,expected",
    [
        ("2024", [2024]),
        ("2020-2023", [2020, 2021, 2022, 2023]),
        ("2021,2024", [2021, 2024]),
        ("2020-2021, 2024", [2020, 2021, 2024]),
    ],
)
def test_a_year_spec_expands_to_the_years_it_names(spec, expected):
    assert parse_years(spec) == expected
