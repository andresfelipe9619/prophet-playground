"""The procyclingstats parser — cycling/scraper.py.

`parse_results_page` does no I/O, so it is tested against saved-style HTML here.
That is the only kind of verification available: the sandbox this was written in
blocks the site, so **the parser has never seen a real page** and these fixtures
are written from its documented layout, not captured from it.

What the tests pin is the behaviour that would otherwise fail silently:

- **Gaps become totals.** A page publishes the winner's elapsed time and
  everyone else's gap. Storing the gaps would produce a frame that looks normal
  and ranks the field backwards by four hours.
- **`,,` means "same time as the rider above".** Read literally it is a blank,
  which would quietly drop the time of most of a bunch finish.
- **The table is found by its headers, not its CSS classes.** A class-based
  selector that misses returns zero rows; a header row that no longer says
  "Rnk" is worth stopping on.
- **An unknown rank marker raises.** That is how a page whose markup changed
  gets caught instead of turning riders into 'not ranked' rows.
"""

import os

import pandas as pd
import pytest

from cycling.common import DNF, FINISHED, GC, ONE_DAY, STAGE
from cycling.processor import preprocess_results, time_order_violations
from cycling.scraper import (
    ScrapeError,
    default_path,
    find_results_table,
    merge_into,
    page_url,
    parse_race_date,
    parse_results_page,
    parse_stages,
)

HEADER = "<th>Rnk</th><th>BIB</th><th>H2H</th><th>Rider</th><th>Age</th><th>Team</th>" \
         "<th>UCI</th><th>Pnt</th><th>Time</th>"


def row(rank, rider, team, time, bib=1):
    return (
        f"<tr><td>{rank}</td><td>{bib}</td><td></td>"
        f'<td><a href="rider/{rider.lower().replace(" ", "-")}">{rider}</a>'
        f'<span class="hide"> {team}</span></td><td>25</td>'
        f'<td><a href="team/{team.lower().replace(" ", "-")}-2024">{team}</a></td>'
        f"<td>200</td><td>100</td><td>{time}</td></tr>"
    )


def page(body_rows, date="29 June 2024", sidebar=True):
    """A results page in the site's shape: an info list, a small sidebar table
    sharing the results layout, and the classification itself."""
    sidebar_html = (
        f"<table class='basic'><thead><tr><th>Rnk</th><th>Rider</th></tr></thead><tbody>"
        f"<tr><td>1</td><td><a href='rider/x'>Sprint Winner</a></td></tr></tbody></table>"
        if sidebar else ""
    )
    date_html = (f"<ul class='infolist'><li><div>Date:</div><div>{date}</div></li>"
                 "<li><div>Start time:</div><div>13:00</div></li></ul>") if date else ""
    return (
        f"<html><body>{date_html}{sidebar_html}"
        f"<table class='results basic'><thead><tr>{HEADER}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></body></html>"
    )


STAGE_PAGE = page([
    row(1, "Rider A", "Team One", "4:15:22"),
    row(2, "Rider B", "Team Two", "0:14"),
    row(3, "Rider C", "Team Three", ",,"),
    row(4, "Rider D", "Team Four", "1:02"),
    row("DNF", "Rider E", "Team Five", ""),
])


def parsed(html=STAGE_PAGE, **kwargs):
    return parse_results_page(html, race="tour-de-france", kind=STAGE, stage=1, **kwargs)


# ---------------------------------------------------------------- the table

def test_the_results_table_is_found_by_its_headers():
    table, fields = find_results_table(__import__("bs4").BeautifulSoup(STAGE_PAGE, "html.parser"))
    assert fields[0] == "rank"
    assert fields[3] == "rider"
    assert fields[-1] == "time"


def test_the_longest_matching_table_wins_over_a_sidebar_sharing_its_layout():
    rows = parsed()
    assert "Sprint Winner" not in set(r["Rider"] for r in rows)
    assert len(rows) == 5


def test_a_page_with_no_results_table_raises_and_says_what_it_found():
    with pytest.raises(ScrapeError, match="No table on this page"):
        parsed("<html><body><table><tr><th>Date</th><th>Winner</th></tr></table></body></html>")


def test_an_empty_result_raises_unless_the_caller_expects_it():
    empty = page([])
    with pytest.raises(ScrapeError, match="Parsed 0 riders"):
        parsed(empty)
    assert parsed(empty, allow_empty=True) == []


def test_a_page_of_only_abandons_is_refused():
    with pytest.raises(ScrapeError, match="Every row"):
        parsed(page([row("DNF", "Rider E", "Team Five", "")]))


# ----------------------------------------------------------------- the rows

def test_the_rider_name_excludes_the_team_smuggled_into_the_same_cell():
    assert [r["Rider"] for r in parsed()][:2] == ["Rider A", "Rider B"]
    assert [r["Team"] for r in parsed()][:2] == ["Team One", "Team Two"]


def test_gaps_are_resolved_against_the_winners_time():
    times = [r["TimeSeconds"] for r in parsed()]
    assert times[0] == 15322.0            # the winner's own elapsed time
    assert times[1] == 15322.0 + 14.0     # a gap, added to it
    assert times[3] == 15322.0 + 62.0


def test_the_same_time_marker_inherits_the_gap_above_it():
    # ',,' read literally is a blank, which would drop most of a bunch finish.
    times = [r["TimeSeconds"] for r in parsed()]
    assert times[2] == times[1]


def test_an_abandon_keeps_the_rider_with_no_rank_and_no_time():
    last = parsed()[-1]
    assert (last["Status"], last["Rank"], last["TimeSeconds"]) == (DNF, "", "")
    assert last["Rider"] == "Rider E"


def test_an_unknown_rank_marker_raises():
    with pytest.raises(ScrapeError, match="Unknown rank marker"):
        parsed(page([row(1, "Rider A", "Team One", "4:15:22"),
                     row("HORS", "Rider B", "Team Two", "")]))


def test_an_unreadable_time_raises_rather_than_being_dropped():
    with pytest.raises(ScrapeError, match="Could not read"):
        parsed(page([row(1, "Rider A", "Team One", "4h15m")]))


def test_without_the_winners_time_no_gap_is_written_as_a_total():
    # Nothing to anchor to, so the times are NaN rather than gaps in a column
    # that is supposed to hold totals.
    rows = parsed(page([row(1, "Rider A", "Team One", ""),
                        row(2, "Rider B", "Team Two", "0:14")]))
    assert [r["TimeSeconds"] for r in rows] == ["", ""]


def test_what_it_parses_satisfies_the_contract():
    results = preprocess_results(pd.DataFrame(parsed()), validate=True)
    assert results.attrs["result_kind"] == STAGE
    assert not time_order_violations(results).any()
    assert (results["status"] == FINISHED).sum() == 4


# ----------------------------------------------------------------- the date

def test_the_date_comes_off_the_page_when_it_is_published():
    assert parse_race_date(STAGE_PAGE) == "29/06/2024"
    assert parsed()[0]["Date"] == "29/06/2024"


def test_a_day_first_numeric_date_is_read_day_first():
    assert parse_race_date(page([], date="05/06/2024")) == "05/06/2024"


def test_a_page_with_no_date_raises_unless_one_is_given():
    undated = page([row(1, "Rider A", "Team One", "4:15:22")], date=None)
    with pytest.raises(ScrapeError, match="No date found"):
        parsed(undated)
    assert parsed(undated, date="29/06/2024")[0]["Date"] == "29/06/2024"


# ------------------------------------------------------- urls, paths, merge

def test_the_url_names_the_page_for_each_kind():
    assert page_url("tour-de-france", 2024, STAGE, 1).endswith("race/tour-de-france/2024/stage-1")
    assert page_url("tour-de-france", 2024, GC).endswith("race/tour-de-france/2024/gc")
    assert page_url("milano-sanremo", 2024, ONE_DAY).endswith("race/milano-sanremo/2024/result")


def test_a_stage_result_without_a_stage_number_is_refused():
    with pytest.raises(ValueError, match="needs a stage number"):
        page_url("tour-de-france", 2024, STAGE)


def test_the_file_name_carries_the_kind_because_one_file_holds_one_kind():
    assert default_path("tour-de-france", 2024, STAGE, "d") == os.path.join(
        "d", "tour-de-france_2024_stage.csv")
    assert default_path("tour-de-france", 2024, GC, "d").endswith("_gc.csv")


def test_stage_specs_parse():
    assert parse_stages("1-3") == [1, 2, 3]
    assert parse_stages("1,3,5") == [1, 3, 5]
    assert parse_stages("7") == [7]


def test_a_rescrape_merges_without_duplicating_a_rider(tmp_path):
    path = str(tmp_path / "tour.csv")
    first = pd.DataFrame(parsed())
    merge_into(first, path)
    combined = merge_into(first, path)
    assert len(combined) == len(first)


def test_an_existing_hand_corrected_row_is_not_overwritten(tmp_path):
    path = str(tmp_path / "tour.csv")
    corrected = pd.DataFrame(parsed())
    corrected.loc[0, "Team"] = "Corrected Team"
    merge_into(corrected, path)
    combined = merge_into(pd.DataFrame(parsed()), path)
    assert combined.loc[0, "Team"] == "Corrected Team"


def test_a_second_stage_extends_the_file(tmp_path):
    path = str(tmp_path / "tour.csv")
    merge_into(pd.DataFrame(parsed()), path)
    second = pd.DataFrame(parse_results_page(STAGE_PAGE, race="tour-de-france", kind=STAGE,
                                            stage=2, date="30/06/2024"))
    combined = merge_into(second, path)
    assert len(combined) == 10
    assert set(combined["Stage"]) == {1, 2}


# --------------------------------------------------------------- the driver

def fake_fetch(pages):
    """A stand-in for fetch_page: a dict of url suffix -> html, else None (404)."""
    def fetch(url, session=None, **kwargs):
        for suffix, html in pages.items():
            if url.endswith(suffix):
                return html
        return None
    return fetch


def test_a_stage_not_yet_published_is_skipped_not_fatal(monkeypatch):
    from cycling.scraper import scrape_results
    monkeypatch.setattr("cycling.scraper.fetch_page", fake_fetch({"stage-1": STAGE_PAGE}))
    scraped = scrape_results("tour-de-france", 2024, kind=STAGE, stages=[1, 2, 3], delay=0)
    assert set(scraped["Stage"]) == {1}


def test_nothing_published_at_all_raises(monkeypatch):
    from cycling.scraper import scrape_results
    monkeypatch.setattr("cycling.scraper.fetch_page", fake_fetch({}))
    with pytest.raises(ScrapeError, match="No results parsed"):
        scrape_results("tour-de-france", 2024, kind=STAGE, stages=[1, 2], delay=0)


def test_a_structural_problem_on_any_stage_still_raises(monkeypatch):
    # A missing page is tolerated because other pages worked; a changed table is
    # not, because dropping that stage would hide the change.
    from cycling.scraper import scrape_results
    monkeypatch.setattr("cycling.scraper.fetch_page", fake_fetch({
        "stage-1": STAGE_PAGE,
        "stage-2": page([row(1, "Rider A", "Team One", "4:15:22"),
                         row("HORS", "Rider B", "Team Two", "")]),
    }))
    with pytest.raises(ScrapeError, match="Unknown rank marker"):
        scrape_results("tour-de-france", 2024, kind=STAGE, stages=[1, 2], delay=0)
