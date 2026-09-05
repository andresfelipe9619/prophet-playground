"""The football data contract — football/processor.py.

The invariant this file is mostly about: **opening and closing odds are never
mixed.** Closing prices are the sharp baseline; opening prices are soft. One
`odds_home` column that silently means "closing" for some rows and "opening"
for others produces a bar that looks passable and is not real — the football
counterpart of mixing the two eras of Baloto, and just as invisible in the
frame's shape.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from football.common import ODDS_COLUMNS
from football.processor import (
    CLOSING_SOURCES,
    MatchFormatError,
    check_match_format,
    load_and_preprocess,
    load_seasons,
    odds_coverage,
    preprocess_matches,
    resolve_odds_source,
)

CORE = {
    "Date": ["09/08/2019", "10/08/2019", "11/08/2019"],
    "HomeTeam": ["Arsenal", "Chelsea", "Everton"],
    "AwayTeam": ["Burnley", "Fulham", "Leeds"],
    "FTHG": [2, 1, 0],
    "FTAG": [1, 1, 3],
}


def raw(**extra):
    return pd.DataFrame({**CORE, **extra})


def closing(**extra):
    return raw(AvgCH=[1.5, 2.0, 4.0], AvgCD=[4.0, 3.4, 3.6], AvgCA=[6.0, 3.8, 1.9], **extra)


def opening(**extra):
    return raw(AvgH=[1.5, 2.0, 4.0], AvgD=[4.0, 3.4, 3.6], AvgA=[6.0, 3.8, 1.9], **extra)


# --------------------------------------------------------------- the core

def test_the_required_columns_are_required():
    with pytest.raises(MatchFormatError, match="Missing required columns"):
        preprocess_matches(pd.DataFrame({"Date": ["09/08/2019"], "HomeTeam": ["Arsenal"]}))


def test_the_tidy_shape_is_produced():
    matches = preprocess_matches(closing(), validate=False)
    assert list(matches.columns)[:6] == ["ds", "home_team", "away_team",
                                          "home_goals", "away_goals", "outcome"]
    assert len(matches) == 3


def test_outcomes_are_derived_from_the_score():
    matches = preprocess_matches(closing(), validate=False)
    assert list(matches["outcome"]) == ["H", "D", "A"]


def test_dates_are_parsed_day_first():
    matches = preprocess_matches(closing(), validate=False)
    assert matches["ds"].iloc[0] == pd.Timestamp("2019-08-09")


def test_the_two_digit_year_form_is_accepted():
    """Older football-data files write dd/mm/yy rather than dd/mm/yyyy."""
    frame = closing()
    frame["Date"] = ["09/08/19", "10/08/19", "11/08/19"]
    matches = preprocess_matches(frame, validate=False)
    assert matches["ds"].iloc[0] == pd.Timestamp("2019-08-09")


def test_an_unparseable_date_raises_rather_than_shifting_a_season():
    frame = closing()
    frame["Date"] = ["09/08/2019", "not a date", "11/08/2019"]
    with pytest.raises(MatchFormatError, match="could not be parsed"):
        preprocess_matches(frame, validate=False)


def test_matches_come_back_sorted_by_date():
    frame = closing()
    frame["Date"] = ["11/08/2019", "09/08/2019", "10/08/2019"]
    matches = preprocess_matches(frame, validate=False)
    assert matches["ds"].is_monotonic_increasing


def test_unplayed_fixtures_are_dropped_with_a_warning():
    """football-data leaves the rest of an in-progress season blank."""
    frame = closing()
    frame.loc[2, ["FTHG", "FTAG"]] = [np.nan, np.nan]
    with pytest.warns(UserWarning, match="have not been played"):
        matches = preprocess_matches(frame)
    assert len(matches) == 2
    assert matches["home_goals"].dtype.kind == "i"


def test_team_names_are_stripped():
    frame = closing()
    frame["HomeTeam"] = [" Arsenal", "Chelsea ", "Everton"]
    matches = preprocess_matches(frame, validate=False)
    assert list(matches["home_team"]) == ["Arsenal", "Chelsea", "Everton"]


# ------------------------------------------------- the opening/closing guard

def test_closing_odds_win_over_opening_when_both_are_present():
    """The whole point: the sharp price is chosen, never the soft one."""
    frame = closing(AvgH=[9.9, 9.9, 9.9], AvgD=[9.9, 9.9, 9.9], AvgA=[9.9, 9.9, 9.9])
    matches = preprocess_matches(frame, validate=False)
    assert matches.attrs["odds_source"] == "market_closing_average"
    assert matches.attrs["odds_are_closing"] is True
    assert matches["odds_home"].iloc[0] == 1.5, "the opening column must not win"


def test_a_market_average_wins_over_a_single_bookmaker():
    frame = closing(B365CH=[9.9] * 3, B365CD=[9.9] * 3, B365CA=[9.9] * 3)
    matches = preprocess_matches(frame, validate=False)
    assert matches.attrs["odds_source"] == "market_closing_average"
    assert matches["odds_home"].iloc[0] == 1.5


def test_a_single_book_is_used_when_no_average_is_available():
    frame = raw(B365CH=[1.5, 2.0, 4.0], B365CD=[4.0, 3.4, 3.6], B365CA=[6.0, 3.8, 1.9])
    matches = preprocess_matches(frame, validate=False)
    assert matches.attrs["odds_source"] == "bet365_closing"
    assert matches.attrs["odds_are_closing"] is True


def test_an_opening_only_file_is_marked_as_such():
    matches = preprocess_matches(opening(), validate=False)
    assert matches.attrs["odds_source"] == "market_opening_average"
    assert matches.attrs["odds_are_closing"] is False
    assert matches.attrs["odds_source"] not in CLOSING_SOURCES


def test_a_missing_closing_price_is_never_filled_from_an_opening_one():
    """The failure mode this module exists to prevent, tested directly.

    Row 1 has no closing price but does have an opening one. It must come back
    NaN: a visible hole is recoverable, a silently mixed baseline is not.
    """
    frame = closing(AvgH=[1.9, 1.9, 1.9], AvgD=[3.5, 3.5, 3.5], AvgA=[4.2, 4.2, 4.2])
    frame.loc[1, ["AvgCH", "AvgCD", "AvgCA"]] = [np.nan, np.nan, np.nan]

    matches = preprocess_matches(frame, validate=False)
    assert matches.attrs["odds_source"] == "market_closing_average"
    assert matches[list(ODDS_COLUMNS)].iloc[1].isna().all()
    assert odds_coverage(matches) == pytest.approx(2 / 3)


def test_a_partial_closing_price_is_discarded_whole():
    """Two of three prices is not a market — it cannot be normalised."""
    frame = closing()
    frame.loc[0, "AvgCD"] = np.nan
    matches = preprocess_matches(frame, validate=False)
    assert matches[list(ODDS_COLUMNS)].iloc[0].isna().all()


def test_an_impossible_price_is_treated_as_missing():
    """A decimal price at or below 1.0 pays nothing and cannot be real."""
    frame = closing()
    frame.loc[0, "AvgCH"] = 1.0
    matches = preprocess_matches(frame, validate=False)
    assert matches[list(ODDS_COLUMNS)].iloc[0].isna().all()


def test_a_file_with_no_odds_at_all_yields_nan_columns():
    matches = preprocess_matches(raw(), validate=False)
    assert matches.attrs["odds_source"] is None
    assert matches[list(ODDS_COLUMNS)].isna().all().all()
    assert odds_coverage(matches) == 0.0


def test_resolve_odds_source_follows_the_documented_priority():
    assert resolve_odds_source(["AvgCH", "AvgCD", "AvgCA", "AvgH", "AvgD", "AvgA"])["name"] \
        == "market_closing_average"
    assert resolve_odds_source(["AvgH", "AvgD", "AvgA"])["name"] == "market_opening_average"
    assert resolve_odds_source(["Date"]) is None


def test_a_half_present_source_is_not_selected():
    """Two of the three columns is not a usable source."""
    assert resolve_odds_source(["AvgCH", "AvgCD"]) is None


# ------------------------------------------------------------- the report

def test_a_closing_odds_file_needs_no_warning():
    assert check_match_format(preprocess_matches(closing(), validate=False)) is None


def test_an_opening_only_file_warns_that_the_bar_is_soft():
    report = check_match_format(preprocess_matches(opening(), validate=False))
    assert report is not None
    assert "opening prices" in report["message"]
    assert "unproven" in report["message"]
    assert report["odds_are_closing"] is False


def test_a_file_without_odds_warns_that_there_is_no_bar_at_all():
    report = check_match_format(preprocess_matches(raw(), validate=False))
    assert "no market baseline" in report["message"]
    assert "beating chance is not the bar" in report["message"]


def test_incomplete_coverage_is_reported_not_hidden():
    frame = closing()
    frame.loc[1, ["AvgCH", "AvgCD", "AvgCA"]] = np.nan
    report = check_match_format(preprocess_matches(frame, validate=False))
    assert report["odds_coverage"] == pytest.approx(2 / 3)
    assert "mix two different markets" in report["message"]


def test_validation_warns_rather_than_raising():
    """Like the lottery loader: the rows are real matches, the caller may want them."""
    with pytest.warns(UserWarning, match="opening prices"):
        preprocess_matches(opening())


def test_validate_false_stays_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        preprocess_matches(opening(), validate=False)


# ------------------------------------------------------------- from disk

def test_load_and_preprocess_reads_a_csv(tmp_path):
    path = tmp_path / "E0.csv"
    closing().to_csv(path, index=False)
    matches = load_and_preprocess(str(path), validate=False)
    assert len(matches) == 3
    assert matches.attrs["odds_source"] == "market_closing_average"


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_preprocess(str(tmp_path / "nope.csv"))


def test_closing_odds_only_refuses_a_soft_file(tmp_path):
    path = tmp_path / "E0.csv"
    opening().to_csv(path, index=False)
    with pytest.raises(MatchFormatError, match="no closing odds"):
        load_and_preprocess(str(path), validate=False, closing_odds_only=True)


def test_seasons_sharing_a_source_concatenate(tmp_path):
    paths = []
    for i, year in enumerate((2019, 2020)):
        frame = closing()
        frame["Date"] = [f"09/08/{year}", f"10/08/{year}", f"11/08/{year}"]
        path = tmp_path / f"E0_{year}.csv"
        frame.to_csv(path, index=False)
        paths.append(str(path))

    merged = load_seasons(paths, validate=False)
    assert len(merged) == 6
    assert merged["ds"].is_monotonic_increasing
    assert merged.attrs["odds_source"] == "market_closing_average"


def test_seasons_with_different_odds_sources_refuse_to_merge(tmp_path):
    """The realistic case: 2018/19 has no closing odds and 2019/20 does.

    Stacking them would put two different markets in one column, and every
    model evaluated on the result would be judged against two bars at once.
    """
    old = tmp_path / "E0_2018.csv"
    opening().to_csv(old, index=False)
    new = tmp_path / "E0_2019.csv"
    closing().to_csv(new, index=False)

    with pytest.raises(MatchFormatError, match="different odds sources"):
        load_seasons([str(old), str(new)], validate=False)


def test_load_seasons_needs_at_least_one_file():
    with pytest.raises(MatchFormatError, match="No season files"):
        load_seasons([])
