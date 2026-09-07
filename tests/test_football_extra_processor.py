"""The football-data.co.uk "extra" contract — football/extra_processor.py.

These files (new/COL.csv and friends) are a different shape from the main
league CSVs: Home/Away/HG/AG instead of HomeTeam/FTHG, several leagues and
seasons stacked in one file, and OPENING ODDS ONLY. This module maps them onto
the same tidy frame everything else consumes, and it must never let those
opening prices masquerade as a closing baseline.
"""

import os

import pytest

from football.extra_processor import (
    available_leagues,
    load_extra,
    preprocess_extra,
)
from football.processor import CLOSING_SOURCES, MatchFormatError

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "new_COL_sample.csv")


def test_maps_extra_columns_onto_the_tidy_shape():
    # The opening-odds warning is load-bearing (the baseline is soft) — pin it.
    with pytest.warns(UserWarning, match="opening"):
        matches = load_extra(FIXTURE, league="Colombia Primera A")
    assert list(matches.columns[:6]) == ["ds", "home_team", "away_team",
                                         "home_goals", "away_goals", "outcome"]
    assert (matches["home_team"].iloc[0], matches["away_team"].iloc[0]) == ("Millonarios", "Nacional")
    assert matches["outcome"].tolist() == ["H", "D", "A"]
    assert str(matches["ds"].iloc[0].date()) == "2023-02-04"


def test_odds_source_is_opening_and_never_closing():
    with pytest.warns(UserWarning, match="opening"):
        matches = load_extra(FIXTURE, league="Colombia Primera A")
    assert matches.attrs["odds_source"].endswith("_opening")
    assert matches.attrs["odds_source"] not in CLOSING_SOURCES
    assert matches.attrs["odds_are_closing"] is False


def test_league_filter_is_required_when_the_file_has_several():
    with pytest.raises(MatchFormatError):
        load_extra(FIXTURE, league=None)


def test_available_leagues_lists_what_is_in_the_file():
    assert set(available_leagues(FIXTURE)) == {"Colombia Primera A", "Colombia Primera B"}


def test_missing_core_columns_raise():
    import pandas as pd
    with pytest.raises(MatchFormatError):
        preprocess_extra(pd.DataFrame({"Home": ["A"], "Away": ["B"]}))
