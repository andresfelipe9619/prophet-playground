"""Outcome semantics — football/common.py.

Small file, but it owns an ordering that every probability vector, score and
column triple in the package depends on. Transposing two of them is a bug no
range check can catch, since all three are valid probabilities.
"""

import pytest

from football.common import (
    AWAY,
    DRAW,
    HOME,
    N_OUTCOMES,
    ODDS_COLUMNS,
    OUTCOMES,
    PROBABILITY_COLUMNS,
    outcome_from_goals,
    outcome_index,
    outcome_label,
)


def test_the_outcome_order_is_home_draw_away():
    """Load-bearing: every triple in this package is in this order."""
    assert OUTCOMES == (HOME, DRAW, AWAY) == ("H", "D", "A")
    assert N_OUTCOMES == 3


def test_the_column_triples_follow_the_same_order():
    assert PROBABILITY_COLUMNS == ("p_home", "p_draw", "p_away")
    assert ODDS_COLUMNS == ("odds_home", "odds_draw", "odds_away")
    for columns in (PROBABILITY_COLUMNS, ODDS_COLUMNS):
        assert [c.split("_")[-1] for c in columns] == ["home", "draw", "away"]


@pytest.mark.parametrize(
    "home_goals, away_goals, expected",
    [(2, 1, HOME), (0, 0, DRAW), (1, 1, DRAW), (0, 3, AWAY), (5, 4, HOME)],
)
def test_outcome_from_goals(home_goals, away_goals, expected):
    assert outcome_from_goals(home_goals, away_goals) == expected


def test_outcome_index_matches_the_declared_order():
    assert [outcome_index(o) for o in OUTCOMES] == [0, 1, 2]


def test_an_unknown_outcome_is_refused():
    with pytest.raises(ValueError, match="Unknown outcome"):
        outcome_index("X")


def test_every_outcome_has_a_label():
    assert {outcome_label(o) for o in OUTCOMES} == {"Home win", "Draw", "Away win"}
