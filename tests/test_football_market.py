"""The market baseline — football/market.py.

Odds are not probabilities: the three implied probabilities of a match sum to
more than 1, and that excess is the bookmaker's margin. A model compared
against raw 1/odds is being compared against a baseline that is deliberately
wrong in the bookmaker's favour, and will look better than it is.
"""

import numpy as np
import pytest

from football.common import OUTCOMES, PROBABILITY_COLUMNS
from football.market import (
    DEFAULT_METHOD,
    METHODS,
    compare_methods,
    fair_odds,
    implied_probabilities,
    market_probabilities,
    overround,
)
from football.sample_data import load_sample_and_preprocess

TYPICAL = [2.10, 3.40, 3.80]   # a realistic Premier League price
FAIR = [3.0, 3.0, 3.0]         # exactly zero margin


def test_a_fair_book_has_no_overround():
    assert overround(FAIR) == pytest.approx(0.0)


def test_a_real_book_has_a_few_percent_of_margin():
    assert 0.01 < overround(TYPICAL) < 0.10


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_a_probability_vector(method):
    p = implied_probabilities(TYPICAL, method=method)
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all() and (p < 1).all()


@pytest.mark.parametrize("method", METHODS)
def test_removing_the_margin_lowers_every_probability(method):
    """The raw 1/odds are inflated; de-margining can only take away."""
    raw = 1.0 / np.array(TYPICAL)
    assert (implied_probabilities(TYPICAL, method=method) <= raw + 1e-12).all()


@pytest.mark.parametrize("method", METHODS)
def test_a_zero_margin_book_is_left_alone(method):
    assert implied_probabilities(FAIR, method=method) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize("method", METHODS)
def test_the_ordering_of_the_outcomes_is_preserved(method):
    """De-margining redistributes; it must never reorder favourite and longshot."""
    p = implied_probabilities(TYPICAL, method=method)
    assert list(np.argsort(p)) == list(np.argsort(1.0 / np.array(TYPICAL)))


def test_the_shortest_price_is_the_most_likely_outcome():
    p = implied_probabilities([1.50, 4.00, 7.00])
    assert int(np.argmax(p)) == 0


def test_multiplicative_gives_the_longshot_more_than_additive():
    """The methods disagree exactly where it matters, and in a known direction.

    Multiplicative scales every probability down by the same factor, which
    leaves proportionally more on the longshot; additive takes an equal slice
    off each, which hits the longshot hardest.
    """
    multiplicative = implied_probabilities(TYPICAL, method="multiplicative")
    additive = implied_probabilities(TYPICAL, method="additive")
    longshot = int(np.argmin(multiplicative))
    favourite = int(np.argmax(multiplicative))
    assert multiplicative[longshot] > additive[longshot]
    assert multiplicative[favourite] < additive[favourite]


def test_the_methods_agree_to_within_a_couple_of_points():
    """Far apart enough to matter, close enough that a real edge survives the choice."""
    table = compare_methods(TYPICAL)
    spread = table[list(PROBABILITY_COLUMNS)].max() - table[list(PROBABILITY_COLUMNS)].min()
    assert (spread < 0.02).all()


def test_the_default_is_named_and_available():
    assert DEFAULT_METHOD in METHODS


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="Unknown method"):
        implied_probabilities(TYPICAL, method="wishful")


def test_odds_below_one_are_refused():
    """A price at or below 1.0 usually means the column is fractional or American."""
    with pytest.raises(ValueError, match="must be greater than 1.0"):
        implied_probabilities([0.9, 3.4, 3.8])
    with pytest.raises(ValueError, match="must be greater than 1.0"):
        implied_probabilities([1.0, 3.4, 3.8])


def test_the_wrong_number_of_odds_is_refused():
    with pytest.raises(ValueError, match="odds per match"):
        implied_probabilities([2.0, 3.0])


def test_a_missing_price_yields_nan_rather_than_dropping_the_row():
    """The result has to line up with the frame it came from."""
    out = implied_probabilities([[2.1, 3.4, 3.8], [np.nan, 3.0, 3.0]])
    assert out.shape == (2, 3)
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def test_a_single_triple_comes_back_as_a_single_vector():
    assert implied_probabilities(TYPICAL).shape == (len(OUTCOMES),)
    assert implied_probabilities([TYPICAL]).shape == (1, len(OUTCOMES))


def test_fair_odds_invert_the_probabilities():
    p = implied_probabilities(TYPICAL)
    assert fair_odds(p) == pytest.approx(1.0 / p)
    assert (fair_odds(p) > np.array(TYPICAL)).all(), "a fair price always beats a margined one"


# ------------------------------------------------------------ on a frame

def test_market_probabilities_attaches_three_columns():
    matches = load_sample_and_preprocess(seed=0)
    out = market_probabilities(matches)
    assert set(PROBABILITY_COLUMNS) <= set(out.columns)
    assert out[list(PROBABILITY_COLUMNS)].sum(axis=1).round(9).eq(1.0).all()
    assert len(out) == len(matches)


def test_the_odds_source_survives_the_conversion():
    """A probability from opening prices is not the same baseline as one from closing."""
    matches = load_sample_and_preprocess(seed=0)
    out = market_probabilities(matches)
    assert out.attrs["odds_source"] == matches.attrs["odds_source"]
    assert out.attrs["odds_are_closing"] is True
    assert out.attrs["probability_method"] == DEFAULT_METHOD


def test_a_frame_without_odds_columns_is_refused():
    matches = load_sample_and_preprocess(seed=0).drop(columns=["odds_home"])
    with pytest.raises(ValueError, match="no odds columns"):
        market_probabilities(matches)
