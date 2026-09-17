"""Edge, staking and the two different bars — football/value.py.

The invariant worth more than all the arithmetic here is the middle state. A
model can be more optimistic than the de-margined market and still have no bet,
because the margin sits between the two bars; that band is where betting systems
live, and `test_the_margin_band_is_a_real_band` is what stops a refactor
collapsing it into "model > market, therefore bet".
"""

import numpy as np
import pytest

from football.common import OUTCOMES, outcome_index
from football.market import implied_probabilities
from football.value import (
    DISAGREEMENT_ONLY,
    NO_VALUE,
    VALUE,
    break_even_probability,
    classify,
    expected_value,
    kelly_fraction,
    margin_cost,
    value_table,
)

# A typical priced match: ~5% margin, home favourite.
ODDS = np.array([2.10, 3.40, 3.80])


def test_break_even_is_the_raw_price():
    assert break_even_probability(2.0) == pytest.approx(0.5)
    assert break_even_probability(ODDS).sum() > 1.0  # the margin, by definition


def test_a_fair_coin_at_fair_odds_has_no_edge():
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert kelly_fraction(0.5, 2.0, fraction=1.0) == pytest.approx(0.0)


def test_kelly_matches_the_textbook_case():
    # The canonical example: a 60% shot at even money stakes 20% of the bankroll.
    assert kelly_fraction(0.6, 2.0, fraction=1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.6, 2.0, fraction=0.25) == pytest.approx(0.05)


def test_a_bad_bet_stakes_nothing_rather_than_going_negative():
    # A negative Kelly means "this bet is bad", not "lay it". Zero is the
    # actionable form of that answer, and the only one a stake column can show.
    assert kelly_fraction(0.3, 2.0, fraction=1.0) == 0.0
    assert (kelly_fraction([0.1, 0.2, 0.3], ODDS) >= 0).all()


def test_the_margin_band_is_a_real_band():
    """The state most betting systems mistake for an edge.

    The model is more optimistic than the de-margined market on the home win,
    but still short of the raw price it would have to beat. That is a
    disagreement, not a bet, and the two must not classify the same.
    """
    market = implied_probabilities(ODDS)
    home = outcome_index("H")
    fair, raw = market[home], 1.0 / ODDS[home]
    assert fair < raw  # the margin sits between them

    between = (fair + raw) / 2.0
    probabilities = market.copy()
    probabilities[home] = between
    assert classify(probabilities, market, ODDS)[home] == DISAGREEMENT_ONLY
    assert expected_value(between, ODDS[home]) < 0  # and it would lose money

    probabilities[home] = raw + 0.05
    assert classify(probabilities, market, ODDS)[home] == VALUE
    assert expected_value(raw + 0.05, ODDS[home]) > 0

    probabilities[home] = fair - 0.05
    assert classify(probabilities, market, ODDS)[home] == NO_VALUE


def test_agreeing_with_the_market_is_never_value():
    # A model that reproduces the de-margined price exactly has found nothing,
    # and must not be shown a stake on any outcome.
    market = implied_probabilities(ODDS)
    assert set(classify(market, market, ODDS)) == {NO_VALUE}
    assert (kelly_fraction(market, ODDS) == 0).all()


def test_margin_cost_is_the_width_of_that_band():
    cost = margin_cost(ODDS)
    assert (cost > 0).all()
    assert cost == pytest.approx(break_even_probability(ODDS) - implied_probabilities(ODDS))


def test_value_table_shows_both_bars_for_every_outcome():
    table = value_table([0.55, 0.25, 0.20], ODDS, bankroll=1_000_000)
    assert list(table["outcome"]) == list(OUTCOMES)
    # Both bars present and distinct: one is what the market thinks, the other
    # is what you have to beat to profit.
    assert (table["break_even_probability"] > table["market_probability"]).all()
    assert table["market_probability"].sum() == pytest.approx(1.0)
    assert (table["stake"] == table["kelly_stake"] * 1_000_000).all()


def test_the_table_only_stakes_where_it_says_value():
    table = value_table([0.55, 0.25, 0.20], ODDS)
    staked = table[table["kelly_stake"] > 0]
    assert set(staked["verdict"]) <= {VALUE}
    assert (staked["expected_value"] > 0).all()
