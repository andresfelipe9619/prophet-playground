"""Exact prize combinatorics — lottery/analysis/prizes.py. Pure math, no data needed."""

import pytest

from lottery.analysis.prizes import (
    apply_payouts,
    breakeven_jackpot,
    category_probabilities,
    expected_value,
    main_match_probability,
    total_combinations,
)

TOTAL_TICKETS = 15_401_568  # C(43,5) * 16

# A plausible tier table; amounts are caller-supplied by design.
PAYOUTS = {
    (5, True): 5_000_000_000,
    (5, False): 100_000_000,
    (4, True): 10_000_000,
    (4, False): 500_000,
    (3, True): 100_000,
    (3, False): 20_000,
}
PRICE = 5_700


def test_total_combinations():
    assert total_combinations() == TOTAL_TICKETS


def test_main_match_probabilities_sum_to_one():
    assert sum(main_match_probability(k) for k in range(6)) == pytest.approx(1.0)


def test_every_category_probability_sums_to_one():
    assert category_probabilities()["probability"].sum() == pytest.approx(1.0)


def test_the_jackpot_is_one_in_every_ticket():
    table = category_probabilities()
    jackpot = table[(table["main_matches"] == 5) & table["super_match"]].iloc[0]
    assert jackpot["probability"] == pytest.approx(1 / TOTAL_TICKETS)
    assert jackpot["odds_one_in"] == pytest.approx(TOTAL_TICKETS)


def test_the_table_is_ordered_best_prize_first():
    table = category_probabilities()
    assert table.iloc[0]["main_matches"] == 5 and bool(table.iloc[0]["super_match"])
    assert "superbalota" in table.iloc[0]["category"]


def test_apply_payouts_defaults_missing_tiers_to_zero():
    table = apply_payouts(category_probabilities(), PAYOUTS)
    losing = table[(table["main_matches"] == 0) & ~table["super_match"]].iloc[0]
    assert losing["payout"] == 0.0
    assert losing["expected_contribution"] == 0.0


def test_expected_value_is_negative_at_a_realistic_prize_table():
    result = expected_value(category_probabilities(), PAYOUTS, PRICE)
    assert result["expected_value"] < 0
    assert 0 < result["return_to_player"] < 1
    assert result["expected_return"] == pytest.approx(result["expected_value"] + PRICE)


def test_win_probability_counts_only_paying_categories():
    result = expected_value(category_probabilities(), PAYOUTS, PRICE)
    table = result["table"]
    assert result["win_any_prize_probability"] == pytest.approx(
        table.loc[table["payout"] > 0, "probability"].sum()
    )
    assert result["odds_any_prize_one_in"] == pytest.approx(1 / result["win_any_prize_probability"])


def test_breakeven_jackpot_actually_breaks_even():
    """Solving for the jackpot and plugging it back in must land on zero EV."""
    jackpot = breakeven_jackpot(category_probabilities(), PAYOUTS, PRICE)
    assert jackpot > PAYOUTS[(5, True)]

    result = expected_value(category_probabilities(), {**PAYOUTS, (5, True): jackpot}, PRICE)
    assert result["expected_value"] == pytest.approx(0.0, abs=1e-6)
    assert result["return_to_player"] == pytest.approx(1.0)
