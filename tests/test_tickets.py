"""Ticket generation, checking and strategy measurement — lottery/analysis/tickets.py.

The load-bearing invariant is the Bonferroni column on `compare_strategies`:
k strategies scored against the same draws get k chances at a false positive,
so a table that reports one uncorrected verdict has reintroduced the bug.
"""

import numpy as np
import pandas as pd
import pytest

from lottery.analysis.prizes import total_combinations
from lottery.analysis.tickets import (
    MAIN_NUMBERS,
    STRATEGIES,
    Ticket,
    check_against_history,
    check_ticket,
    cold_ticket,
    compare_strategies,
    draw_from_row,
    evaluate_strategy,
    generate_portfolio,
    history_summary,
    hot_ticket,
    portfolio_coverage,
    random_ticket,
    stability_check,
    ticket_from_predictions,
)
from lottery.models.common import MAIN_BALLS_DRAWN


# ------------------------------------------------------------------ validity

def test_a_ticket_needs_five_distinct_main_numbers():
    with pytest.raises(ValueError, match="distinct main numbers"):
        Ticket(main=(1, 1, 2, 3, 4), super_ball=5)


@pytest.mark.parametrize("main", [(0, 2, 3, 4, 5), (1, 2, 3, 4, 44)])
def test_main_numbers_must_be_in_range(main):
    with pytest.raises(ValueError, match="outside"):
        Ticket(main=main, super_ball=5)


@pytest.mark.parametrize("super_ball", [0, 17])
def test_superbalota_must_be_in_range(super_ball):
    with pytest.raises(ValueError, match="Superbalota"):
        Ticket(main=(1, 2, 3, 4, 5), super_ball=super_ball)


def test_ticket_string_is_sorted_for_display():
    assert str(Ticket(main=(41, 3, 27, 12, 19), super_ball=8)) == "3 - 12 - 19 - 27 - 41  +  8"


# ---------------------------------------------------------------- generation

@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_produces_a_legal_ticket(strategy, sample):
    rng = np.random.default_rng(1)
    ticket = STRATEGIES[strategy](sample[1], rng)
    assert isinstance(ticket, Ticket)
    assert len(ticket.main_set) == MAIN_BALLS_DRAWN


def test_generation_is_reproducible_from_a_seed():
    a = random_ticket(np.random.default_rng(7))
    b = random_ticket(np.random.default_rng(7))
    assert a == b


def test_hot_and_cold_lean_opposite_ways():
    """A rigged history: 1-5 drawn constantly, so hot must favour them and cold avoid them."""
    balls = pd.DataFrame([[1, 2, 3, 4, 5, 9]] * 300)
    rng = np.random.default_rng(0)
    hot = [hot_ticket(balls, rng, strength=6.0) for _ in range(30)]
    cold = [cold_ticket(balls, rng, strength=6.0) for _ in range(30)]

    hot_share = np.mean([len(t.main_set & {1, 2, 3, 4, 5}) for t in hot])
    cold_share = np.mean([len(t.main_set & {1, 2, 3, 4, 5}) for t in cold])
    assert hot_share > cold_share


def test_ticket_from_predictions_fills_collisions():
    """Models often predict the same number in several slots — the ticket still needs five."""
    predictions = {0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 3}
    ticket = ticket_from_predictions(predictions, np.random.default_rng(0))
    assert len(ticket.main_set) == MAIN_BALLS_DRAWN
    assert 7 in ticket.main_set
    assert ticket.super_ball == 3


def test_ticket_from_predictions_keeps_a_clean_prediction_intact():
    predictions = {0: 3, 1: 12, 2: 19, 3: 27, 4: 41, 5: 8}
    ticket = ticket_from_predictions(predictions, np.random.default_rng(0))
    assert ticket.main_set == frozenset({3, 12, 19, 27, 41})
    assert ticket.super_ball == 8


def test_disjoint_portfolio_never_repeats_a_number():
    tickets = generate_portfolio(8, rng=np.random.default_rng(0), disjoint=True)
    coverage = portfolio_coverage(tickets)
    assert coverage["distinct_main_numbers"] == 8 * MAIN_BALLS_DRAWN == 40


def test_portfolio_falls_back_when_the_pool_runs_out(sample):
    """Only 8 disjoint tickets fit in 43 numbers; the 9th must still be generated."""
    tickets = generate_portfolio(10, strategy="hot", balls_expanded=sample[1],
                                 rng=np.random.default_rng(0), disjoint=True)
    assert len(tickets) == 10


def test_portfolio_needs_history_for_a_history_dependent_strategy():
    with pytest.raises(ValueError, match="needs balls_expanded"):
        generate_portfolio(3, strategy="hot", disjoint=False)


def test_portfolio_coverage_is_exact():
    tickets = generate_portfolio(4, rng=np.random.default_rng(2), disjoint=True)
    coverage = portfolio_coverage(tickets)
    assert coverage["distinct_main_numbers"] == 20
    assert coverage["pool_coverage_pct"] == pytest.approx(100 * 20 / 43)
    assert coverage["expected_total_main_matches"] == pytest.approx(20 * 5 / 43)
    assert coverage["jackpot_probability"] == pytest.approx(4 / total_combinations())


# ------------------------------------------------------------------ checking

def test_draw_from_row_splits_mains_from_the_superbalota():
    main, super_ball = draw_from_row([3, 12, 19, 27, 41, 8])
    assert main == frozenset({3, 12, 19, 27, 41})
    assert super_ball == 8


def test_check_ticket_counts_matches():
    ticket = Ticket(main=(3, 12, 19, 27, 41), super_ball=8)
    result = check_ticket(ticket, frozenset({3, 12, 40, 41, 42}), 8)
    assert result["main_matches"] == 3
    assert result["super_match"] is True
    assert result["category"] == "3 + superbalota"
    assert result["matched_numbers"] == [3, 12, 41]


def test_check_ticket_is_order_agnostic():
    ticket = Ticket(main=(41, 27, 19, 12, 3), super_ball=8)
    assert check_ticket(ticket, frozenset({3, 12, 19, 27, 41}), 8)["main_matches"] == 5


def test_check_against_history_covers_every_draw(sample):
    df, balls = sample
    results = check_against_history(Ticket(main=(1, 2, 3, 4, 5), super_ball=1), df, balls)
    assert len(results) == len(df)
    assert list(results["ds"]) == list(df["ds"])

    summary = history_summary(results)
    assert summary["times"].sum() == len(df)
    assert summary["share_pct"].sum() == pytest.approx(100.0)


# --------------------------------------------------------------- measurement

def test_evaluate_strategy_rejects_an_unknown_strategy(sample):
    with pytest.raises(ValueError, match="Unknown strategy"):
        evaluate_strategy("lucky", *sample)


def test_evaluate_strategy_refuses_when_there_is_no_room(sample):
    with pytest.raises(ValueError, match="No draws to evaluate"):
        evaluate_strategy("random", *sample, min_history=len(sample[0]) + 10)


def test_evaluate_strategy_only_uses_the_past(sample):
    """Deterministic under a seed, and the ticket count matches the window."""
    result = evaluate_strategy("random", *sample, n_draws_back=40, seed=0, min_history=50)
    again = evaluate_strategy("random", *sample, n_draws_back=40, seed=0, min_history=50)
    assert result["n_tickets_evaluated"] == 40
    assert result == again


def test_random_strategy_does_not_beat_chance(sample):
    result = evaluate_strategy("random", *sample, n_draws_back=100, seed=0)
    assert result["beats_chance"] is False
    assert result["avg_main_matches"] == pytest.approx(result["chance_avg_main_matches"], abs=0.4)


def test_stability_check_flag_rate_is_near_alpha_for_random(sample):
    """`random` cannot have an edge, so its flag rate is the measured false-positive floor."""
    result = stability_check(*sample, strategy="random", n_seeds=20, n_draws_back=60)
    assert result["n_seeds"] == 20
    assert result["flag_rate"] <= 0.25, "a signal-free strategy should rarely flag"
    assert 0 <= result["median_p_value"] <= 1


def test_compare_strategies_reports_a_corrected_verdict(sample):
    table = compare_strategies(*sample, n_draws_back=60, seed=0)
    assert {"beats_chance", "beats_chance_corrected", "bonferroni_threshold"} <= set(table.columns)
    assert len(table) == len(STRATEGIES)
    assert (table["bonferroni_threshold"] == 0.05 / len(STRATEGIES)).all()


def test_the_corrected_verdict_is_never_more_permissive(sample):
    """Bonferroni can only ever be stricter — if it flags, the naive test flagged too."""
    table = compare_strategies(*sample, n_draws_back=60, seed=0)
    assert not (table["beats_chance_corrected"] & ~table["beats_chance"]).any()


def test_compare_strategies_threshold_tracks_the_number_of_strategies(sample):
    table = compare_strategies(*sample, strategies=["random", "hot"], n_draws_back=60, seed=0)
    assert (table["bonferroni_threshold"] == 0.05 / 2).all()
