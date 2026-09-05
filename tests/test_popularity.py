"""Jackpot splitting — lottery/analysis/popularity.py.

This is the only module in the project that improves anything, and the tests
below are mostly about bounding *what*: it moves `E[payout | win]` and must
never appear to move `P(win)`. The score is ordinal, so nothing here asserts
an absolute level — only orderings, which is all the model claims.
"""

import numpy as np
import pytest

from lottery.analysis.popularity import (
    BIAS_WEIGHTS,
    compare_tickets,
    expected_winners,
    jackpot_probability,
    popularity_components,
    popularity_score,
    split_adjusted_value,
    unpopular_ticket,
)
from lottery.analysis.prizes import category_probabilities, total_combinations
from lottery.analysis.tickets import STRATEGIES, Ticket
from lottery.models.common import MAIN_BALLS_DRAWN

# Every documented bias at once: all under 31, all low, consecutive, evenly
# spaced, all in one block of ten.
VERY_POPULAR = Ticket(main=(1, 2, 3, 4, 5), super_ball=7)
# None of them: all above the calendar ceiling, ragged spacing, spread out.
VERY_UNPOPULAR = Ticket(main=(32, 36, 39, 41, 43), super_ball=12)

JACKPOT = 5_000_000_000
TICKETS_SOLD = 3_000_000


def test_the_score_stays_in_range():
    for ticket in (VERY_POPULAR, VERY_UNPOPULAR):
        assert 0.0 <= popularity_score(ticket) <= 1.0


def test_a_birthday_ticket_scores_above_a_high_spread_one():
    assert popularity_score(VERY_POPULAR) > popularity_score(VERY_UNPOPULAR)


def test_the_calendar_bias_is_the_one_that_dominates():
    """Players use dates; 1-31 are picked far more often than 32-43."""
    assert BIAS_WEIGHTS["calendar"] == max(BIAS_WEIGHTS.values())

    under = Ticket(main=(2, 9, 17, 24, 31), super_ball=5)
    over = Ticket(main=(2, 9, 17, 24, 43), super_ball=5)
    assert popularity_components(under)["calendar"] == 1.0
    assert popularity_components(over)["calendar"] == pytest.approx(0.8)
    assert popularity_score(under) > popularity_score(over)


def test_every_component_is_a_fraction():
    for ticket in (VERY_POPULAR, VERY_UNPOPULAR):
        for name, value in popularity_components(ticket).items():
            assert 0.0 <= value <= 1.0, name


def test_the_components_detect_what_they_claim_to():
    consecutive = popularity_components(Ticket(main=(7, 8, 9, 10, 11), super_ball=1))
    assert consecutive["consecutive"] == 1.0
    assert consecutive["arithmetic"] == pytest.approx(1.0)

    ragged = popularity_components(Ticket(main=(1, 2, 3, 4, 43), super_ball=1))
    assert ragged["arithmetic"] < 0.5

    spread = popularity_components(Ticket(main=(3, 14, 25, 36, 43), super_ball=1))
    assert spread["round_decade"] < 0.5
    assert spread["consecutive"] == 0.0


def test_the_score_is_ordinal_not_absolute():
    """Scaling every weight cannot change any ordering — the score is a ranking."""
    doubled = {name: weight * 2 for name, weight in BIAS_WEIGHTS.items()}
    assert popularity_score(VERY_POPULAR, weights=doubled) == pytest.approx(
        popularity_score(VERY_POPULAR))


def test_a_bare_sequence_scores_the_same_as_a_ticket():
    assert popularity_score((1, 2, 3, 4, 5)) == pytest.approx(popularity_score(VERY_POPULAR))


# --------------------------------------------------------------- splitting

def test_expected_winners_scales_with_tickets_sold():
    one = expected_winners(VERY_POPULAR, 1_000_000)
    ten = expected_winners(VERY_POPULAR, 10_000_000)
    assert ten == pytest.approx(one * 10)


def test_no_spread_collapses_to_the_uniform_baseline():
    """With a multiplier of 1 every combination is equally played, whatever it scores."""
    baseline = TICKETS_SOLD / total_combinations()
    for ticket in (VERY_POPULAR, VERY_UNPOPULAR):
        assert expected_winners(ticket, TICKETS_SOLD,
                                popularity_multiplier=1.0) == pytest.approx(baseline)


def test_a_popular_ticket_expects_more_co_winners():
    assert (expected_winners(VERY_POPULAR, TICKETS_SOLD)
            > expected_winners(VERY_UNPOPULAR, TICKETS_SOLD))


def test_the_unpopular_ticket_is_worth_more_if_it_wins():
    popular = split_adjusted_value(VERY_POPULAR, JACKPOT, TICKETS_SOLD)
    unpopular = split_adjusted_value(VERY_UNPOPULAR, JACKPOT, TICKETS_SOLD)
    assert unpopular["expected_jackpot_share"] > popular["expected_jackpot_share"]


def test_the_band_brackets_the_point_estimate():
    """The multiplier is the least defensible input, so a single figure would hide it."""
    value = split_adjusted_value(VERY_UNPOPULAR, JACKPOT, TICKETS_SOLD)
    assert value["share_low"] <= value["expected_jackpot_share"] <= value["share_high"]
    assert value["share_low"] < value["share_high"], "a band, not a point"


def test_the_share_never_exceeds_the_jackpot():
    for ticket in (VERY_POPULAR, VERY_UNPOPULAR):
        value = split_adjusted_value(ticket, JACKPOT, TICKETS_SOLD)
        assert 0 < value["expected_jackpot_share"] <= JACKPOT
        assert 0 < value["fraction_of_jackpot"] <= 1.0


def test_choosing_numbers_cannot_change_the_probability_of_winning():
    """The claim the module is careful never to make, pinned as a test."""
    table = category_probabilities()
    exact = float(table.loc[(table["main_matches"] == MAIN_BALLS_DRAWN)
                            & table["super_match"], "probability"].iloc[0])
    assert jackpot_probability() == pytest.approx(exact)
    assert jackpot_probability() == pytest.approx(1 / total_combinations())


def test_compare_tickets_ranks_by_what_a_win_is_worth():
    table = compare_tickets([VERY_POPULAR, VERY_UNPOPULAR], JACKPOT, TICKETS_SOLD)
    assert len(table) == 2
    assert list(table["expected_jackpot_share"]) == sorted(
        table["expected_jackpot_share"], reverse=True)
    assert table.iloc[0]["popularity_score"] < table.iloc[1]["popularity_score"]
    assert set(BIAS_WEIGHTS) <= set(table.columns), "components are shown, not just the total"


# ------------------------------------------------------- the strategy hook

def test_unpopular_is_registered_as_a_strategy():
    """It is held to the same standard as the others, and correctly fails on hit rate."""
    assert "unpopular" in STRATEGIES


def test_unpopular_ticket_is_a_legal_ticket():
    ticket = unpopular_ticket(rng=np.random.default_rng(0))
    assert isinstance(ticket, Ticket)
    assert len(ticket.main_set) == MAIN_BALLS_DRAWN


def test_unpopular_ticket_beats_a_random_pick_on_popularity():
    from lottery.analysis.tickets import random_ticket

    rng = np.random.default_rng(1)
    chosen = np.mean([popularity_score(unpopular_ticket(rng=rng, candidates=40))
                      for _ in range(15)])
    baseline = np.mean([popularity_score(random_ticket(rng)) for _ in range(200)])
    assert chosen < baseline


def test_more_candidates_never_makes_the_pick_more_popular():
    few = popularity_score(unpopular_ticket(rng=np.random.default_rng(3), candidates=5))
    many = popularity_score(unpopular_ticket(rng=np.random.default_rng(3), candidates=200))
    assert many <= few


def test_unpopular_ticket_ignores_the_draw_history():
    """Popularity is about what other players choose, not what the machine drew."""
    a = unpopular_ticket(balls_expanded=None, rng=np.random.default_rng(5))
    b = unpopular_ticket(balls_expanded="ignored", rng=np.random.default_rng(5))
    assert a == b
