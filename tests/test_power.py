"""Power analysis — lottery/analysis/power.py.

The module exists so a null result can carry its own resolution. These tests
check the arithmetic against the closed form rather than against itself, and
pin the two things a careless edit would break: the sqrt(N) scaling, and the
superbalota using a Bernoulli variance rather than the main-ball one.
"""

import math

import pytest
from scipy.stats import norm

from lottery.analysis.power import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    achieved_power,
    chance_moments,
    describe,
    draws_to_years,
    minimum_detectable_effect,
    power_curve,
    required_draws,
    required_draws_table,
    super_minimum_detectable_effect,
)
from lottery.models.baseline import expected_main_matches
from lottery.models.common import DRAW_WEEKDAYS, MAIN_BALLS_DRAWN, SUPER_POOL


def test_chance_moments_match_the_exact_hypergeometric():
    mean, sd = chance_moments()
    exact = expected_main_matches(MAIN_BALLS_DRAWN)
    assert mean == pytest.approx(exact["mean"])
    assert sd == pytest.approx(math.sqrt(exact["var"]))


def test_mde_matches_the_closed_form():
    """delta = (z_alpha + z_power) * sd / sqrt(N), computed independently here."""
    n = 500
    mean, sd = chance_moments()
    expected = (norm.ppf(1 - DEFAULT_ALPHA) + norm.ppf(DEFAULT_POWER)) * sd / math.sqrt(n)

    mde = minimum_detectable_effect(n)
    assert mde["absolute"] == pytest.approx(expected)
    assert mde["relative"] == pytest.approx(expected / mean)
    assert mde["detectable_mean"] == pytest.approx(mean + expected)
    assert mde["n_draws"] == n


def test_the_mde_only_falls_with_the_square_root_of_n():
    """Four times the data buys half the resolution — the reason no history is enough."""
    small = minimum_detectable_effect(250)["absolute"]
    large = minimum_detectable_effect(1000)["absolute"]
    assert large == pytest.approx(small / 2, rel=1e-9)


def test_more_data_never_makes_the_mde_worse():
    values = [minimum_detectable_effect(n)["relative"] for n in (100, 500, 1000, 5000)]
    assert values == sorted(values, reverse=True)


def test_required_draws_round_trips_with_the_mde():
    """Ask for the N that detects an edge, then confirm that N detects exactly it."""
    for edge in (0.05, 0.10, 0.25):
        n = required_draws(edge)
        assert minimum_detectable_effect(n)["relative"] <= edge + 1e-9
        assert minimum_detectable_effect(n - 1)["relative"] > edge


def test_achieved_power_hits_the_target_at_the_required_n():
    for edge in (0.10, 0.25, 0.50):
        assert achieved_power(required_draws(edge), edge) >= DEFAULT_POWER


def test_achieved_power_rises_with_data_and_with_effect_size():
    assert achieved_power(100, 0.25) < achieved_power(1000, 0.25)
    assert achieved_power(500, 0.05) < achieved_power(500, 0.50)


def test_a_zero_effect_has_power_equal_to_alpha():
    """With no real edge, the test fires exactly as often as its own false-positive rate."""
    assert achieved_power(1000, 0.0) == pytest.approx(DEFAULT_ALPHA)


@pytest.mark.parametrize("n_draws", [0, -5])
def test_mde_refuses_an_impossible_sample(n_draws):
    with pytest.raises(ValueError, match="at least 1"):
        minimum_detectable_effect(n_draws)
    with pytest.raises(ValueError, match="at least 1"):
        super_minimum_detectable_effect(n_draws)


@pytest.mark.parametrize("edge", [0.0, -0.1])
def test_required_draws_refuses_a_non_positive_edge(edge):
    with pytest.raises(ValueError, match="must be positive"):
        required_draws(edge)


# ------------------------------------------------------ the superbalota half

def test_the_superbalota_uses_a_bernoulli_variance():
    """One number out of the pool is a coin flip, not a hypergeometric draw."""
    p = 1 / SUPER_POOL
    result = super_minimum_detectable_effect(1000)
    assert result["chance_rate"] == pytest.approx(p)
    assert result["sd"] == pytest.approx(math.sqrt(p * (1 - p)))
    assert result["detectable_rate"] == pytest.approx(p + result["absolute"])


def test_the_two_helpers_are_not_interchangeable():
    """Reusing the main-ball figures for the superbalota understates the data needed.

    At the same N the superbalota's relative MDE is ~3.3x the main balls', which
    is why this is a separate function rather than a parameter.
    """
    n = 1000
    ratio = (super_minimum_detectable_effect(n)["relative"]
             / minimum_detectable_effect(n)["relative"])
    assert ratio == pytest.approx(3.3, abs=0.1)


# --------------------------------------------------------------- the tables

def test_required_draws_table_is_ordered_and_complete():
    table = required_draws_table(relative_edges=(0.50, 0.25, 0.10))
    assert list(table.columns) == ["relative_edge", "target_mean", "required_draws",
                                   "years_of_history"]
    assert list(table["required_draws"]) == sorted(table["required_draws"])
    mean, _ = chance_moments()
    assert table["target_mean"].iloc[0] == pytest.approx(mean * 1.5)


def test_smaller_edges_need_more_history_than_the_game_has_existed():
    """The row that makes the table land: some edges are simply unprovable."""
    table = required_draws_table()
    assert table["years_of_history"].max() > 50


def test_draws_to_years_uses_the_real_schedule():
    assert draws_to_years(len(DRAW_WEEKDAYS) * 52) == pytest.approx(1.0)
    assert draws_to_years(156, weekdays=(2, 5)) == pytest.approx(1.5)


def test_power_curve_is_monotonic_in_effect_size():
    curve = power_curve(500, relative_edges=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    assert list(curve["power"]) == sorted(curve["power"])
    assert curve["power"].iloc[-1] > 0.99


def test_describe_quotes_the_resolution_not_just_the_verdict():
    text = describe(15)
    assert "no edge above" in text
    assert "not 'no edge'" in text
