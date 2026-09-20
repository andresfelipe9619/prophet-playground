"""What edge could this many matches have revealed? — football/power.py

The arithmetic half of a null result. These pin the shape (more matches find
smaller edges, the two directions agree with each other) and, more importantly,
that the spread it all turns on is **measured rather than assumed** — which is
the one thing that differs from the lottery's exact hypergeometric case.
"""

import numpy as np
import pytest

from football.common import OUTCOMES
from football.power import (
    REFERENCE_SCORE_SD,
    achieved_power,
    describe,
    matches_to_seasons,
    minimum_detectable_edge,
    observed_score_sd,
    power_curve,
    required_matches,
    required_matches_table,
)


def _forecasts(n=400, seed=0, spread=0.1):
    """A market vector and a model that departs from it by a known amount."""
    rng = np.random.default_rng(seed)
    market = rng.dirichlet([4.0, 3.0, 3.0], size=n)
    model = market * np.exp(rng.normal(0.0, spread, market.shape))
    model = model / model.sum(axis=1, keepdims=True)
    outcomes = [OUTCOMES[rng.choice(3, p=row)] for row in market]
    return model, market, outcomes


# ------------------------------------------------------- the measured spread


def test_the_spread_is_measured_from_a_run_rather_than_assumed():
    """The whole reason this module differs from the lottery's: no closed form."""
    model, market, outcomes = _forecasts(spread=0.1)
    tight = observed_score_sd(model, market, outcomes)

    model, market, outcomes = _forecasts(spread=0.5)
    loose = observed_score_sd(model, market, outcomes)

    assert loose > tight, "a model that departs further from the price must disagree more"


def test_a_forecast_that_is_the_market_has_no_spread_at_all():
    """The endpoint: identical forecasts produce an identically zero difference."""
    _, market, outcomes = _forecasts()
    assert observed_score_sd(market, market, outcomes) == pytest.approx(0.0, abs=1e-12)


def test_matches_with_no_price_are_dropped_rather_than_counted():
    model, market, outcomes = _forecasts(n=100)
    blinded = market.copy()
    blinded[:20] = np.nan

    assert np.isfinite(observed_score_sd(model, blinded, outcomes))


def test_too_few_matches_gives_nan_rather_than_a_number():
    model, market, outcomes = _forecasts(n=1)
    assert np.isnan(observed_score_sd(model, market, outcomes))


# ------------------------------------------------------------ the arithmetic


def test_more_matches_detect_smaller_edges():
    small = minimum_detectable_edge(400)["absolute"]
    large = minimum_detectable_edge(4000)["absolute"]
    assert large < small


def test_the_edge_falls_with_the_square_root_of_the_match_count():
    """Which is why 'collect more data' stops being an answer so quickly."""
    ten_x = minimum_detectable_edge(4000)["absolute"]
    one_x = minimum_detectable_edge(400)["absolute"]
    assert ten_x == pytest.approx(one_x / np.sqrt(10), rel=1e-6)


def test_a_wider_spread_needs_more_matches():
    assert required_matches(0.005, score_sd=0.16) > required_matches(0.005, score_sd=0.08)


def test_required_matches_scales_with_the_square_of_the_spread():
    """So using the wrong end of the measured range is a factor-of-ten error."""
    doubled = required_matches(0.005, score_sd=0.16)
    base = required_matches(0.005, score_sd=0.08)
    assert doubled == pytest.approx(4 * base, rel=0.01)


def test_the_two_directions_agree():
    """required_matches(mde(n)) should land back on n."""
    n = 1200
    edge = minimum_detectable_edge(n)["absolute"]
    assert required_matches(edge) == pytest.approx(n, rel=0.01)


def test_power_at_the_minimum_detectable_edge_is_the_requested_power():
    n = 800
    edge = minimum_detectable_edge(n, power=0.8)["absolute"]
    assert achieved_power(n, edge) == pytest.approx(0.8, abs=0.01)


def test_a_season_of_one_league_cannot_see_a_realistic_edge():
    """The headline. 380 matches bottoms out around a hundredth of an RPS, and
    a good model takes a few thousandths out of a closing line."""
    mde = minimum_detectable_edge(380)["absolute"]
    assert mde > 0.005, "a season should not be able to resolve a realistic edge"
    assert required_matches(0.002) > 5000


def test_a_nonsense_spread_is_refused_rather_than_returning_infinity():
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="score_sd"):
            minimum_detectable_edge(400, score_sd=bad)
        with pytest.raises(ValueError, match="score_sd"):
            required_matches(0.005, score_sd=bad)


def test_an_impossible_request_is_refused():
    with pytest.raises(ValueError, match="n_matches"):
        minimum_detectable_edge(0)
    with pytest.raises(ValueError, match="edge must be positive"):
        required_matches(0)


# -------------------------------------------------------------- the surfaces


def test_the_table_reports_seasons_because_matches_mean_nothing_alone():
    table = required_matches_table()
    # DEFAULT_EDGES runs large to small, so the match counts run small to large.
    assert list(table["edge"]) == sorted(table["edge"], reverse=True)
    assert list(table["required_matches"]) == sorted(table["required_matches"])
    assert (table["seasons_of_one_league"] > 0).all()
    assert matches_to_seasons(380) == pytest.approx(1.0)


def test_the_power_curve_rises_with_the_edge():
    curve = power_curve(1000)
    assert list(curve["power"]) == sorted(curve["power"], reverse=True)
    assert (curve["power"].between(0.0, 1.0)).all()


def test_describe_says_whether_the_spread_was_measured_or_assumed():
    """A reader who cannot tell those apart cannot tell a resolution from a guess."""
    assert "reference spread" in describe(380)
    assert "measured on this run" in describe(380, score_sd=0.11)
    assert f"{REFERENCE_SCORE_SD:.4f}" in describe(380)
