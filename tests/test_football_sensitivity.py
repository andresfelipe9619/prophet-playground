"""Can the market test find an edge that is really there? — football/sensitivity.py

The empirical half of a null result. `power.py` says what the test should be
able to see; this runs it and checks.

The controls carry the file. **Strength 0 is a forecast that *is* the market**,
so its detection rate is the false-positive floor and must sit near alpha — and
a control firing well above alpha means the harness is wrong before it means
the test is, which this module learned the hard way and records in its
docstring.
"""

import numpy as np
import pytest

from football.power import observed_score_sd
from football.scoring import per_match_scores
from football.sensitivity import (
    DEFAULT_MARKET_NOISE,
    _season,
    detection_rate,
    independent_seeds,
    planted_forecast,
    sensitivity_report,
    sensitivity_threshold,
)

pytestmark = pytest.mark.slow


# ------------------------------------------------------------ the answer key


def test_the_truth_lines_up_with_the_match_it_belongs_to():
    """The bug this module shipped once: the truth joined by row order rather
    than by match key, so the 'omniscient' forecast knew about other fixtures.

    At market_noise=0 the de-margined price IS the truth, so any gap beyond the
    generator's two-decimal rounding of its own odds is a misalignment.
    """
    truth, market, outcomes = _season(7, 16, 0.0)
    assert len(truth) == len(market) == len(outcomes)
    assert np.abs(truth - market).max() < 0.01


def test_a_forecast_at_strength_zero_is_exactly_the_market():
    """The load-bearing endpoint, the same one ensemble.py pins at weight 0."""
    truth, market, _ = _season(3, 12, 0.4)
    np.testing.assert_allclose(planted_forecast(truth, market, 0.0), market, atol=1e-12)


def test_a_forecast_at_strength_one_is_the_truth():
    truth, market, _ = _season(3, 12, 0.4)
    np.testing.assert_allclose(planted_forecast(truth, market, 1.0), truth, atol=1e-12)


def test_a_planted_forecast_is_always_a_probability_vector():
    truth, market, _ = _season(3, 12, 0.4)
    for strength in (0.0, 0.3, 1.0):
        out = planted_forecast(truth, market, strength)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
        assert (out >= 0).all()


def test_independent_seeds_do_not_share_a_stream():
    """The lottery module's 17.5% false-positive rate came from skipping this."""
    a, b = independent_seeds(0)
    assert a != b
    left = np.random.default_rng(a).normal(size=50)
    right = np.random.default_rng(b).normal(size=50)
    assert not np.allclose(left, right)


# ----------------------------------------------------------- the control


def test_the_control_sits_near_alpha():
    """A forecast carrying no information the price lacks must clear the bar at
    the false-positive rate and no more."""
    result = detection_rate(0.0, n_seasons=40, seed=5)
    assert result["detection_rate_corrected"] <= 0.20, (
        "a forecast that IS the market beat the market too often — suspect the harness")
    assert result["mean_effect"] == pytest.approx(0.0, abs=1e-9)


def test_a_perfect_market_leaves_nothing_to_plant():
    """Not a failure: the domain's own statement that nothing beats a perfect
    price. At market_noise=0 every strength collapses onto the control, which
    is why the default is a book that is good but beatable."""
    informed = detection_rate(1.0, n_seasons=25, market_noise=0.0, seed=9)
    assert informed["mean_effect"] == pytest.approx(0.0, abs=1e-3)
    assert informed["detection_rate_corrected"] <= 0.25


# ------------------------------------------------------- the positive control


def test_a_planted_edge_against_a_beatable_book_is_found():
    """The point of the module: the instrument can hear something."""
    result = detection_rate(0.05, n_seasons=30, seed=11)
    assert result["detection_rate_corrected"] >= 0.60
    assert result["mean_effect"] > 0


def test_the_effect_rises_with_the_planted_strength():
    """Monotone, unlike the detection rate — see the next test."""
    truth, market, outcomes = _season(21, 16, DEFAULT_MARKET_NOISE)
    effects = []
    for strength in (0.05, 0.25, 1.0):
        forecast = planted_forecast(truth, market, strength)
        effects.append(float((per_match_scores(market, outcomes, "rps")
                              - per_match_scores(forecast, outcomes, "rps")).mean()))
    assert effects == sorted(effects)


def test_a_louder_forecast_is_harder_to_prove_right():
    """The finding worth keeping: the spread grows faster than the effect, so
    the statistic shrinks as the model departs further from the price. It is
    ensemble.py's argument for pooling, arriving from another direction."""
    truth, market, outcomes = _season(21, 16, DEFAULT_MARKET_NOISE)
    quiet = planted_forecast(truth, market, 0.05)
    loud = planted_forecast(truth, market, 1.0)

    assert observed_score_sd(loud, market, outcomes) > observed_score_sd(quiet, market, outcomes)

    def z(forecast):
        difference = (per_match_scores(market, outcomes, "rps")
                      - per_match_scores(forecast, outcomes, "rps"))
        return difference.mean() / (difference.std(ddof=1) / np.sqrt(len(difference)))

    assert z(loud) < z(quiet)


# --------------------------------------------------------------- the report


def test_the_report_carries_the_control_row_and_both_verdicts():
    report = sensitivity_report(strengths=(0.0, 0.25), n_seasons=15, seed=3)
    assert set(report["strength"]) == {0.0, 0.25}
    assert {"detection_rate", "detection_rate_corrected"} <= set(report.columns)
    control = report[report["strength"] == 0.0].iloc[0]
    assert control["detection_rate_corrected"] <= 0.25


def test_the_threshold_is_nan_when_nothing_clears_it():
    """A real answer: this much football cannot reliably find any of these."""
    report = sensitivity_report(strengths=(0.0,), n_seasons=10, seed=4)
    assert np.isnan(sensitivity_threshold(report, target_rate=0.8))
