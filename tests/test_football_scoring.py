"""Proper scoring rules for the three-way football outcome — football/scoring.py.

The outcome is ORDERED (H < D < A as a rank of "how much the home side won by"),
so the headline metric is RPS, which charges less for a near miss than for a
far one. Brier is symmetric and does not. Every function here takes probs in
OUTCOMES order; getting that order wrong silently corrupts RPS.
"""

import numpy as np
import pytest

from football.scoring import (
    METRICS,
    brier_score,
    log_loss,
    per_match_scores,
    ranked_probability_score,
    skill_score,
)


def test_perfect_forecast_scores_zero_on_every_metric():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = ["H", "D", "A"]
    assert brier_score(probs, outcomes) == pytest.approx(0.0)
    assert ranked_probability_score(probs, outcomes) == pytest.approx(0.0)
    assert log_loss(probs, outcomes) == pytest.approx(0.0)


def test_uniform_forecast_has_known_brier_and_rps():
    probs = np.full((3, 3), 1 / 3)
    outcomes = ["H", "D", "A"]
    # Brier per match: (1-1/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 0.6667
    assert brier_score(probs, outcomes) == pytest.approx(2 / 3)
    # RPS for a uniform forecast: H gets 5/18, D gets 1/9, A gets 5/18.
    # Mean: (5/18 + 1/9 + 5/18) / 3 = (5/18 + 2/18 + 5/18) / 3 = 2/9.
    assert ranked_probability_score(probs, outcomes) == pytest.approx(2 / 9)


def test_rps_rewards_the_near_miss_but_brier_does_not():
    # Away win occurred. Forecast X put all mass on Draw (adjacent);
    # forecast Y put all mass on Home win (far). RPS should prefer X; Brier ties.
    near = np.array([[0.0, 1.0, 0.0]])
    far = np.array([[1.0, 0.0, 0.0]])
    outcomes = ["A"]
    assert ranked_probability_score(near, outcomes) < ranked_probability_score(far, outcomes)
    assert brier_score(near, outcomes) == pytest.approx(brier_score(far, outcomes))


def test_per_match_scores_mean_equals_the_aggregate():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=20)
    outcomes = rng.choice(["H", "D", "A"], size=20)
    for metric in METRICS:
        agg = {"brier": brier_score, "rps": ranked_probability_score, "log_loss": log_loss}[metric]
        assert per_match_scores(probs, outcomes, metric).mean() == pytest.approx(agg(probs, outcomes))


def test_skill_score_positive_when_model_beats_baseline():
    outcomes = ["H", "H", "H", "H"]
    good = np.full((4, 3), 0.0) + np.array([0.9, 0.05, 0.05])
    weak = np.full((4, 3), 1 / 3)
    assert skill_score(good, weak, outcomes, metric="rps") > 0
    assert skill_score(weak, good, outcomes, metric="rps") < 0


def test_skill_score_drops_rows_where_either_side_is_nan():
    outcomes = ["H", "D", "A"]
    model = np.array([[0.8, 0.1, 0.1], [np.nan, np.nan, np.nan], [0.2, 0.3, 0.5]])
    market = np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3], [np.nan, np.nan, np.nan]])
    # Only row 0 survives in both; the call must not raise and must score 1 match.
    value = skill_score(model, market, outcomes, metric="brier")
    assert np.isfinite(value)


def test_single_forecast_metric_raises_on_nan():
    with pytest.raises(ValueError):
        brier_score(np.array([[np.nan, np.nan, np.nan]]), ["H"])
