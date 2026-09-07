"""Does a football model beat the market? — football/evaluation.py.

The football analogue of lottery/models/baseline.py:beats_chance_test. It forms
the per-match difference between the market's proper score and the model's,
and tests whether that mean improvement is greater than zero — one-sided,
Bonferroni-aware, effect size attached.
"""

import numpy as np
import pytest

from football.evaluation import beats_market_test


def _outcomes(rng, n):
    return rng.choice(["H", "D", "A"], size=n)


def test_model_equal_to_market_shows_no_edge():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([3, 2, 2], size=300)
    outcomes = [["H", "D", "A"][i] for i in [np.argmax(p) for p in probs]]
    result = beats_market_test(probs, probs.copy(), outcomes)
    assert result["effect"] == pytest.approx(0.0, abs=1e-9)
    assert result["beats_market"] is False
    assert result["beats_market_corrected"] is False


def test_strictly_better_model_beats_the_market():
    rng = np.random.default_rng(1)
    n = 400
    outcomes = _outcomes(rng, n)
    truth = np.zeros((n, 3))
    truth[np.arange(n), [{"H": 0, "D": 1, "A": 2}[o] for o in outcomes]] = 1.0
    # Model: 80% toward the truth. Market: 55% toward the truth. Model must win.
    uniform = np.full((n, 3), 1 / 3)
    model = 0.8 * truth + 0.2 * uniform
    market = 0.55 * truth + 0.45 * uniform
    result = beats_market_test(model, market, outcomes, metric="rps")
    assert result["skill_score"] > 0
    assert result["p_value_greater"] < 0.01
    assert result["beats_market"] is True
    assert result["beats_market_corrected"] is True


def test_a_model_worse_than_the_market_does_not_beat_it():
    # One-sidedness is the invariant: a model strictly worse on every match must
    # land p_value_greater near 1.0, not a small two-sided p-value read as a win.
    rng = np.random.default_rng(7)
    n = 400
    outcomes = _outcomes(rng, n)
    truth = np.zeros((n, 3))
    truth[np.arange(n), [{"H": 0, "D": 1, "A": 2}[o] for o in outcomes]] = 1.0
    uniform = np.full((n, 3), 1 / 3)
    model = 0.30 * truth + 0.70 * uniform
    market = 0.60 * truth + 0.40 * uniform
    result = beats_market_test(model, market, outcomes, metric="rps")
    assert result["skill_score"] < 0
    assert result["p_value_greater"] > 0.99
    assert result["beats_market"] is False
    assert result["beats_market_corrected"] is False


def test_both_verdict_keys_always_present_even_on_empty_input():
    result = beats_market_test(np.empty((0, 3)), np.empty((0, 3)), [])
    assert "beats_market" in result and "beats_market_corrected" in result
    assert result["n_observations"] == 0


def test_more_comparisons_tighten_the_corrected_threshold():
    rng = np.random.default_rng(2)
    n = 300
    outcomes = _outcomes(rng, n)
    truth = np.zeros((n, 3))
    truth[np.arange(n), [{"H": 0, "D": 1, "A": 2}[o] for o in outcomes]] = 1.0
    uniform = np.full((n, 3), 1 / 3)
    model = 0.62 * truth + 0.38 * uniform
    market = 0.55 * truth + 0.45 * uniform
    one = beats_market_test(model, market, outcomes, n_comparisons=1)
    many = beats_market_test(model, market, outcomes, n_comparisons=10)
    assert many["bonferroni_threshold"] < one["bonferroni_threshold"]


def test_nan_rows_are_dropped_pairwise():
    model = np.array([[0.7, 0.2, 0.1], [np.nan, np.nan, np.nan], [0.3, 0.3, 0.4]])
    market = np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3], [np.nan, np.nan, np.nan]])
    result = beats_market_test(model, market, ["H", "D", "A"])
    assert result["n_observations"] == 1
