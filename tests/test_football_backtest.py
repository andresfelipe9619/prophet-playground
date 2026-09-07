"""Walk-forward: does Dixon-Coles beat the market? — football/backtest.py.

Mirrors lottery/backtest.py. The two checks that matter are a positive and a
negative control: on a SOFT simulated market (market_noise high) a real model
should beat it; on a SHARP one (market_noise = 0, the book prices the truth)
it should not. A backtest that fires on the sharp market is measuring its own
wiring, not an edge.

The fixture is a 16-team synthetic league with a generous training slice
(``min_train=160`` of ~240 matches). Dixon-Coles fitted on a 12-team single
round robin is a noisy read of the truth and lands too close to even a soft
market for the positive control to be anything but flaky; a larger league and
more training data give the model enough signal to separate the two markets
reliably at ``seed=0``.

The positive control deliberately uses a very soft, unrealistic book
(``market_noise=1.3``, far beyond a real opening line). The assertion is only
that a real model beats a demonstrably soft market, not a realistic one.
"""

import pytest

from football.backtest import run_all, run_holdout
from football.sample_data import generate_matches
from football.processor import preprocess_matches

pytestmark = pytest.mark.slow


def _matches(market_noise, seed=0, n_teams=16):
    raw = generate_matches(n_teams=n_teams, seed=seed, market_noise=market_noise).drop(
        columns=["TrueH", "TrueD", "TrueA"])
    return preprocess_matches(raw, validate=False)


def test_beats_a_soft_market():
    result = run_all(_matches(market_noise=1.3), n_windows=120, min_train=160)
    assert result["skill_score"] > 0
    assert result["beats_market"] is True


def test_does_not_beat_a_sharp_market():
    result = run_all(_matches(market_noise=0.0), n_windows=120, min_train=160)
    assert result["beats_market_corrected"] is False


def test_frozen_and_expanding_have_the_same_keys():
    matches = _matches(market_noise=0.3)
    cutoff = matches["ds"].quantile(0.7)
    frozen = run_holdout(matches, cutoff=cutoff, mode="frozen")
    expanding = run_holdout(matches, cutoff=cutoff, mode="expanding")
    assert set(frozen) == set(expanding)
    assert frozen["mode"] == "frozen" and expanding["mode"] == "expanding"


def test_effect_size_and_interval_are_reported():
    result = run_all(_matches(market_noise=0.8), n_windows=60, min_train=160)
    assert "effect" in result and "ci_low" in result and "ci_high" in result
    assert result["n_windows_scored"] > 0
