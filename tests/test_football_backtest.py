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

from football.backtest import MODEL_NAMES, compare_models, run_all, run_holdout
from football.processor import preprocess_matches
from football.sample_data import generate_matches

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


# ---------------------------------------------------------- several models at once

def test_every_model_is_scored_on_the_same_matches():
    """The property that makes a multi-model table readable at all.

    A window is skipped when *any* requested model cannot predict it, so the
    rows compare like with like. Two models scored on different subsets of
    matches are not comparable, and nothing in the table's shape would say so.
    """
    table = compare_models(_matches(market_noise=0.8), n_windows=40, min_train=160)
    assert list(table["model"]) == list(MODEL_NAMES)
    assert table["n_windows_scored"].nunique() == 1
    assert table["n_windows_scored"].iloc[0] > 0
    # The market half of every paired comparison is the same matches, so every
    # row must see the identical market score.
    assert table["market_score"].nunique() == 1


def test_the_threshold_is_divided_by_how_many_models_ran():
    # Three models against one set of matches is three chances for luck to clear
    # an uncorrected 5%. A table reporting one naive verdict per model would have
    # reintroduced exactly the bug the lottery side is built around.
    table = compare_models(_matches(market_noise=0.8), n_windows=40, min_train=160)
    assert table["bonferroni_threshold"].iloc[0] == pytest.approx(0.05 / len(MODEL_NAMES))

    pair = compare_models(_matches(market_noise=0.8), n_windows=40, min_train=160,
                          models=("dixon_coles", "elo"))
    assert pair["bonferroni_threshold"].iloc[0] == pytest.approx(0.05 / 2)


def test_a_blend_at_weight_zero_is_the_market_itself():
    """The reading the whole blend rests on, checked rather than asserted in prose.

    At weight 0 the blend *is* the market, so the paired difference is exactly
    zero on every match: the same score, no effect, and no edge. Any improvement
    as the weight rises is therefore the model adding something the price did
    not already contain. If this drifts, that reading is gone.
    """
    table = compare_models(_matches(market_noise=1.3), n_windows=40, min_train=160,
                           models=("blend",), blend_weight=0.0)
    row = table.iloc[0]
    assert row["model_score"] == pytest.approx(row["market_score"])
    assert row["effect"] == pytest.approx(0.0, abs=1e-12)
    assert bool(row["beats_market_corrected"]) is False


def test_a_blend_beats_a_soft_market_when_the_model_does():
    # The positive control for the pooling path: against a demonstrably soft
    # book, putting weight on a real model has to help.
    table = compare_models(_matches(market_noise=1.3), n_windows=80, min_train=160,
                           models=("blend",), blend_weight=1.0)
    assert bool(table["beats_market_corrected"].iloc[0]) is True


def test_nothing_beats_a_market_that_prices_the_truth():
    # The negative control, for every model at once. A table that fires here is
    # measuring its own wiring.
    table = compare_models(_matches(market_noise=0.0), n_windows=80, min_train=160)
    assert not table["beats_market_corrected"].any()


def test_an_unknown_model_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown models"):
        compare_models(_matches(market_noise=0.5), n_windows=20, min_train=160,
                       models=("dixon_coles", "neural_net"))


# ----------------------------------------------------- the recalibration gate


def test_recalibration_scores_only_the_windows_where_it_applied():
    """The leading forecasts pass through uncalibrated, so scoring them would
    mix rows the correction reached with rows it could not and pull any
    difference toward zero. Every model is trimmed by the same amount, so the
    comparison stays on one identical set of matches."""
    matches = _matches(market_noise=0.9)
    plain = compare_models(matches, n_windows=120, min_train=160, models=("elo",))
    tuned = compare_models(matches, n_windows=120, min_train=160, models=("elo",),
                           calibrate="temperature", calibrate_min_fit=40)

    assert int(tuned["n_windows_scored"].iloc[0]) == int(plain["n_windows_scored"].iloc[0]) - 40
    assert tuned["calibrate"].iloc[0] == "temperature"
    assert plain["calibrate"].iloc[0] is None


def test_recalibration_keeps_every_model_on_the_same_matches():
    table = compare_models(_matches(market_noise=0.9), n_windows=120, min_train=160,
                           models=("dixon_coles", "elo"),
                           calibrate="temperature", calibrate_min_fit=40)
    assert table["n_windows_scored"].nunique() == 1


def test_the_manifest_records_that_the_run_was_recalibrated():
    """Two runs over the same matches that disagree are not comparable unless
    the thing that differed is written down."""
    table = compare_models(_matches(market_noise=0.9), n_windows=100, min_train=160,
                           models=("elo",), calibrate="temperature")
    assert table.attrs["manifest"]["inputs"]["calibrate"] == "temperature"
