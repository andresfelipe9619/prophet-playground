"""Does 30% mean 30%, and does saying so out loud change anything?

The controls here are built the way `football/sample_data.py`'s are: from a
forecast whose calibration is known by construction, because on real data
nobody knows. A perfectly calibrated forecast is one whose outcomes were
*drawn from it* — that is what calibration means, and it makes the answer
checkable rather than assessable.

The over-confident arm is that same forecast squared and renormalised, which
has an exact known repair: a temperature of 2. A recalibrator that finds it is
doing what it claims; one that finds 1.4 is not.
"""

import numpy as np
import pytest

from football.calibration import (
    CALIBRATORS,
    IsotonicCalibrator,
    TemperatureScaler,
    calibration_in_the_large,
    expected_calibration_error,
    prequential_calibrate,
    reliability_curve,
)
from football.common import OUTCOMES
from football.scoring import ranked_probability_score

N = 3000


def _calibrated(seed=0, n=N):
    """A forecast and outcomes drawn from it — calibrated by construction."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet([4.0, 3.0, 3.0], size=n)
    outcomes = [OUTCOMES[rng.choice(3, p=row)] for row in probs]
    return probs, outcomes


def _sharpened(probs, power=2.0):
    """The same forecast stating its case too strongly, by a known amount.

    Squaring and renormalising is exactly what a temperature of 1/power does,
    so the repair has a right answer and the tests can demand it.
    """
    sharp = probs ** power
    return sharp / sharp.sum(axis=1, keepdims=True)


# ------------------------------------------------------------ reliability


def test_a_calibrated_forecast_sits_on_the_diagonal():
    """Within the noise each bin actually has, which is not a fixed tolerance.

    A flat 0.05 fails honestly here: the 0.8-0.9 bin holds nine forecasts, where
    one standard error is 0.13, so a gap of 0.07 is a bin doing exactly what a
    calibrated forecast does. Charging every bin the same tolerance regardless
    of how full it is would be the reliability-diagram version of reading a
    sparse bin as a measurement — the thing `min_count` exists to stop.
    """
    probs, outcomes = _calibrated()
    for row in reliability_curve(probs, outcomes):
        if np.isnan(row["observed_frequency"]):
            continue
        forecast, count = row["mean_forecast"], row["count"]
        standard_error = np.sqrt(forecast * (1.0 - forecast) / count)
        assert abs(forecast - row["observed_frequency"]) < 3 * standard_error


def test_an_over_confident_forecast_sits_off_it_in_a_known_direction():
    """High forecasts overshoot and low ones undershoot — that is what
    over-confidence *is*, and a curve that does not show it is not a curve."""
    probs, outcomes = _calibrated()
    curve = reliability_curve(_sharpened(probs), outcomes)
    usable = [r for r in curve if not np.isnan(r["observed_frequency"])]

    high = [r for r in usable if r["mean_forecast"] > 0.6]
    low = [r for r in usable if r["mean_forecast"] < 0.2]
    assert high and low, "the sharpened forecast produced no extreme bins to check"
    assert all(r["observed_frequency"] < r["mean_forecast"] for r in high)
    assert all(r["observed_frequency"] > r["mean_forecast"] for r in low)


def test_a_sparse_bin_reports_no_frequency_rather_than_a_wrong_one():
    """Three matches cannot measure a frequency. A count is honest; a point is not."""
    probs = np.array([[0.95, 0.03, 0.02]])
    curve = reliability_curve(probs, ["H"], bins=10, min_count=5)
    populated = [row for row in curve if row["count"]]
    assert populated, "the fixture put nothing in any bin"
    for row in populated:
        assert np.isnan(row["observed_frequency"])
        assert row["count"] > 0


def test_every_forecast_lands_in_exactly_one_bin():
    """Including 1.0, which numpy's own binning would otherwise put past the end."""
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.25, 0.25]])
    curve = reliability_curve(probs, ["H", "D", "H"], bins=10)
    assert sum(row["count"] for row in curve) == probs.size


def test_ece_is_near_zero_on_a_calibrated_forecast_and_large_on_a_sharpened_one():
    probs, outcomes = _calibrated()
    good = expected_calibration_error(probs, outcomes)
    bad = expected_calibration_error(_sharpened(probs), outcomes)

    assert good["ece"] < 0.02
    assert bad["ece"] > 5 * good["ece"]


def test_ece_reports_how_much_of_the_forecast_it_could_measure():
    """A small error over a tenth of the forecasts is not a small error."""
    probs, outcomes = _calibrated(n=40)
    report = expected_calibration_error(probs, outcomes)
    assert 0.0 <= report["coverage"] <= 1.0
    assert report["coverage"] < 1.0, "fixture was too large to leave any bin sparse"


def test_ece_is_nan_rather_than_zero_when_nothing_is_measurable():
    probs = np.array([[0.4, 0.3, 0.3]])
    report = expected_calibration_error(probs, ["H"])
    assert np.isnan(report["ece"])
    assert report["n_bins_used"] == 0


# -------------------------------------------------- calibration in the large


def test_a_calibrated_forecast_is_not_flagged_once_the_correction_is_applied():
    """Note what is *not* asserted: that every interval covers zero.

    On this fixture one of the three does not — a two-sided 95% interval misses
    by construction about one time in twenty, and three tests over one set of
    matches give three chances at it. That is not a defect in the fixture, it is
    the whole reason `miscalibrated_corrected` exists, and a test demanding all
    three intervals cover zero would be asserting the bug away.
    """
    for seed in (0, 1, 2):
        probs, outcomes = _calibrated(seed=seed)
        for row in calibration_in_the_large(probs, outcomes):
            assert row["miscalibrated_corrected"] is False
            assert row["n_observations"] == len(outcomes)


def test_a_forecast_shifted_toward_the_home_win_is_flagged():
    probs, outcomes = _calibrated()
    shifted = probs + np.array([0.10, -0.05, -0.05])
    shifted = np.clip(shifted, 1e-6, 1.0)
    shifted = shifted / shifted.sum(axis=1, keepdims=True)

    home = [row for row in calibration_in_the_large(shifted, outcomes)
            if row["outcome"] == "H"][0]
    assert home["miscalibrated_corrected"] is True
    assert home["difference"] < 0, "forecasting H too often should leave the base rate below it"


def test_both_verdicts_are_reported_and_the_correction_is_the_tighter_one():
    """Three tests over one set of matches is three chances at a false positive."""
    probs, outcomes = _calibrated()
    rows = calibration_in_the_large(probs, outcomes, alpha=0.05)
    for row in rows:
        assert {"miscalibrated", "miscalibrated_corrected", "bonferroni_threshold"} <= set(row)
        assert row["bonferroni_threshold"] == pytest.approx(0.05 / len(OUTCOMES))
        if row["miscalibrated_corrected"]:
            assert row["miscalibrated"], "corrected passed where naive did not"


def test_the_test_is_two_sided():
    """Under-forecasting is mis-calibration too, and a one-sided test waves it through."""
    probs, outcomes = _calibrated()
    under = probs - np.array([0.10, -0.05, -0.05])
    under = np.clip(under, 1e-6, 1.0)
    under = under / under.sum(axis=1, keepdims=True)

    home = [row for row in calibration_in_the_large(under, outcomes)
            if row["outcome"] == "H"][0]
    assert home["miscalibrated_corrected"] is True
    assert home["difference"] > 0


# ---------------------------------------------------------------- calibrators


def test_temperature_is_one_on_an_already_calibrated_forecast():
    probs, outcomes = _calibrated()
    assert TemperatureScaler.fit(probs, outcomes).temperature == pytest.approx(1.0, abs=0.08)


def test_temperature_recovers_the_amount_a_forecast_was_sharpened_by():
    """The exact-answer test. Squaring is a temperature of 2, so the fit must find 2."""
    probs, outcomes = _calibrated()
    fitted = TemperatureScaler.fit(_sharpened(probs, power=2.0), outcomes)
    assert fitted.temperature == pytest.approx(2.0, rel=0.1)


def test_temperature_one_is_exactly_the_identity():
    """The endpoint that makes any gain readable, the same role weight 0 plays
    in ensemble.py. If this drifts, 'recalibration helped' stops meaning anything."""
    probs, _ = _calibrated(n=50)
    np.testing.assert_allclose(TemperatureScaler(1.0).transform(probs), probs, atol=1e-12)


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_every_calibrator_returns_probability_vectors(name):
    probs, outcomes = _calibrated(n=600)
    out = CALIBRATORS[name].fit(probs, outcomes).transform(probs)
    assert out.shape == probs.shape
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert (out >= 0).all()


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_every_calibrator_improves_a_sharpened_forecast_when_fitted_on_it(name):
    """In-sample, which is exactly why this is not how the backtest uses them —
    see the prequential tests below. Here it only establishes that the repair
    works at all before asking whether it survives being honest."""
    probs, outcomes = _calibrated()
    sharp = _sharpened(probs)
    fixed = CALIBRATORS[name].fit(sharp, outcomes).transform(sharp)

    assert ranked_probability_score(fixed, outcomes) < ranked_probability_score(sharp, outcomes)


def test_isotonic_is_at_least_as_flexible_as_temperature_in_sample():
    """If the free map barely beats the one parameter, the model's problem is
    not its confidence — which is the reading isotonic exists to provide."""
    probs, outcomes = _calibrated()
    sharp = _sharpened(probs)
    iso = IsotonicCalibrator.fit(sharp, outcomes).transform(sharp)
    temp = TemperatureScaler.fit(sharp, outcomes).transform(sharp)

    assert ranked_probability_score(iso, outcomes) <= ranked_probability_score(temp, outcomes) + 1e-6


# --------------------------------------------------------------- the gate


def test_the_opening_forecasts_pass_through_untouched():
    """They have no past to be corrected from, and borrowing their own future
    is the leak this whole module is arranged around."""
    probs, outcomes = _calibrated(n=300)
    out, n_calibrated = prequential_calibrate(_sharpened(probs), outcomes, min_fit=100)

    np.testing.assert_allclose(out[:100], _sharpened(probs)[:100])
    assert n_calibrated == 200


def test_a_correction_is_fitted_only_on_what_came_before_it():
    """The load-bearing property, stated as an experiment: change the future and
    the past must not move. Nothing else here proves the gate is real."""
    probs, outcomes = _calibrated(n=400)
    sharp = _sharpened(probs)

    baseline, _ = prequential_calibrate(sharp, outcomes, min_fit=100)

    tampered = list(outcomes)
    tampered[300:] = ["H"] * (len(tampered) - 300)
    changed, _ = prequential_calibrate(sharp, tampered, min_fit=100)

    np.testing.assert_allclose(baseline[:300], changed[:300])
    assert not np.allclose(baseline[300:], changed[300:]), (
        "rewriting the last quarter's outcomes changed nothing — the correction "
        "is not being refitted at all")


def test_prequential_recalibration_still_helps_an_over_confident_forecast():
    """The honest version of the in-sample test above: fitted only on the past,
    scored only on the rows where it applied."""
    probs, outcomes = _calibrated()
    sharp = _sharpened(probs)
    fixed, n = prequential_calibrate(sharp, outcomes, min_fit=200)

    tail = slice(len(sharp) - n, None)
    assert (ranked_probability_score(fixed[tail], outcomes[tail])
            < ranked_probability_score(sharp[tail], outcomes[tail]))


def test_prequential_recalibration_barely_moves_a_calibrated_forecast():
    """Nothing to repair, so nothing much may happen. A recalibrator that
    'improves' a forecast which was already right is fitting noise."""
    probs, outcomes = _calibrated()
    fixed, n = prequential_calibrate(probs, outcomes, min_fit=200)

    tail = slice(len(probs) - n, None)
    before = ranked_probability_score(probs[tail], outcomes[tail])
    after = ranked_probability_score(fixed[tail], outcomes[tail])
    assert abs(after - before) < 0.01 * before


def test_refit_every_changes_runtime_and_not_the_guarantee():
    probs, outcomes = _calibrated(n=500)
    sharp = _sharpened(probs)
    often, _ = prequential_calibrate(sharp, outcomes, min_fit=100, refit_every=1)
    rarely, _ = prequential_calibrate(sharp, outcomes, min_fit=100, refit_every=25)

    np.testing.assert_allclose(often[:100], rarely[:100])
    # Both corrections are real, so both land nearer the truth than the input.
    for out in (often, rarely):
        assert (ranked_probability_score(out[100:], outcomes[100:])
                < ranked_probability_score(sharp[100:], outcomes[100:]))


def test_mismatched_lengths_raise_rather_than_silently_truncating():
    probs, outcomes = _calibrated(n=50)
    with pytest.raises(ValueError):
        prequential_calibrate(probs, outcomes[:-1])
