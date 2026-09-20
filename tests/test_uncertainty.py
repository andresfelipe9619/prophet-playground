"""How much of a forecast is the model and how much is the sample it saw.

`football/uncertainty.py` and `cycling/uncertainty.py`. The invariants worth
pinning are the ones that contradict the obvious expectation, because those are
what a later refactor will "fix" back into being wrong.
"""

import numpy as np
import pytest

from cycling.processor import preprocess_results
from cycling.sample_data import generate_stage_race
from cycling.uncertainty import appearances, bootstrap_worths, relative_band_width
from football.common import OUTCOMES
from football.dixon_coles import DixonColes
from football.processor import preprocess_matches
from football.sample_data import generate_matches
from football.uncertainty import (
    band_width,
    bootstrap_predictions,
    downgrade_uncertain_value,
)
from football.value import DISAGREEMENT_ONLY, NO_VALUE, VALUE

pytestmark = pytest.mark.slow


def _fit(matches, half_life):
    return DixonColes.fit(matches, half_life=half_life)


@pytest.fixture(scope="module")
def season():
    raw = generate_matches(n_teams=10, seed=2, market_noise=0.5).drop(
        columns=["TrueH", "TrueD", "TrueA"])
    return preprocess_matches(raw, validate=False)


# ===================================================================== football


def test_the_band_narrows_as_the_training_window_grows(season):
    """The one thing an uncertainty band must do."""
    fixture = [(season["home_team"].iloc[0], season["away_team"].iloc[0])]
    thin = bootstrap_predictions(season.head(30), fixture, _fit, n_resamples=60, seed=1)
    thick = bootstrap_predictions(season, fixture, _fit, n_resamples=60, seed=1)

    assert (thick["high"] - thick["low"]).mean() < (thin["high"] - thin["low"]).mean()


def test_a_non_finite_refit_does_not_erase_the_whole_band(season):
    """The first version of this let one degenerate resample turn every band to
    NaN while still reporting dozens of 'usable' draws."""
    fixture = [(season["home_team"].iloc[0], season["away_team"].iloc[0])]
    bands = bootstrap_predictions(season.head(25), fixture, _fit, n_resamples=60, seed=3)

    assert bands["n_usable"].min() > 0
    assert bands[["low", "high"]].notna().all().all()


def test_the_bound_count_falls_as_data_accumulates(season):
    """The measured case for a bootstrap over a Hessian: at small samples almost
    every refit ends against a parameter bound, where a curvature estimate would
    be describing the curvature of a wall."""
    fixture = [(season["home_team"].iloc[0], season["away_team"].iloc[0])]
    thin = bootstrap_predictions(season.head(25), fixture, _fit, n_resamples=60, seed=1)
    thick = bootstrap_predictions(season, fixture, _fit, n_resamples=60, seed=1)

    assert thin.attrs["n_at_bound"] > thick.attrs["n_at_bound"]


def test_the_band_brackets_something_and_names_its_fixture(season):
    fixture = [(season["home_team"].iloc[0], season["away_team"].iloc[0])]
    bands = bootstrap_predictions(season, fixture, _fit, n_resamples=40, seed=1)

    assert set(bands["outcome"]) == set(OUTCOMES)
    assert (bands["low"] <= bands["high"]).all()
    assert len(band_width(bands)) == 1


# ------------------------------------------------------- the value downgrade


def test_a_value_call_whose_band_crosses_the_price_is_demoted():
    """A point clearing 1/odds by a hair while its band straddles it is noise
    that landed with a favourable sign."""
    odds = np.array([2.0, 4.0, 5.0])          # break-even 0.50, 0.25, 0.20
    states = np.array([VALUE, VALUE, NO_VALUE], dtype=object)
    low = np.array([0.45, 0.30, 0.10])        # first straddles, second clears

    out = downgrade_uncertain_value(states, low, odds)
    assert list(out) == [DISAGREEMENT_ONLY, VALUE, NO_VALUE]


def test_nothing_is_ever_upgraded():
    """A band's top end clearing the price while the point does not is still a
    model without an edge on its own estimate."""
    odds = np.array([2.0])
    out = downgrade_uncertain_value(np.array([DISAGREEMENT_ONLY], dtype=object),
                                    np.array([0.9]), odds)
    assert list(out) == [DISAGREEMENT_ONLY]


def test_a_missing_band_demotes_rather_than_passing_through():
    """No band is not evidence of a safe bet."""
    out = downgrade_uncertain_value(np.array([VALUE], dtype=object),
                                    np.array([np.nan]), np.array([2.0]))
    assert list(out) == [DISAGREEMENT_ONLY]


# ====================================================================== cycling


@pytest.fixture(scope="module")
def races():
    return preprocess_results(
        generate_stage_race(n_riders=20, n_stages=16, seed=3,
                            climbing_stages=set(range(1, 17))), validate=False)


def test_a_riders_band_narrows_as_races_accumulate(races):
    """Absolute width, which is the one that behaves. See the next test."""
    riders = sorted(races["rider"].unique())
    thin = bootstrap_worths(races[races["stage"] <= 4], riders, n_resamples=60, seed=1)
    thick = bootstrap_worths(races, riders, n_resamples=60, seed=1)

    assert (thick["high"] - thick["low"]).median() < (thin["high"] - thin["low"]).median()


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_the_absolute_width_narrows_but_the_relative_one_does_not_track_evidence(seed):
    """Both halves of the warning `relative_band_width` carries.

    Measured across seeds: the absolute width falls every time, while the ratio
    to the worth rises, falls or stays flat depending on how the field
    separated. An earlier version of this test asserted the ratio *grows*, which
    is true of one seed and not of the others — reading a direction into it is
    the mistake the docstring now warns about.
    """
    full = preprocess_results(
        generate_stage_race(n_riders=20, n_stages=16, seed=seed,
                            climbing_stages=set(range(1, 17))), validate=False)
    riders = sorted(full["rider"].unique())
    thin = bootstrap_worths(full[full["stage"] <= 4], riders, n_resamples=50, seed=1)
    thick = bootstrap_worths(full, riders, n_resamples=50, seed=1)

    def absolute(bands):
        return (bands["high"] - bands["low"]).median()

    assert absolute(thick) < absolute(thin), "the absolute width must narrow with races"

    # No assertion on the direction of the ratio, deliberately: there is not one.
    ratios = [relative_band_width(b)["relative_width"].median() for b in (thin, thick)]
    assert all(np.isfinite(r) and r > 0 for r in ratios)


def test_a_point_outside_its_own_band_is_flagged_rather_than_corrected(races):
    """A percentile interval need not contain the point, and for a shrunk
    ratio-scale estimator it often does not. That is the fit saying it would
    not reproduce that number on a redrawn calendar — information, not a bug."""
    riders = sorted(races["rider"].unique())
    bands = bootstrap_worths(races, riders, n_resamples=60, seed=1)

    assert "contains_point" in bands.columns
    assert not bands["contains_point"].all(), "fixture no longer exercises the case"
    assert bands["contains_point"].any()


def test_every_requested_rider_gets_a_row_even_when_thinly_raced(races):
    riders = [*sorted(races["rider"].unique())[:3], "A Rider Who Never Started"]
    bands = bootstrap_worths(races, riders, n_resamples=20, seed=1)

    assert list(bands["rider"]) == riders
    assert bands[bands["rider"] == "A Rider Who Never Started"]["n_usable"].iloc[0] == 0


def test_appearances_explains_the_band(races):
    riders = sorted(races["rider"].unique())[:4]
    counts = appearances(races, riders)
    assert list(counts["rider"]) == riders
    assert (counts["n_races"] > 0).all()


def test_an_empty_calendar_is_refused_rather_than_returning_a_band(races):
    with pytest.raises(ValueError, match="No races"):
        bootstrap_worths(races.iloc[:0], ["someone"], n_resamples=5)
