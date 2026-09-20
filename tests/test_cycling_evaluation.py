"""Does a cycling forecast beat the pre-race ranking? — cycling/evaluation.py
and cycling/plackett_luce.py.

The controls are the same shape as football's, and for the same reason. The
positive one uses a race where ability decides the order and hands a forecaster
the generator's own `ability_true`: if knowing the truth does not beat the
ranking, the harness is wrong and nothing else here means anything. The negative
one is a forecaster that *is* the baseline, which must come out at exactly zero.

`ability_true` is the generator's answer key and nothing outside tests may read
it; this is a test.
"""

import numpy as np
import pytest

from cycling.baseline import form_worths, uniform_worths, worths_from_rating
from cycling.evaluation import beats_baseline_test, compare_forecasters, walk_forward
from cycling.plackett_luce import PlackettLuce
from cycling.processor import preprocess_results
from cycling.sample_data import generate_stage_race, load_sample_and_preprocess, rider_abilities


def _mountain_race(seed=0, n_stages=14):
    """Every stage a climbing stage, so ability actually decides the order.

    The default synthetic Grand Tour makes two stages in three a sprint, where
    ability is worth 4 seconds against 45 of noise — the finishing order there
    is very nearly a coin toss, and no forecast can or should beat a ranking on
    it. A positive control has to run where the signal is.
    """
    raw = generate_stage_race(n_stages=n_stages, seed=seed,
                              climbing_stages=set(range(1, n_stages + 1)))
    return preprocess_results(raw, validate=False)


def _ranking(history, riders, as_of):
    return form_worths(history, riders, as_of=as_of)


def test_a_forecaster_identical_to_the_baseline_scores_exactly_zero():
    # The null the whole comparison sits on. Any drift here and a "beats the
    # ranking" verdict stops meaning anything.
    results = load_sample_and_preprocess(seed=0)
    table, _ = compare_forecasters(
        results, {"ranking": _ranking, "copia": _ranking}, baseline="ranking", min_history=3)
    row = table.iloc[0]
    assert row["effect"] == pytest.approx(0.0, abs=1e-12)
    assert row["model_score"] == pytest.approx(row["baseline_score"])
    assert bool(row["beats_baseline_corrected"]) is False


def test_knowing_the_truth_beats_the_ranking():
    """The positive control: the answer key has to win where signal exists."""
    results = _mountain_race()
    truth = dict(zip(*rider_abilities(seed=0)[["rider", "ability_true"]].to_numpy().T, strict=True))

    def oracle(history, riders, as_of):
        return worths_from_rating(
            np.array([float(truth[rider]) for rider in riders]), scale=0.6)

    table, _ = compare_forecasters(
        results, {"ranking": _ranking, "oracle": oracle}, baseline="ranking", min_history=2)
    row = table.iloc[0]
    assert row["model_score"] < row["baseline_score"]
    assert row["skill_score"] > 0
    assert bool(row["beats_baseline_corrected"]) is True


def test_a_uniform_draw_loses_to_the_ranking():
    # The claim the baseline module's docstring makes, measured: a uniform draw
    # over the start list is not a baseline, it is worse than one.
    results = _mountain_race()
    table, _ = compare_forecasters(
        results, {"ranking": _ranking, "uniform": lambda h, r, a: uniform_worths(len(r))},
        baseline="ranking", min_history=2)
    assert table.iloc[0]["effect"] < 0


def test_the_threshold_is_divided_by_the_number_of_challengers():
    results = load_sample_and_preprocess(seed=0)
    table, _ = compare_forecasters(
        results,
        {"ranking": _ranking,
         "uniform": lambda h, r, a: uniform_worths(len(r)),
         "copia": _ranking},
        baseline="ranking", min_history=3)
    assert len(table) == 2
    assert table["bonferroni_threshold"].iloc[0] == pytest.approx(0.05 / 2)


def test_a_forecaster_never_sees_the_race_it_forecasts():
    """The leakage check, done by looking at what the forecaster was handed.

    `walk_forward` passes each forecaster the results strictly before the race.
    A spy records the maximum date it ever saw, which must stay below the date
    of the race it was asked about.
    """
    results = load_sample_and_preprocess(seed=0)
    seen = []

    def spy(history, riders, as_of):
        seen.append((history["ds"].max() if len(history) else None, as_of))
        return uniform_worths(len(riders))

    walk_forward(results, {"spy": spy}, min_history=2)
    assert seen
    for latest_seen, as_of in seen:
        assert latest_seen is None or latest_seen < as_of


def test_a_forecaster_that_fails_on_one_race_drops_out_of_that_race_only():
    results = load_sample_and_preprocess(seed=0)
    calls = {"n": 0}

    def flaky(history, riders, as_of):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("no forecast for this one")
        return uniform_worths(len(riders))

    scores = walk_forward(results, {"flaky": flaky}, min_history=2)
    assert scores["score"].isna().sum() == 1
    assert scores["score"].notna().sum() > 1


def test_the_baseline_has_to_be_one_of_the_forecasters():
    results = load_sample_and_preprocess(seed=0)
    with pytest.raises(ValueError, match="not among the forecasters"):
        compare_forecasters(results, {"ranking": _ranking}, baseline="mercado")


def test_an_empty_comparison_is_a_refusal_not_a_verdict():
    result = beats_baseline_test([np.nan, np.nan], [np.nan, 1.0])
    assert result["n_races"] == 0
    assert result["beats_baseline"] is False


# ------------------------------------------------------------- the fitted model

def test_the_fitted_strengths_recover_the_generators_truth():
    """A model estimating rider strength has to find the strength that made the data."""
    results = _mountain_race()
    truth = rider_abilities(seed=0)
    ability = dict(zip(truth["rider"], truth["ability_true"], strict=True))

    model = PlackettLuce.fit(results)
    riders = sorted(results["rider"].unique())
    fitted = np.log(model.worths_for(riders))
    actual = np.array([ability[rider] for rider in riders])
    assert np.corrcoef(fitted, actual)[0, 1] > 0.7


def test_the_worths_are_normalised_and_positive():
    model = PlackettLuce.fit(load_sample_and_preprocess(seed=0))
    worths = np.array(list(model.worths.values()))
    assert worths.mean() == pytest.approx(1.0)
    assert (worths > 0).all() and np.isfinite(worths).all()


def test_an_unseen_rider_gets_the_field_average_rather_than_a_refusal():
    # Unlike football, where a fixture with an unknown team cannot be predicted
    # at all, a 180-rider start list with a neo-pro in it is an ordinary day.
    model = PlackettLuce.fit(load_sample_and_preprocess(seed=0))
    assert model.worth("Neo Pro") == pytest.approx(1.0)


def test_the_pure_maximum_likelihood_fit_stays_finite():
    # Without shrinkage a rider nobody finished behind has an unbounded worth
    # and one who never finished ahead has a worth of zero, which would take a
    # logarithm to minus infinity in every score computed from it.
    model = PlackettLuce.fit(load_sample_and_preprocess(seed=0), prior_strength=0.0)
    worths = np.array(list(model.worths.values()))
    assert np.isfinite(worths).all() and (worths > 0).all()


def test_shrinkage_pulls_a_thin_fit_toward_the_field():
    results = load_sample_and_preprocess(seed=0)
    thin = results[results["stage"] <= 2]
    loose = np.array(list(PlackettLuce.fit(thin, prior_strength=0.5).worths.values()))
    tight = np.array(list(PlackettLuce.fit(thin, prior_strength=50.0).worths.values()))
    assert tight.std() < loose.std()


def test_row_order_does_not_change_the_fit():
    results = load_sample_and_preprocess(seed=0)
    shuffled = results.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a = PlackettLuce.fit(results)
    b = PlackettLuce.fit(shuffled)
    for rider in a.riders:
        assert a.worth(rider) == pytest.approx(b.worth(rider), rel=1e-6)
