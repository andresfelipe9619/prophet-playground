"""One strength per kind of day — cycling/plackett_luce.py's TerrainPlackettLuce.

The model the sport obviously wants, tested as a claim rather than assumed to
work. Three things are pinned.

**The target race's terrain is an argument, never an inference.** That is
`features.py`'s rule and this class is where it would be broken, so the forecast
takes a terrain and the fit takes labels only from history.

**A terrain without enough races falls back to the unconditional fit**, and
`available` says which ones did not. A silent fallback would leave the model
looking conditional everywhere while being unconditional half the time.

**Shrinkage is toward the rider's own overall strength, not toward the field.**
Seven mountain stages is thin, and shrinking toward 1 would make a conditional
fit a noisier copy of the unconditional one rather than a refinement of it.

The measurement is in `test_the_conditional_fit_finds_the_specialists`, which is
`sensitivity.py`'s move in another domain: plant specialists, then check the
machinery finds them. Without the planted world, "terrain did not help" would be
a statement about the detector as much as about the sport.
"""

import numpy as np
import pytest

from cycling.features import CLIMB, SPRINT
from cycling.plackett_luce import PlackettLuce, TerrainPlackettLuce
from cycling.sample_data import load_sample_and_preprocess


def _log_worths(model, riders, terrain=None):
    return np.log(model.worths_for(riders, terrain)
                  if isinstance(model, TerrainPlackettLuce) else model.worths_for(riders))


def _truth(results, riders, column):
    table = results.drop_duplicates("rider").set_index("rider")
    return table.loc[riders, column].to_numpy(dtype=float)


def test_each_terrain_that_has_enough_races_gets_its_own_fit():
    results = load_sample_and_preprocess(seed=0)
    model = TerrainPlackettLuce.fit(results)
    # Every third stage is a mountain stage: 7 climbs and 14 sprints.
    assert model.available == (CLIMB, SPRINT)
    assert model.n_races == {CLIMB: 7, SPRINT: 14}
    assert model.worths_for(["Rider 001"], CLIMB)[0] != model.worths_for(["Rider 001"], SPRINT)[0]


def test_a_thin_terrain_falls_back_to_the_unconditional_fit_and_says_so():
    results = load_sample_and_preprocess(seed=0)
    early = results[results["stage"] <= 5]  # one mountain stage only
    model = TerrainPlackettLuce.fit(early)

    assert CLIMB not in model.available
    riders = sorted(set(early["rider"]))[:20]
    assert model.worths_for(riders, CLIMB) == pytest.approx(model.worths_for(riders))
    # The count is still reported, so the fallback is readable rather than inferred
    # from a model that quietly looks the same as another one.
    assert model.n_races[CLIMB] == 1


def test_an_unknown_terrain_is_the_unconditional_fit_rather_than_a_refusal():
    # A race whose terrain nobody supplied is an ordinary case — the roadbook
    # covers what it covers — and the honest answer is the model that does not
    # condition, not an exception.
    results = load_sample_and_preprocess(seed=0)
    model = TerrainPlackettLuce.fit(results)
    riders = sorted(set(results["rider"]))[:20]
    assert model.worths_for(riders, None) == pytest.approx(model.worths_for(riders, "cobbles"))


def test_the_conditional_fit_finds_the_specialists_that_were_planted():
    # The control this comparison needs. With `specialisation=2.0` a rider's
    # mountain strength and flat strength are different numbers, and the
    # conditional fit tracks the mountain truth far better than the
    # unconditional one — which, fitted mostly on the days where ability shows,
    # is measurably *anti*-correlated with flat ability (-0.26 measured).
    results = load_sample_and_preprocess(seed=0, specialisation=2.0)
    riders = sorted(set(results["rider"]))
    climbing_truth = _truth(results, riders, "climb_true")

    unconditional = PlackettLuce.fit(results)
    conditional = TerrainPlackettLuce.fit(results)

    plain = np.corrcoef(_log_worths(unconditional, riders), climbing_truth)[0, 1]
    by_terrain = np.corrcoef(_log_worths(conditional, riders, CLIMB), climbing_truth)[0, 1]
    assert by_terrain > plain + 0.1


def test_shrinkage_is_toward_the_riders_own_strength_not_the_field():
    # The difference between a refinement and a noisier copy, and it is visible
    # in one contrast: shrink the mountain stages hard toward the rider's
    # overall worth and the fit keeps the field's spread; shrink them equally
    # hard toward the field average and the spread is gone. A conditional model
    # built the second way would be seven stages of noise around 1.
    results = load_sample_and_preprocess(seed=0)
    climbs = results[results["stage"] % 3 == 0]
    riders = sorted(set(climbs["rider"]))

    overall = PlackettLuce.fit(results)
    toward_rider = PlackettLuce.fit(climbs, prior_strength=5000.0, prior_worths=overall.worths)
    toward_field = PlackettLuce.fit(climbs, prior_strength=5000.0)

    overall_worths = overall.worths_for(riders)
    assert toward_rider.worths_for(riders) == pytest.approx(overall_worths, rel=0.05)
    assert np.std(toward_rider.worths_for(riders)) > 0.3
    assert np.std(toward_field.worths_for(riders)) < 0.01


def test_a_rider_with_no_races_of_that_terrain_keeps_their_overall_strength():
    # The fallback that makes a conditional fit usable on a Grand Tour: a rider
    # who has not yet ridden a mountain stage is not reset to the field, they
    # are still whoever the flat stages said they were.
    results = load_sample_and_preprocess(seed=0)
    climbs = results[results["stage"] % 3 == 0]
    newcomer = "Neo Pro"
    prior = {**PlackettLuce.fit(results).worths, newcomer: 3.0}

    model = PlackettLuce.fit(climbs, prior_worths=prior)
    # Not in the climbing frame at all, so the fit never saw them and the worth
    # they arrived with is the worth they leave with.
    assert model.worth(newcomer) == pytest.approx(3.0, rel=0.05)


def test_the_unconditional_fit_is_untouched_by_the_prior_machinery():
    # `prior_worths=None` must reproduce the old fit exactly: the terrain work
    # generalised `_fit_mm`, and a silent change there would move every cycling
    # number in the project.
    results = load_sample_and_preprocess(seed=0)
    riders = sorted(set(results["rider"]))
    flat_prior = dict.fromkeys(riders, 1.0)
    assert (PlackettLuce.fit(results).worths_for(riders)
            == pytest.approx(PlackettLuce.fit(results, prior_worths=flat_prior).worths_for(riders)))
