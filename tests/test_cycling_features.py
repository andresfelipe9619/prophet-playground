"""Terrain, specialisation, team and fatigue — cycling/features.py.

The tests that carry weight here are about **where a number came from**, not
about its value.

Terrain is inferred from how a race finished, which is a fact about a result.
That is fine for history and is leakage for the race being predicted, and the
leak would be invisible — a label read off tomorrow's result looks exactly like
a label read off a roadbook. So `terrain_of` refuses a race at or after `as_of`,
and `test_terrain_cannot_be_read_off_the_race_being_predicted` is the test that
would notice that refusal being softened.

Every other feature goes through one `_history` call, and the `<` in it is the
same `<` `baseline.form_worths` depends on.
"""

import numpy as np
import pandas as pd
import pytest

from cycling.features import (
    CLIMB,
    SPRINT,
    UNKNOWN,
    bunch_share,
    feature_frame,
    infer_terrain,
    placing_percentile,
    race_days,
    specialisation,
    team_strength,
    terrain_form,
    terrain_history,
    terrain_of,
)
from cycling.sample_data import load_sample_and_preprocess


@pytest.fixture(scope="module")
def results():
    return load_sample_and_preprocess(seed=0)


def test_terrain_is_read_off_the_bunch_and_the_two_classes_do_not_overlap(results):
    # The generator puts every third stage in the mountains. Measured, the two
    # classes sit at ~0.80 and ~0.01 shares on the winner's time — the threshold
    # at 0.5 is not a tuned number, it is the middle of a gap.
    labels = infer_terrain(results).set_index("stage")
    climbing = labels[labels.index % 3 == 0]
    flat = labels[labels.index % 3 != 0]
    assert set(climbing["terrain"]) == {CLIMB}
    assert set(flat["terrain"]) == {SPRINT}
    assert climbing["bunch_share"].max() < 0.2 < flat["bunch_share"].min()


def test_a_race_without_times_is_unknown_rather_than_guessed(results):
    blank = results.copy()
    blank["time_seconds"] = np.nan
    assert set(infer_terrain(blank)["terrain"]) == {UNKNOWN}


def test_a_field_too_small_to_read_gives_no_share(results):
    tiny = results[results["stage"] == 1].head(6)
    assert np.isnan(bunch_share(tiny))


def test_terrain_cannot_be_read_off_the_race_being_predicted(results):
    # The whole shape of this module: a label for the target race must come from
    # the roadbook. This is the assertion that fails if the `>=` softens.
    target = results[results["stage"] == 9]
    as_of = target["ds"].min()

    assert terrain_of(results, "synthetic-grand-tour", "stage", 6, as_of=as_of) == CLIMB
    with pytest.raises(ValueError, match="read off its own result"):
        terrain_of(results, "synthetic-grand-tour", "stage", 9, as_of=as_of)


def test_terrain_history_keeps_only_that_kind_of_day_and_only_the_past(results):
    as_of = results[results["stage"] == 10]["ds"].min()
    climbs = terrain_history(results, CLIMB, as_of=as_of)
    assert set(climbs["stage"]) == {3, 6, 9}
    # `terrain=None` is the unconditional history, so a caller needs no second
    # code path to go back to the unconditional model.
    assert set(terrain_history(results, None, as_of=as_of)["stage"]) == set(range(1, 10))


def test_placing_percentile_is_comparable_across_field_sizes():
    frame = pd.DataFrame({
        "ds": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 2),
        "race": ["a"] * 3 + ["b"] * 2, "kind": ["one_day"] * 5, "stage": [np.nan] * 5,
        "rank": [1.0, 2.0, np.nan, 1.0, 2.0], "rider": list("abcab"),
        "team": ["t"] * 5, "status": ["FIN", "FIN", "DNF", "FIN", "FIN"],
        "time_seconds": [1.0, 2.0, np.nan, 1.0, 2.0],
    })
    scored = placing_percentile(frame)
    assert list(scored["placing_percentile"]) == [1.0, 0.5, 0.0, 1.0, 0.0]


def test_features_see_nothing_at_or_after_as_of(results):
    # One `_history` call sits under all of them, so one test covers the lot:
    # a feature built at stage 10 must be identical to one built from the frame
    # truncated before stage 10 by hand.
    target = results[results["stage"] == 10]
    as_of = target["ds"].min()
    riders = list(target["rider"])[:20]
    before = results[results["ds"] < as_of]

    for built in (terrain_form, race_days, team_strength, specialisation):
        pd.testing.assert_frame_equal(
            built(results, riders, as_of=as_of), built(before, riders, as_of=as_of))


def test_team_strength_excludes_the_rider_it_describes(results):
    # Otherwise it is the rider's own form wearing a team jersey, and it enters
    # a model twice.
    target = results[results["stage"] == 12]
    as_of = target["ds"].min()
    teams = dict(zip(target["rider"], target["team"], strict=True))
    riders = list(target["rider"])[:8]

    strengths = team_strength(results, riders, as_of=as_of, teams=teams).set_index("rider")
    history = placing_percentile(results[results["ds"] < as_of])
    for rider in riders:
        team = teams[rider]
        others = history[(history["team"] == team) & (history["rider"] != rider)]
        assert strengths.loc[rider, "team_strength"] == pytest.approx(
            others["placing_percentile"].mean())
        assert strengths.loc[rider, "n_team_races"] == len(others)


def test_fatigue_counts_race_days_not_calendar_days(results):
    target = results[results["stage"] == 15]
    as_of = target["ds"].min()
    riders = list(target["rider"])[:5]

    days = race_days(results, riders, as_of=as_of, window_days=7).set_index("rider")
    # Seven calendar days, and a rider still in the race has ridden all of them.
    assert set(days["race_days"]) <= {7}
    assert days["days_since_last"].max() <= 7


def test_an_abandoned_rider_stops_accumulating_race_days(results):
    quit_early = (results[results["status"] != "FIN"]
                  .sort_values("ds").iloc[0]["rider"])
    last_seen = results[results["rider"] == quit_early]["ds"].max()
    end = results["ds"].max() + pd.Timedelta(days=1)

    days = race_days(results, [quit_early], as_of=end, window_days=3650).set_index("rider")
    assert days.loc[quit_early, "days_since_last"] == pytest.approx((end - last_seen).days)


def test_feature_frame_records_the_terrain_it_was_told(results):
    target = results[results["stage"] == 12]
    as_of = target["ds"].min()
    frame = feature_frame(results, list(target["rider"])[:10], as_of=as_of, terrain=CLIMB)
    assert frame.attrs["terrain"] == CLIMB
    assert frame.attrs["as_of"] == as_of
    assert frame["n_terrain_races"].max() == 3  # stages 3, 6 and 9
