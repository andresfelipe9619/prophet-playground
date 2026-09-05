"""Synthetic stage races — cycling/sample_data.py.

The generator's job is to look enough like a real Grand Tour that a test passing
on it means something: bunch finishes with identical times, riders who abandon
and never reappear, and ability worth minutes in the mountains and nothing in a
sprint. It also carries the answer — `ability_true` — which is what lets a test
check that a forecast which *is* the truth beats the ranking baseline.

Every check here goes through the real parser in `cycling/processor.py`, so the
contract is exercised rather than bypassed.
"""

import pandas as pd
import pytest

from cycling.common import DNF, FINISHED, GC, STAGE
from cycling.processor import preprocess_results, time_order_violations
from cycling.sample_data import (
    generate_stage_race,
    general_classification,
    load_sample_and_preprocess,
    load_sample_gc,
    rider_abilities,
)

SMALL = {"n_stages": 6, "n_riders": 40}


def test_the_same_seed_gives_the_same_race():
    first = generate_stage_race(seed=7, **SMALL)
    second = generate_stage_race(seed=7, **SMALL)
    pd.testing.assert_frame_equal(first, second)
    assert not generate_stage_race(seed=8, **SMALL).equals(first)


def test_it_goes_through_the_real_contract():
    results = load_sample_and_preprocess(**SMALL)
    assert results.attrs["result_kind"] == STAGE
    assert not time_order_violations(results).any()


def test_a_bunch_finish_produces_identical_times_with_distinct_ranks():
    results = load_sample_and_preprocess(**SMALL)
    stage_one = results[results["stage"] == 1]
    assert stage_one["time_seconds"].duplicated().any()
    assert stage_one["rank"].is_unique


def test_turning_bunch_finishes_off_gives_every_rider_their_own_time():
    results = preprocess_results(
        generate_stage_race(bunch_finishes=False, **SMALL), validate=False)
    stage_one = results[results["stage"] == 1]
    assert not stage_one["time_seconds"].duplicated().any()


def test_riders_abandon_and_do_not_come_back():
    results = load_sample_and_preprocess(n_stages=21, n_riders=176, seed=3,
                                        abandon_hazard=0.02)
    abandons = results[results["status"] == DNF]
    assert len(abandons) > 0

    for rider, when in abandons.groupby("rider")["stage"].min().items():
        later = results[(results["rider"] == rider) & (results["stage"] > when)]
        assert later.empty, f"{rider} reappears after abandoning on stage {when}"


def test_an_abandon_carries_no_rank_and_no_time():
    results = load_sample_and_preprocess(n_stages=21, n_riders=176, seed=3,
                                        abandon_hazard=0.05)
    abandons = results[results["status"] == DNF]
    assert abandons["rank"].isna().all()
    assert abandons["time_seconds"].isna().all()


def test_ability_is_worth_more_in_the_mountains_than_in_a_sprint():
    # If it were not, every stage would be equally informative and a model that
    # ignored the course would look as good as one that did not.
    results = load_sample_and_preprocess(n_stages=6, n_riders=60, seed=2,
                                        climbing_stages={3})

    def front_of_race_spread(stage):
        """Seconds between the winner and 20th place — the gaps a GC is won on.

        Measured over the front of the race rather than the whole field, because
        the tail is riders who were dropped on both kinds of stage and its spread
        says nothing about how much ability was worth on the day.
        """
        day = results[(results["stage"] == stage) & (results["rank"] <= 20)]
        return day["time_seconds"].max() - day["time_seconds"].min()

    # A flat stage's front arrives together on one time; a mountain stage splits
    # it by minutes, which is where a general classification is actually decided.
    assert front_of_race_spread(1) == 0
    assert front_of_race_spread(3) > 120


def test_the_general_classification_ranks_on_accumulated_time():
    gc = load_sample_gc(**SMALL)
    assert gc.attrs["result_kind"] == GC
    assert gc["rank"].is_monotonic_increasing
    assert gc["time_seconds"].is_monotonic_increasing


def test_only_riders_who_finished_every_stage_are_classified():
    raw = generate_stage_race(n_stages=21, n_riders=176, seed=3, abandon_hazard=0.05)
    stages = preprocess_results(raw, validate=False)
    gc = preprocess_results(general_classification(raw), validate=False)

    abandoned = set(stages.loc[stages["status"] == DNF, "rider"])
    assert abandoned
    assert not abandoned & set(gc["rider"])


def test_the_truth_predicts_the_classification():
    # The one check the synthetic data exists for: a forecast that *is* the
    # generative ability must order the GC nearly perfectly. On real results
    # nobody knows the answer, so this can only be checked here.
    gc = load_sample_gc(n_stages=21, n_riders=100, seed=5)
    ability = rider_abilities(n_riders=100, seed=5).set_index("rider")["ability_true"]
    correlation = gc["rank"].corr(gc["rider"].map(ability), method="spearman")
    assert correlation < -0.8    # stronger ability, lower rank number


def test_the_answer_key_is_attached_only_by_the_sample_loader():
    assert "ability_true" in load_sample_and_preprocess(**SMALL).columns
    # Nothing that loads real results has it, so nothing outside tests can read it.
    assert "ability_true" not in preprocess_results(
        generate_stage_race(**SMALL), validate=False).columns


def test_the_generated_file_shape_is_the_on_disk_contract(tmp_path):
    path = str(tmp_path / "synthetic.csv")
    generate_stage_race(**SMALL).to_csv(path, index=False)
    from cycling.processor import load_and_preprocess
    results = load_and_preprocess(path, validate=False)
    assert (results["status"] == FINISHED).sum() > 0
