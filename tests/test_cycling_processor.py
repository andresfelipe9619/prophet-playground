"""The cycling data contract — cycling/processor.py.

Three invariants, each invisible in the shape of the frame:

**One kind of result per frame.** A rank in a stage result and a rank in a
general classification are different quantities. A frame holding both has a
`rank` column meaning two things — the cycling counterpart of mixing opening
and closing odds.

**Non-finishers stay.** They are not random: abandons concentrate among the
riders in worst form, so dropping them makes the remaining problem easier than
the real one and inflates everything measured afterwards.

**`time_seconds` is a total, never a gap.** A results page publishes the
winner's time and everyone else's gap to it; a column of gaps looks perfectly
normal, so the check hunts for the fingerprint — a rider timed faster than
someone placed ahead of them.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from cycling.common import DNF, FINISHED, GC, NR, ONE_DAY, RESULT_COLUMNS, STAGE
from cycling.processor import (
    ResultFormatError,
    check_result_format,
    load_and_preprocess,
    load_races,
    preprocess_results,
    time_order_violations,
)


def rows(**overrides):
    """Three finishers and one abandon from one stage — the smallest real result."""
    frame = pd.DataFrame({
        "Date": ["29/06/2024"] * 4,
        "Race": ["tour-de-france"] * 4,
        "Kind": [STAGE] * 4,
        "Stage": [1, 1, 1, 1],
        "Rank": [1, 2, 3, ""],
        "Rider": ["Rider A", "Rider B", "Rider C", "Rider D"],
        "Team": ["Team 1", "Team 2", "Team 3", "Team 4"],
        "Status": [FINISHED, FINISHED, FINISHED, DNF],
        "TimeSeconds": [15322.0, 15336.0, 15336.0, ""],
    })
    for column, values in overrides.items():
        frame[column] = values
    return frame


def gc_rows():
    return pd.DataFrame({
        "Date": ["21/07/2024"] * 2,
        "Race": ["tour-de-france"] * 2,
        "Kind": [GC] * 2,
        "Stage": [21, 21],
        "Rank": [1, 2],
        "Rider": ["Rider A", "Rider B"],
        "Team": ["Team 1", "Team 2"],
        "Status": [FINISHED] * 2,
        "TimeSeconds": [301150.0, 301176.0],
    })


def write(frame, tmp_path, name):
    path = str(tmp_path / name)
    frame.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------- the core

def test_the_required_columns_are_required():
    with pytest.raises(ResultFormatError, match="Missing required columns"):
        preprocess_results(pd.DataFrame({"Date": ["29/06/2024"], "Race": ["x"]}))


def test_the_tidy_shape_and_the_kind_are_produced():
    results = preprocess_results(rows(), validate=False)
    assert list(results.columns) == RESULT_COLUMNS
    assert results.attrs["result_kind"] == STAGE


def test_a_date_that_is_not_day_first_raises():
    with pytest.raises(ResultFormatError, match="day-first"):
        preprocess_results(rows(Date=["not a date"] * 4), validate=False)


def test_an_unknown_kind_or_status_raises():
    with pytest.raises(ResultFormatError, match="Unknown result kind"):
        preprocess_results(rows(Kind=["prologue"] * 4), validate=False)
    with pytest.raises(ResultFormatError, match="Unknown status"):
        preprocess_results(rows(Status=[FINISHED, FINISHED, FINISHED, "GONE"]), validate=False)


# ------------------------------------------------------- one kind per frame

def test_a_frame_mixing_stage_results_and_a_classification_raises():
    mixed = pd.concat([rows(), gc_rows()], ignore_index=True)
    with pytest.raises(ResultFormatError, match="mixes result kinds"):
        preprocess_results(mixed, validate=False)


def test_load_races_refuses_two_kinds_and_merges_one(tmp_path):
    stage_path = write(rows(), tmp_path, "stage.csv")
    gc_path = write(gc_rows(), tmp_path, "gc.csv")

    with pytest.raises(ResultFormatError, match="different result kinds"):
        load_races([stage_path, gc_path], validate=False)

    second = rows(Date=["30/06/2024"] * 4, Stage=[2] * 4)
    merged = load_races([stage_path, write(second, tmp_path, "stage2.csv")], validate=False)
    assert len(merged) == 8
    assert merged.attrs["result_kind"] == STAGE
    assert merged["ds"].is_monotonic_increasing


def test_a_one_day_race_carrying_a_stage_number_raises():
    with pytest.raises(ResultFormatError, match="no stage number"):
        preprocess_results(rows(Kind=[ONE_DAY] * 4), validate=False)


def test_a_classification_after_a_stage_may_carry_that_stage():
    results = preprocess_results(gc_rows(), validate=False)
    assert set(results["stage"]) == {21}


# ------------------------------------------------------- ranks and statuses

def test_a_rank_and_an_abandon_cannot_both_be_true():
    with pytest.raises(ResultFormatError, match="both a rank and a non-finishing"):
        preprocess_results(rows(Rank=[1, 2, 3, 4]), validate=False)


def test_a_finisher_without_a_rank_raises():
    with pytest.raises(ResultFormatError, match="finishers with no rank"):
        preprocess_results(rows(Rank=[1, 2, "", ""],
                                Status=[FINISHED] * 4), validate=False)


def test_two_riders_sharing_a_rank_is_a_parse_error_not_a_tie():
    with pytest.raises(ResultFormatError, match="share a rank"):
        preprocess_results(rows(Rank=[1, 2, 2, ""]), validate=False)


def test_the_same_rank_in_two_different_results_is_fine():
    two_stages = pd.concat([rows(), rows(Date=["30/06/2024"] * 4, Stage=[2] * 4)],
                           ignore_index=True)
    assert len(preprocess_results(two_stages, validate=False)) == 8


def test_a_missing_status_column_is_inferred_and_warned_about():
    with pytest.warns(UserWarning, match="No Status column"):
        results = preprocess_results(rows().drop(columns="Status"), validate=True)
    assert list(results["status"]) == [FINISHED, FINISHED, FINISHED, NR]


# ------------------------------------------------------------ non-finishers

def test_non_finishers_survive_loading(tmp_path):
    results = load_and_preprocess(write(rows(), tmp_path, "s.csv"), validate=False)
    assert (results["status"] == DNF).sum() == 1
    assert results.loc[results["status"] == DNF, "rank"].isna().all()


def test_dropping_them_is_an_explicit_choice_made_after_the_checks(tmp_path):
    path = write(rows(), tmp_path, "s.csv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = load_and_preprocess(path, validate=True, finishers_only=True)
    assert len(results) == 3
    assert results.attrs["result_kind"] == STAGE
    # The frame as published had an abandon, so the "no abandons" warning must
    # not fire just because the caller filtered it away afterwards.
    assert not any("probably filtered" in str(w.message) for w in caught)


def test_a_frame_with_no_abandons_at_all_is_flagged():
    clean = rows().iloc[:3]
    report = check_result_format(preprocess_results(clean, validate=False))
    assert "Not one non-finisher" in report["message"]
    assert report["n_non_finishers"] == 0


# -------------------------------------------------------- totals, not gaps

def test_gaps_stored_as_totals_are_detected():
    # What a broken scrape looks like: the winner's elapsed time, then everyone
    # else's gap sitting in the same column.
    as_gaps = rows(TimeSeconds=[15322.0, 14.0, 14.0, ""])
    results = preprocess_results(as_gaps, validate=False)
    violations = time_order_violations(results)
    assert violations.sum() == 2
    assert "total elapsed time" in check_result_format(results)["message"]


def test_resolved_totals_pass_and_equal_times_are_allowed():
    # A bunch finish gives the whole group the winner's time; ranks stay ordered.
    results = preprocess_results(rows(), validate=False)
    assert not time_order_violations(results).any()
    assert check_result_format(results) is None


def test_a_rider_with_no_time_is_reported_rather_than_filled_in():
    results = preprocess_results(rows(TimeSeconds=[15322.0, np.nan, 15336.0, ""]),
                                 validate=False)
    report = check_result_format(results)
    assert "1 of 3 ranked rider(s) have no time" in report["message"]


def test_a_one_second_tolerance_absorbs_rounding_not_a_gap():
    results = preprocess_results(rows(TimeSeconds=[15322.4, 15322.0, 15336.0, ""]),
                                 validate=False)
    assert not time_order_violations(results).any()


def test_validation_warns_rather_than_raising_on_weak_data():
    # The rows are real results; the caller may want them anyway. Only a broken
    # contract raises.
    with pytest.warns(UserWarning, match="total elapsed time"):
        preprocess_results(rows(TimeSeconds=[15322.0, 14.0, 14.0, ""]), validate=True)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_preprocess(str(tmp_path / "nope.csv"))


def test_an_empty_frame_is_reported_not_crashed_on():
    empty = preprocess_results(rows().iloc[:0], validate=False)
    assert check_result_format(empty)["n_results"] == 0
