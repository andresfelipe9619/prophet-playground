"""The forecast/actual comparison half of lottery/utils/processor.py.

`load_actual_2024_data`, `compare_numbers_by_date`,
`check_actual_in_past_predictions` and `process_and_compare_forecasts` are what
the three `scripts/*_forecast.py` entry points call after fitting, and pytest
never imports those scripts — so until these tests existed the whole path was
exercised only by running a forecast by hand.

**Read what these functions produce as a hindcast, not a result.** Counting how
many of tomorrow's numbers appeared in some earlier prediction is exactly the
sum the premise says cannot mean anything: with 5 numbers from 43 and enough
past predictions, three-number overlaps arrive on schedule by chance alone.
`lottery/backtest.py` is where a claim gets measured against the hypergeometric
baseline. These are file-writing plumbing, and the tests pin them as plumbing:
the merge is an inner join on the date, a match count is a set intersection over
the dash-separated numbers, and only *strictly earlier* predictions are
considered — the last of which is the one that would be a leak if it slipped.
"""

import pandas as pd
import pytest

from lottery.utils.processor import (
    check_actual_in_past_predictions,
    compare_numbers_by_date,
    load_actual_2024_data,
    process_and_compare_forecasts,
)


def actual(rows):
    """Actual draws in the shape check_actual_in_past_predictions expects."""
    return pd.DataFrame(
        [{"ds": pd.Timestamp(date), "numbers_actual": numbers} for date, numbers in rows]
    )


def predicted(rows):
    return pd.DataFrame(
        [{"ds": pd.Timestamp(date), "numbers_predicted": numbers} for date, numbers in rows]
    )


# --- load_actual_2024_data ----------------------------------------------------

def test_the_actual_results_file_is_read_day_first_and_renamed(tmp_path):
    path = tmp_path / "actual.csv"
    pd.DataFrame(
        [{"Date": "03/01/2024", "Ball Number": "3-12-19-27-41"}]
    ).to_csv(path, index=False)

    out = load_actual_2024_data(str(path))

    assert list(out.columns) == ["ds", "numbers"]
    # dd/mm/yyyy: the 3rd of January, not the 1st of March. Read the other way
    # round the row still parses and lands eleven months out of place.
    assert out.loc[0, "ds"] == pd.Timestamp("2024-01-03")


# --- compare_numbers_by_date --------------------------------------------------

def test_a_match_count_is_the_size_of_the_set_intersection():
    merged = compare_numbers_by_date(
        actual([("2024-01-03", "3-12-19-27-41")]),
        predicted([("2024-01-03", "3-12-40-27-5")]),
    )

    assert list(merged["matches"]) == [3]


def test_only_dates_present_in_both_frames_survive_the_merge():
    merged = compare_numbers_by_date(
        actual([("2024-01-03", "1-2-3-4-5"), ("2024-01-06", "6-7-8-9-10")]),
        predicted([("2024-01-06", "6-7-11-12-13"), ("2024-01-10", "1-2-3-4-5")]),
    )

    assert list(merged["ds"]) == [pd.Timestamp("2024-01-06")]
    assert list(merged["matches"]) == [2]


def test_a_frame_without_the_numbers_columns_raises_rather_than_scoring_nothing():
    with pytest.raises(ValueError, match="Missing 'numbers'"):
        compare_numbers_by_date(
            pd.DataFrame([{"ds": pd.Timestamp("2024-01-03"), "whatever": "1-2-3"}]),
            pd.DataFrame([{"ds": pd.Timestamp("2024-01-03"), "other": "1-2-3"}]),
        )


# --- check_actual_in_past_predictions -----------------------------------------

def test_only_predictions_strictly_before_the_draw_are_considered():
    """The one line here that would be a leak if it slipped: `<`, not `<=`.

    A prediction dated the day of the draw is not a past prediction, and
    counting it would quietly turn a hindcast into a look at the answer.
    """
    results = check_actual_in_past_predictions(
        actual([("2024-01-06", "1-2-3-4-5")]),
        predicted([("2024-01-06", "1-2-3-4-5")]),   # same day, perfect overlap
    )

    assert "5_matches_dates" not in results.columns


def test_an_earlier_prediction_overlapping_in_three_numbers_is_listed_by_date():
    results = check_actual_in_past_predictions(
        actual([("2024-01-06", "1-2-3-4-5")]),
        predicted([("2024-01-03", "1-2-3-40-41")]),
    )

    assert results.loc[0, "3_matches_dates"] == "2024-01-03"
    assert results.loc[0, "actual_date"] == "2024-01-06"


def test_overlaps_below_three_numbers_are_not_reported_at_all():
    results = check_actual_in_past_predictions(
        actual([("2024-01-06", "1-2-3-4-5")]),
        predicted([("2024-01-03", "1-2-40-41-42")]),
    )

    assert [c for c in results.columns if c.endswith("_matches_dates")] == []


def test_several_past_predictions_at_the_same_depth_are_joined_into_one_cell():
    results = check_actual_in_past_predictions(
        actual([("2024-01-06", "1-2-3-4-5")]),
        predicted([("2024-01-01", "1-2-3-40-41"), ("2024-01-03", "1-2-3-42-43")]),
    )

    assert results.loc[0, "3_matches_dates"] == "2024-01-01, 2024-01-03"


# --- process_and_compare_forecasts --------------------------------------------

def test_the_three_result_files_are_written_where_the_caller_asked(tmp_path):
    """The scripts' last line. Nothing else in the suite reaches it."""
    dates = pd.to_datetime(["2024-01-03", "2024-01-06"])
    forecasts = [
        pd.DataFrame({"ds": dates, "yhat_adjusted_0": [1, 6]}),
        pd.DataFrame({"ds": dates, "yhat_adjusted_1": [2, 7]}),
        pd.DataFrame({"ds": dates, "yhat_adjusted_2": [3, 8]}),
    ]
    actual_path = tmp_path / "actual.csv"
    pd.DataFrame([
        {"Date": "03/01/2024", "Ball Number": "1-2-30"},
        {"Date": "06/01/2024", "Ball Number": "6-7-8"},
    ]).to_csv(actual_path, index=False)

    out_dir = tmp_path / "results"
    comparison, cross_date = process_and_compare_forecasts(forecasts, str(actual_path), str(out_dir))

    written = sorted(p.name for p in out_dir.iterdir())
    assert written == [
        "actual_in_past_predictions_2024.csv",
        "final_combined_forecast.csv",
        "matched_numbers_by_date_2024.csv",
    ]
    # The per-position columns are joined back into one dash-separated draw,
    # which is the same string shape the data contract uses.
    assert list(comparison["numbers_predicted"]) == ["1-2-3", "6-7-8"]
    assert list(comparison["matches"]) == [2, 3]
    assert len(cross_date) == 2


def test_the_output_directory_is_created_if_it_does_not_exist(tmp_path):
    dates = pd.to_datetime(["2024-01-03"])
    actual_path = tmp_path / "actual.csv"
    pd.DataFrame([{"Date": "03/01/2024", "Ball Number": "1-2-3"}]).to_csv(actual_path, index=False)

    out_dir = tmp_path / "deep" / "nested" / "results"
    process_and_compare_forecasts(
        [pd.DataFrame({"ds": dates, "yhat_adjusted_0": [1]})], str(actual_path), str(out_dir)
    )

    assert (out_dir / "final_combined_forecast.csv").exists()
