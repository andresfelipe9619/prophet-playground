"""Randomness diagnostics — analysis/randomness.py.

The sorted-data trap is the invariant here: when official results are
published sorted ascending, each column is an order statistic and
per-position chi-square reports spurious structure. `pooled_uniformity_test`
is the sort-proof verdict, and it must refuse to pool ranges that differ.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.randomness import (
    autocorrelation_check,
    chi_square_uniformity,
    frequency_table,
    gap_table,
    hot_cold_numbers,
    is_sorted_ascending,
    pooled_uniformity_test,
    randomness_report,
    runs_test,
)
from models.common import main_positions, super_position


def test_frequency_table_covers_the_whole_range(position_series, n_columns):
    table = frequency_table(position_series[0], 0, n_columns)
    assert list(table["number"]) == list(range(1, 44))
    assert table["count"].sum() == len(position_series[0])

    super_table = frequency_table(
        position_series[super_position(n_columns)], super_position(n_columns), n_columns
    )
    assert list(super_table["number"]) == list(range(1, 17))


def test_frequency_table_includes_numbers_never_drawn(n_columns):
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=3), "y": [1, 1, 2]})
    table = frequency_table(frame, 0, n_columns)
    assert len(table) == 43
    assert int(table.loc[table["number"] == 40, "count"].iloc[0]) == 0


def test_is_sorted_ascending_detects_published_sorted_results():
    sorted_rows = pd.DataFrame([[1, 8, 19, 27, 41, 5], [2, 3, 30, 33, 40, 9]])
    assert is_sorted_ascending(sorted_rows)


def test_is_sorted_ascending_ignores_the_superbalota_column():
    """The superbalota is independent and routinely breaks the ordering."""
    rows = pd.DataFrame([[1, 8, 19, 27, 41, 2], [2, 3, 30, 33, 40, 1]])
    assert is_sorted_ascending(rows)


def test_sample_data_is_not_sorted(sample):
    """The synthetic data stays unsorted so the tests show the clean i.i.d. case."""
    assert not is_sorted_ascending(sample[1])


def test_pooled_uniformity_refuses_to_mix_ball_ranges(sample, n_columns):
    """Pooling 1-16 with 1-43 is not expressible — value range-checking cannot catch it,
    since 1-16 is a subset of 1-43."""
    balls = sample[1]
    with pytest.raises(ValueError, match="different ball ranges"):
        pooled_uniformity_test(balls, [0, 1, super_position(n_columns)])


def test_pooled_uniformity_over_the_main_positions(sample, n_columns):
    balls = sample[1]
    result = pooled_uniformity_test(balls, main_positions(n_columns))
    assert result["n_categories"] == 43
    assert result["n_observations"] == len(balls) * 5
    assert result["p_value"] > 0.01, "synthetic data is uniform by construction"


def test_pooled_uniformity_over_the_superbalota_alone(sample, n_columns):
    result = pooled_uniformity_test(sample[1], [super_position(n_columns)])
    assert result["n_categories"] == 16
    assert result["n_observations"] == len(sample[1])


def test_pooled_uniformity_is_sort_proof(sample, n_columns):
    """Sorting each draw must not change the pooled verdict — that is the point of it."""
    balls = sample[1]
    mains = list(main_positions(n_columns))
    unsorted_result = pooled_uniformity_test(balls, mains)

    sorted_balls = balls.copy()
    sorted_balls[mains] = np.sort(balls[mains].to_numpy(), axis=1)
    sorted_result = pooled_uniformity_test(sorted_balls, mains)

    assert sorted_result["chi2"] == pytest.approx(unsorted_result["chi2"])
    assert sorted_result["p_value"] == pytest.approx(unsorted_result["p_value"])


def test_chi_square_flags_a_rigged_position(n_columns):
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=200), "y": [7] * 200})
    result = chi_square_uniformity(frequency_table(frame, 0, n_columns))
    assert result["p_value"] < 1e-6


def test_runs_test_flags_a_perfectly_alternating_sequence():
    result = runs_test([1, 40] * 60)
    assert result["p_value"] < 0.01


def test_runs_test_on_random_data_looks_random():
    rng = np.random.default_rng(3)
    assert runs_test(rng.integers(1, 44, size=400))["p_value"] > 0.05


def test_runs_test_handles_a_degenerate_sequence():
    result = runs_test([5, 5, 5, 5])
    assert np.isnan(result["z"])


def test_autocorrelation_check_on_random_data(position_series):
    result = autocorrelation_check(position_series[0]["y"].to_numpy())
    assert len(result["acf"]) == result["n_lags"] + 1
    assert result["acf"][0] == pytest.approx(1.0)
    assert result["ljung_box_p_value"] > 0.01


def test_gap_table_marks_numbers_never_seen(n_columns):
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=4), "y": [1, 2, 1, 2]})
    table = gap_table(frame, 0, n_columns)
    assert len(table) == 43
    unseen = table[table["number"] == 30].iloc[0]
    assert unseen["times_seen"] == 0
    assert pd.isna(unseen["last_date"])


def test_hot_cold_is_sorted_by_delta(position_series, n_columns):
    table = hot_cold_numbers(position_series[0], 0, n_columns, recent_draws=20)
    assert list(table["delta_pct"]) == sorted(table["delta_pct"], reverse=True)
    assert table["recent_count"].sum() == 20


def test_randomness_report_says_random_for_random_data(position_series, n_columns):
    report = randomness_report(position_series[0], 0, n_columns)
    assert report["label"] == "Balota 1"
    assert report["n_draws"] == len(position_series[0])
    assert report["looks_random"] is True
