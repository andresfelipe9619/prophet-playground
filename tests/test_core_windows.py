"""core/windows.py — index arithmetic over an ordered sequence, no domain."""

import pandas as pd
import pytest

from core.windows import cutoff_bounds, window_bounds


@pytest.mark.parametrize(
    "n_observations, n_windows, min_train, expected",
    [
        (200, 15, 60, (185, 200)),   # the last 15
        (200, 500, 60, (60, 200)),   # more windows than history: min_train wins
        (100, 15, 100, (100, 100)),  # nothing to evaluate
        (100, 15, 200, (200, 100)),  # start > total: the caller must refuse
    ],
)
def test_window_bounds(n_observations, n_windows, min_train, expected):
    assert window_bounds(n_observations, n_windows, min_train) == expected


def test_window_bounds_never_evaluates_more_than_requested():
    start, total = window_bounds(500, 20, 10)
    assert total - start == 20


def test_the_cutoff_day_counts_as_training():
    dates = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-06", "2024-01-08"])
    assert cutoff_bounds(dates, "2024-01-03") == (2, 2)


def test_cutoff_bounds_always_splits_the_whole_history():
    dates = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-06"])
    for cutoff in ("2023-01-01", "2024-01-03", "2025-01-01"):
        n_train, n_holdout = cutoff_bounds(dates, cutoff)
        assert n_train + n_holdout == len(dates)


def test_cutoff_bounds_sorts_its_input():
    shuffled = pd.to_datetime(["2024-01-08", "2024-01-01", "2024-01-06", "2024-01-03"])
    assert cutoff_bounds(shuffled, "2024-01-03") == (2, 2)


def test_cutoff_bounds_accepts_plain_strings():
    assert cutoff_bounds(["2024-01-01", "2024-01-06"], "2024-01-01") == (1, 1)
