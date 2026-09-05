"""The chance baseline and the z-test that decides whether a model has signal.

Two invariants matter more than the rest and are pinned hard below:
one-sided vs two-sided p-values, and per-window `m_guessed`.
"""

import numpy as np
import pandas as pd
import pytest

from models.baseline import (
    beats_chance_test,
    expected_main_matches,
    expected_super_match_rate,
    most_frequent_pick,
)
from models.common import MAIN_BALLS_DRAWN, MAIN_POOL, SUPER_POOL


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 5])
def test_expected_matches_is_the_exact_hypergeometric_mean(m):
    """Guessing m distinct numbers: E[matches] = m * 5 / 43, exactly."""
    assert expected_main_matches(m)["mean"] == pytest.approx(m * MAIN_BALLS_DRAWN / MAIN_POOL)


def test_guessing_nothing_can_match_nothing():
    chance = expected_main_matches(0)
    assert chance["mean"] == 0.0
    assert chance["var"] == 0.0


def test_super_match_rate_is_one_in_the_pool():
    assert expected_super_match_rate() == pytest.approx(1 / SUPER_POOL)


def test_a_chance_level_model_does_not_beat_chance():
    """Hits drawn from the null itself must not produce a small one-sided p-value."""
    rng = np.random.default_rng(0)
    hits = rng.hypergeometric(MAIN_BALLS_DRAWN, MAIN_POOL - MAIN_BALLS_DRAWN, MAIN_BALLS_DRAWN, size=500)
    result = beats_chance_test(hits, MAIN_BALLS_DRAWN)
    assert result["p_value_greater"] > 0.05
    assert result["observed_mean"] == pytest.approx(result["chance_mean"], abs=0.15)


def test_a_model_worse_than_chance_must_not_read_as_beating_it():
    """The reason only `p_value_greater` may back a 'beats chance' claim.

    A model significantly *worse* than chance also gets a tiny two-sided
    p-value. Reading `p_value` as the verdict would call it a winner.
    """
    hits = [0] * 400  # far below the chance mean of 5 * 5 / 43
    result = beats_chance_test(hits, MAIN_BALLS_DRAWN)
    assert result["z"] < 0
    assert result["p_value"] < 0.01, "two-sided: clearly differs from chance"
    assert result["p_value_greater"] > 0.99, "one-sided: emphatically does not beat it"


def test_a_model_better_than_chance_is_detected():
    hits = [3] * 200  # wildly above 5 * 5 / 43
    result = beats_chance_test(hits, MAIN_BALLS_DRAWN)
    assert result["z"] > 0
    assert result["p_value_greater"] < 1e-6


def test_m_guessed_accepts_a_per_window_list():
    """Collisions between positions change the distinct-guess count per window."""
    hits = [1, 0, 1, 0]
    per_window = beats_chance_test(hits, [5, 4, 3, 5])
    uniform = beats_chance_test(hits, 5)
    assert per_window["chance_mean"] != uniform["chance_mean"]
    assert per_window["chance_mean"] == pytest.approx(
        np.mean([expected_main_matches(m)["mean"] for m in (5, 4, 3, 5)])
    )


def test_empty_input_returns_nan_not_a_crash():
    result = beats_chance_test([], MAIN_BALLS_DRAWN)
    assert np.isnan(result["z"])
    assert np.isnan(result["p_value_greater"])


def test_zero_variance_is_handled():
    """m_guessed=0 has no variance: report the means, refuse to invent a z."""
    result = beats_chance_test([0, 0, 0], 0)
    assert np.isnan(result["z"])
    assert result["observed_mean"] == 0.0


def test_most_frequent_pick_returns_one_legal_ball_per_position(position_series, n_columns):
    from models.common import range_for_position

    pick = most_frequent_pick(position_series)
    assert set(pick) == set(range(n_columns))
    for position, value in pick.items():
        low, high = range_for_position(position, n_columns)
        assert low <= value <= high


def test_most_frequent_pick_upto_only_sees_the_past(position_series):
    """Walk-forward use: the pick at t must not depend on draws at or after t."""
    early = most_frequent_pick(position_series, upto=60)
    truncated = {p: f.iloc[:60] for p, f in position_series.items()}
    assert early == most_frequent_pick(truncated)


def test_most_frequent_pick_is_the_mode():
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=5), "y": [7, 7, 7, 3, 9]})
    assert most_frequent_pick({0: frame}) == {0: 7}
