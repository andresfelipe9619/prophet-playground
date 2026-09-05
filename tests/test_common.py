"""Position semantics and the draw calendar — lottery/models/common.py.

These are the invariants the whole codebase derives from, so they are the
ones a refactor is most likely to break silently.
"""

import pandas as pd
import pytest

from lottery.models.common import (
    MAIN_BALL_RANGE,
    SUPER_BALL_RANGE,
    build_position_series,
    clip_to_range,
    infer_draw_weekdays,
    main_positions,
    max_for_position,
    min_for_position,
    next_draw_dates,
    range_for_position,
    series_label,
    super_position,
    to_long_format,
)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_positions_are_derived_not_hardcoded(n):
    """`n_columns - 1` and `range(5)` only coincide at 6 columns; nothing may assume it."""
    assert super_position(n) == n - 1
    assert list(main_positions(n)) == list(range(n - 1))
    assert super_position(n) not in main_positions(n)


def test_super_and_main_ranges_differ(n_columns):
    assert range_for_position(super_position(n_columns), n_columns) == SUPER_BALL_RANGE
    for p in main_positions(n_columns):
        assert range_for_position(p, n_columns) == MAIN_BALL_RANGE
    assert max_for_position(super_position(n_columns), n_columns) == 16
    assert min_for_position(0, n_columns) == 1


def test_series_label_names_the_superbalota(n_columns):
    assert series_label(super_position(n_columns), n_columns) == "Superbalota"
    assert series_label(0, n_columns) == "Balota 1"


@pytest.mark.parametrize(
    "raw, position, expected",
    [
        (-5.0, 0, 1),      # below the floor
        (999.0, 0, 43),    # above the main ceiling
        (999.0, 5, 16),    # above the superbalota ceiling
        (12.4, 0, 12),     # rounds
        (12.6, 0, 13),
    ],
)
def test_clip_to_range_clamps_and_rounds(raw, position, expected):
    assert clip_to_range(raw, position, 6) == expected


def test_clip_to_range_always_returns_a_legal_ball(position_series, n_columns):
    """Any model output, however wild, must land inside the position's range."""
    for position in range(n_columns):
        low, high = range_for_position(position, n_columns)
        for raw in (-1e6, -0.4, 0, 1e6):
            assert low <= clip_to_range(raw, position, n_columns) <= high


def test_next_draw_dates_only_lands_on_draw_weekdays():
    start = pd.Timestamp("2024-01-01")  # a Monday
    dates = next_draw_dates(start, 10, weekdays=(0, 2, 5))
    assert len(dates) == 10
    assert all(d.weekday() in (0, 2, 5) for d in dates)
    assert all(b > a for a, b in zip(dates, dates[1:]))
    assert dates[0] > start


def test_next_draw_dates_is_not_a_fixed_frequency():
    """Draw days are 2 and 3 days apart — a fixed freq= would produce phantom draws."""
    dates = next_draw_dates(pd.Timestamp("2024-01-01"), 12, weekdays=(0, 2, 5))
    gaps = {(b - a).days for a, b in zip(dates, dates[1:])}
    assert gaps == {2, 3}


def test_infer_draw_weekdays_reads_the_schedule_off_the_tail():
    """A Wednesday/Saturday-only history must not start generating Monday draws."""
    wed_sat = [d for d in pd.date_range("2015-01-01", periods=200) if d.weekday() in (2, 5)]
    assert infer_draw_weekdays(wed_sat) == (2, 5)

    mon_wed_sat = [d for d in pd.date_range("2024-01-01", periods=120) if d.weekday() in (0, 2, 5)]
    assert infer_draw_weekdays(mon_wed_sat) == (0, 2, 5)


def test_infer_draw_weekdays_follows_a_schedule_change():
    """Monday was added over time; only the recent tail should count."""
    old = [d for d in pd.date_range("2015-01-01", periods=400) if d.weekday() in (2, 5)]
    new = [d for d in pd.date_range("2024-01-01", periods=120) if d.weekday() in (0, 2, 5)]
    assert infer_draw_weekdays(old + new, recent=30) == (0, 2, 5)


def test_build_position_series_shape(position_series, sample, n_columns):
    df, balls_expanded = sample
    assert set(position_series) == set(range(n_columns))
    for position, frame in position_series.items():
        assert list(frame.columns) == ["ds", "y"]
        assert len(frame) == len(df)
        assert frame["ds"].is_monotonic_increasing
        assert frame["y"].dtype.kind == "i"


def test_to_long_format_uses_a_sequential_draw_index(position_series, n_columns):
    """statsforecast runs on a draw index, not calendar dates: gaps are irregular."""
    long_df = to_long_format(position_series)
    assert set(long_df.columns) == {"ds", "y", "unique_id"}
    assert set(long_df["unique_id"]) == set(range(n_columns))
    for position, group in long_df.groupby("unique_id"):
        assert list(group["ds"]) == list(range(1, len(group) + 1))


def test_to_long_format_keeps_position_as_an_integer_id(position_series):
    """The id must stay the position, not a label, so predictions can be clipped back."""
    long_df = to_long_format(position_series)
    assert all(isinstance(int(u), int) for u in long_df["unique_id"].unique())
