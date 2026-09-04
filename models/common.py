"""Shared config and helpers for every model/analysis module.

Baloto: pick 5 main balls from 1-43 (no repeats) plus one "superbalota"
from 1-16. Draws run Monday, Wednesday and Saturday. Every module that
needs those bounds or needs to turn `balls_expanded` (from
utils.processor.load_and_preprocess) into one series per ball position
imports from here so the rules only live in one place.
"""

from datetime import timedelta

import pandas as pd

MAIN_BALL_RANGE = (1, 43)
MAIN_BALLS_DRAWN = 5
SUPER_BALL_RANGE = (1, 16)
MAIN_POOL = MAIN_BALL_RANGE[1]
SUPER_POOL = SUPER_BALL_RANGE[1]
DRAW_WEEKDAYS = (0, 2, 5)  # Monday, Wednesday, Saturday
DEFAULT_DATA_PATH = "exported_data/final-final.csv"


def super_position(n_columns):
    return n_columns - 1


def main_positions(n_columns):
    return range(n_columns - 1)


def series_label(position, n_columns):
    return "Superbalota" if position == super_position(n_columns) else f"Balota {position + 1}"


def range_for_position(position, n_columns):
    return SUPER_BALL_RANGE if position == super_position(n_columns) else MAIN_BALL_RANGE


def max_for_position(position, n_columns):
    return range_for_position(position, n_columns)[1]


def min_for_position(position, n_columns):
    return range_for_position(position, n_columns)[0]


def clip_to_range(value, position, n_columns):
    low = min_for_position(position, n_columns)
    high = max_for_position(position, n_columns)
    return int(min(max(round(value), low), high))


def build_position_series(df, balls_expanded):
    """Return {position: DataFrame[ds, y]} for every ball/superbalota column, sorted by ds."""
    series = {}
    for position in range(balls_expanded.shape[1]):
        temp = df[["ds"]].copy()
        temp["y"] = balls_expanded[position]
        temp = temp.dropna().sort_values("ds").reset_index(drop=True)
        temp["y"] = temp["y"].astype(int)
        series[position] = temp
    return series


def infer_draw_weekdays(dates, recent=30):
    """Which weekdays does this dataset actually draw on?

    The schedule has changed over the years (Monday was added to the
    original Wednesday/Saturday pair), so reading it off the most recent
    draws is safer than hardcoding it — an old history stays consistent
    with itself, and a current one picks up Monday automatically.
    """
    tail = pd.Series(pd.to_datetime(pd.Series(dates))).sort_values().tail(recent)
    weekdays = sorted({int(d.weekday()) for d in tail})
    return tuple(weekdays) if weekdays else DRAW_WEEKDAYS


def next_draw_dates(last_date, h, weekdays=None):
    """Next h real draw dates after last_date."""
    weekdays = weekdays or DRAW_WEEKDAYS
    dates = []
    cursor = last_date
    while len(dates) < h:
        cursor = cursor + timedelta(days=1)
        if cursor.weekday() in weekdays:
            dates.append(cursor)
    return dates


def to_long_format(position_series):
    """Concatenate the per-position series into the (unique_id, ds, y) shape statsforecast expects.

    unique_id is the integer position (not the label) so predictions can be
    clipped back to the right [min, max] range without a name lookup.
    """
    frames = []
    for position, frame in position_series.items():
        f = frame[["ds", "y"]].copy()
        f["unique_id"] = position
        f["ds"] = range(1, len(f) + 1)  # sequential draw index; calendar gaps are irregular (draw days only)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
