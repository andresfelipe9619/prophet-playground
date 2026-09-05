"""Shared fixtures.

Everything is built from lottery.utils.sample_data, which is seeded, so every test in
this suite is deterministic and needs no private CSV.
"""

import pandas as pd
import pytest

from lottery.models.common import build_position_series
from lottery.utils.processor import preprocess_draws
from lottery.utils.sample_data import load_sample_and_preprocess


@pytest.fixture(scope="session")
def sample():
    """(df, balls_expanded) of 200 synthetic current-format draws."""
    return load_sample_and_preprocess(n_draws=200)


@pytest.fixture(scope="session")
def n_columns(sample):
    return sample[1].shape[1]


@pytest.fixture(scope="session")
def position_series(sample):
    df, balls_expanded = sample
    return build_position_series(df, balls_expanded)


def draws_frame(rows):
    """Build a raw (Date, Ball) frame the way a CSV or an upload would arrive."""
    return pd.DataFrame([{"Date": date, "Ball": ball} for date, ball in rows])


def parsed(rows):
    return preprocess_draws(draws_frame(rows), validate=False)
