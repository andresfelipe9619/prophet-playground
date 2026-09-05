"""XGBoost feature building and splits — models/xgboost_model.py.

The original code used a shuffled train_test_split, leaking future draws into
training. Every test here is about that not coming back.
"""

import numpy as np
import pandas as pd
import pytest

from models.common import range_for_position
from models.xgboost_model import (
    FEATURE_LAGS,
    chronological_split,
    create_features,
    feature_columns,
    forecast_horizon,
    forecast_next,
    min_history_required,
    train_predict,
    train_predict_one_step,
)


def test_chronological_split_never_puts_the_future_in_training(position_series):
    features = create_features(position_series[0])
    train, test = chronological_split(features, test_size=0.2)
    assert len(train) + len(test) == len(features)
    assert train["ds"].max() < test["ds"].min(), "no future draw may appear in training"
    assert train["ds"].is_monotonic_increasing and test["ds"].is_monotonic_increasing


def test_every_feature_is_backward_looking(position_series):
    """No feature may read its own row's y — that would leak the answer."""
    features = create_features(position_series[0])
    for column in feature_columns(features):
        assert features[column].corr(features["y"]) < 0.5, column


def test_lag_features_hold_the_previous_values():
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=10), "y": range(10)})
    features = create_features(frame, n_lags=2, rolling_window=3)
    assert list(features["y"]) == list(range(2, 10)), "the first n_lags rows have no features"
    assert (features["lag_1"] == features["y"] - 1).all()
    assert (features["lag_2"] == features["y"] - 2).all()


def test_rows_are_dropped_on_missing_features_never_on_a_missing_target():
    """A future row with no y must survive so it can be predicted."""
    frame = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=8),
        "y": [1.0, 2, 3, 4, 5, 6, 7, np.nan],
    })
    features = create_features(frame, n_lags=FEATURE_LAGS, rolling_window=3)
    assert features["y"].isna().sum() == 1
    assert features.iloc[-1][feature_columns(features)].notna().all()


def test_min_history_required_accounts_for_dropped_lag_rows():
    assert min_history_required(n_lags=3, min_training_rows=10) == 13


def test_one_step_returns_none_without_enough_history():
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=6), "y": [1, 2, 3, 4, 5, 6]})
    assert train_predict_one_step(frame, 0, 6) is None


def test_one_step_prediction_is_a_legal_ball(position_series, n_columns):
    for position in (0, n_columns - 1):
        low, high = range_for_position(position, n_columns)
        yhat = train_predict_one_step(position_series[position].iloc[:120], position, n_columns)
        assert yhat is not None and low <= yhat <= high


def test_forecast_next_does_not_read_the_row_it_predicts(position_series, n_columns):
    """The appended future row carries y = NaN; the prediction must still come out."""
    history = position_series[0].iloc[:120]
    next_date = history["ds"].max() + pd.Timedelta(days=2)
    yhat = forecast_next(history, 0, n_columns, next_date)
    assert yhat is not None and 1 <= yhat <= 43


def test_train_predict_clips_every_output(position_series, n_columns):
    position = n_columns - 1  # the superbalota, where clipping bites hardest
    result = train_predict(position_series[position].iloc[:150], position, n_columns)
    assert result["test"]["yhat_adjusted"].between(1, 16).all()
    assert result["train"]["ds"].max() < result["test"]["ds"].min()


def test_forecast_horizon_returns_one_legal_ball_per_date(position_series, n_columns):
    history = position_series[0].iloc[:120]
    future = pd.date_range(history["ds"].max() + pd.Timedelta(days=2), periods=5, freq="3D")
    predictions = forecast_horizon(history, 0, n_columns, future)
    assert len(predictions) == 5
    assert all(1 <= p <= 43 for p in predictions)
