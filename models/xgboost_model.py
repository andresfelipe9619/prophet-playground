"""XGBoost per ball position, with the temporal-leak bug from the old script fixed.

The previous XGBoost.py called `train_test_split(..., random_state=42)` with
its default `shuffle=True` on a time series — that puts future draws in the
training set and past draws in the test set, which silently inflates the
reported accuracy. Here every split is chronological (train on the past,
predict the future) and lag/rolling features are added, since calendar
features alone (day of week, month...) carry no information about which ball
number gets drawn.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from models.common import clip_to_range

DEFAULT_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror"}


FEATURE_LAGS = 3
ROLLING_WINDOW = 20


def create_features(df, n_lags=FEATURE_LAGS, rolling_window=ROLLING_WINDOW):
    """Build the feature matrix. Every feature is backward-looking.

    Rows are dropped on missing *features* only, never on a missing `y`, so a
    future row with no target survives and can be predicted (see forecast_next).
    """
    df = df.copy()
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_mean"] = df["y"].shift(1).rolling(rolling_window, min_periods=1).mean()
    df["rolling_freq_of_last_value"] = (
        df["y"].shift(1).rolling(rolling_window, min_periods=1)
        .apply(lambda w: (w == w[-1]).sum() if len(w) else np.nan, raw=True)
    )
    return df.dropna(subset=feature_columns(df)).reset_index(drop=True)


def feature_columns(df):
    return [c for c in df.columns if c not in ("ds", "y")]


def min_history_required(n_lags=FEATURE_LAGS, min_training_rows=10):
    """Draws needed before this model can predict: lag rows dropped, plus rows to train on."""
    return n_lags + min_training_rows


def chronological_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_predict(df, position, n_columns, test_size=0.2, params=None, num_round=100):
    features = create_features(df)
    train, test = chronological_split(features, test_size=test_size)

    feature_cols = feature_columns(features)
    dtrain = xgb.DMatrix(train[feature_cols], label=train["y"])
    dtest = xgb.DMatrix(test[feature_cols])

    booster = xgb.train(params or DEFAULT_PARAMS, dtrain, num_round)

    test = test.copy()
    test["yhat"] = booster.predict(dtest)
    test["yhat_adjusted"] = test["yhat"].apply(lambda x: clip_to_range(x, position, n_columns))
    return {"model": booster, "train": train, "test": test, "feature_cols": feature_cols}


def forecast_next(df, position, n_columns, next_date, params=None, num_round=100):
    """Predict the next, not-yet-drawn result — a real forecast, not a fitted value.

    The appended row carries no target at all (`y = NaN`): create_features
    drops rows on missing features only, and every feature is built from
    `shift(1)` or from `ds`. If a future change ever added a feature reading
    the row's own y, this would produce NaN rather than silently turning the
    forecast into a function of the last drawn number.
    """
    future_row = pd.DataFrame({"ds": [pd.Timestamp(next_date)], "y": [np.nan]})
    extended = pd.concat([df[["ds", "y"]], future_row], ignore_index=True)
    return train_predict_one_step(extended, position, n_columns, params=params, num_round=num_round)


def train_predict_one_step(df_upto_t, position, n_columns, params=None, num_round=100):
    """Train on every row except the last, predict that last row (1-step walk-forward)."""
    features = create_features(df_upto_t)
    train, predict_row = features.iloc[:-1], features.iloc[[-1]]
    if train["y"].notna().sum() < min_history_required() - FEATURE_LAGS:
        return None

    feature_cols = feature_columns(features)
    dtrain = xgb.DMatrix(train[feature_cols], label=train["y"])
    dpredict = xgb.DMatrix(predict_row[feature_cols])

    booster = xgb.train(params or DEFAULT_PARAMS, dtrain, num_round)
    yhat = float(booster.predict(dpredict)[0])
    return clip_to_range(yhat, position, n_columns)


def forecast_horizon(df_train, position, n_columns, future_dates, params=None, num_round=100):
    """Fit once on `df_train`, then predict `future_dates` recursively without refitting.

    Each prediction is appended to the working history as if it were the drawn
    number, so the next step's lags are built from it. That is the honest way
    to project a lag model past one step when the real values are being held
    out — but expect the output to flatten after a few draws: with no genuine
    signal the model regresses to the pool mean, its own output becomes the
    lag, and it converges on a fixed point. A holdout whose later steps are all
    the same number is that fixed point, not a bug.
    """
    features = create_features(df_train)
    feature_cols = feature_columns(features)
    booster = xgb.train(params or DEFAULT_PARAMS,
                        xgb.DMatrix(features[feature_cols], label=features["y"]), num_round)

    history = df_train[["ds", "y"]].copy()
    predictions = []
    for date in future_dates:
        extended = pd.concat(
            [history, pd.DataFrame({"ds": [pd.Timestamp(date)], "y": [np.nan]})], ignore_index=True
        )
        row = create_features(extended).iloc[[-1]]
        yhat = clip_to_range(float(booster.predict(xgb.DMatrix(row[feature_cols]))[0]),
                             position, n_columns)
        predictions.append(yhat)
        history = pd.concat(
            [history, pd.DataFrame({"ds": [pd.Timestamp(date)], "y": [float(yhat)]})],
            ignore_index=True,
        )
    return predictions
