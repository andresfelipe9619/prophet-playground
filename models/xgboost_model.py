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
import xgboost as xgb

from models.common import clip_to_range

DEFAULT_PARAMS = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror"}


def create_features(df, n_lags=3, rolling_window=20):
    df = df.copy()
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_mean"] = df["y"].shift(1).rolling(rolling_window, min_periods=1).mean()
    df["rolling_freq_of_last_value"] = (
        df["y"].shift(1).rolling(rolling_window, min_periods=1)
        .apply(lambda w: (w == w.iloc[-1]).sum() if len(w) else np.nan, raw=False)
    )
    return df.dropna().reset_index(drop=True)


def chronological_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_predict(df, position, n_columns, test_size=0.2, params=None, num_round=100):
    features = create_features(df)
    train, test = chronological_split(features, test_size=test_size)

    feature_cols = [c for c in features.columns if c not in ("ds", "y")]
    dtrain = xgb.DMatrix(train[feature_cols], label=train["y"])
    dtest = xgb.DMatrix(test[feature_cols])

    booster = xgb.train(params or DEFAULT_PARAMS, dtrain, num_round)

    test = test.copy()
    test["yhat"] = booster.predict(dtest)
    test["yhat_adjusted"] = test["yhat"].apply(lambda x: clip_to_range(x, position, n_columns))
    return {"model": booster, "train": train, "test": test, "feature_cols": feature_cols}


def train_predict_one_step(df_upto_t, position, n_columns, params=None, num_round=100):
    """Train on every row except the last, predict that last row (1-step walk-forward)."""
    features = create_features(df_upto_t, n_lags=3, rolling_window=20)
    if len(features) < 10:
        return None

    train, predict_row = features.iloc[:-1], features.iloc[[-1]]
    feature_cols = [c for c in features.columns if c not in ("ds", "y")]
    dtrain = xgb.DMatrix(train[feature_cols], label=train["y"])
    dpredict = xgb.DMatrix(predict_row[feature_cols])

    booster = xgb.train(params or DEFAULT_PARAMS, dtrain, num_round)
    yhat = float(booster.predict(dpredict)[0])
    return clip_to_range(yhat, position, n_columns)
