"""Classical models via Nixtla's statsforecast: AutoARIMA, AutoETS, AutoTheta.

This replaces the old hand-rolled, fixed-order SARIMAX from ARIMA.py. Instead
of guessing a single (p,d,q)(P,D,Q,s) order for every ball position,
statsforecast searches each series for the best-fitting order per model
(AIC-based) and fits all positions in one vectorized/parallel call, which is
both faster and less arbitrary than the previous approach.

The series are modeled on a plain sequential draw index (1, 2, 3, ...)
rather than calendar dates, since draws only happen on Wed/Sat and there is
no real daily frequency to align to.
"""

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

import numpy as np

from models.common import max_for_position, min_for_position, to_long_format

MODEL_NAMES = ("AutoARIMA", "AutoETS", "AutoTheta")


def build_models(season_length=1):
    return [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
    ]


def fit_predict_all(position_series, h=8, season_length=1, n_jobs=1, level=None):
    """Fit AutoARIMA/AutoETS/AutoTheta on every position at once, return the raw forecast frame.

    `unique_id` is normalized to a column: statsforecast returns it as the
    index before 2.0, and every caller here indexes it as a column.
    Prediction intervals cost real time per window, so `level` is opt-in.
    """
    long_df = to_long_format(position_series)
    sf = StatsForecast(models=build_models(season_length), freq=1, n_jobs=n_jobs)
    sf.fit(long_df)
    forecast = sf.predict(h=h, level=level)
    return forecast if "unique_id" in forecast.columns else forecast.reset_index()


def adjusted_predictions(forecast_df, n_columns, model_name="AutoARIMA"):
    """Clip one model's raw predictions to each position's valid ball range."""
    out = forecast_df[["unique_id", "ds", model_name]].copy()
    positions = out["unique_id"].to_numpy(dtype=int)
    lows = np.array([min_for_position(p, n_columns) for p in positions])
    highs = np.array([max_for_position(p, n_columns) for p in positions])
    out["yhat_adjusted"] = np.clip(np.round(out[model_name].to_numpy()), lows, highs).astype(int)
    return out
