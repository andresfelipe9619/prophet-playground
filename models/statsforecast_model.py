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

from models.common import clip_to_range, to_long_format

MODEL_NAMES = ("AutoARIMA", "AutoETS", "AutoTheta")


def build_models(season_length=1):
    return [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
    ]


def fit_predict_all(position_series, h=8, season_length=1, n_jobs=1):
    """Fit AutoARIMA/AutoETS/AutoTheta on every position at once, return the raw forecast frame."""
    long_df = to_long_format(position_series)
    sf = StatsForecast(models=build_models(season_length), freq=1, n_jobs=n_jobs)
    sf.fit(long_df)
    return sf.predict(h=h, level=[80, 95])


def adjusted_predictions(forecast_df, n_columns, model_name="AutoARIMA"):
    """Clip one model's raw predictions to each position's valid ball range."""
    out = forecast_df[["unique_id", "ds", model_name]].copy()
    out["yhat_adjusted"] = out.apply(
        lambda row: clip_to_range(row[model_name], int(row["unique_id"]), n_columns), axis=1
    )
    return out
