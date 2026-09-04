"""AutoARIMA / AutoETS / AutoTheta forecasts (Nixtla statsforecast), one per ball position.

Replaces the old ARIMA.py, which fit a single manually-chosen SARIMAX order
per series. Here every position gets its own AIC-searched order per model,
fit for all positions in one call. Pick which model's column to use for the
final "combined" prediction via --model (default AutoARIMA).
"""

import sys

from models.common import (
    DEFAULT_DATA_PATH,
    build_position_series,
    infer_draw_weekdays,
    next_draw_dates,
)
from models.statsforecast_model import adjusted_predictions, fit_predict_all
from utils.processor import load_and_preprocess, process_and_compare_forecasts

if __name__ == "__main__":
    actual_2024_file_path = "exported_data/exported_data_2024.csv"
    model_name = sys.argv[1] if len(sys.argv) > 1 else "AutoARIMA"
    h = 60

    df, balls_expanded = load_and_preprocess(DEFAULT_DATA_PATH)
    position_series = build_position_series(df, balls_expanded)
    n_columns = balls_expanded.shape[1]

    raw_forecast = fit_predict_all(position_series, h=h)
    clipped = adjusted_predictions(raw_forecast, n_columns, model_name=model_name)

    future_dates = next_draw_dates(df["ds"].max(), h, weekdays=infer_draw_weekdays(df["ds"]))

    all_predictions = []
    for position in range(n_columns):
        pos_forecast = clipped[clipped["unique_id"] == position].sort_values("ds").reset_index(drop=True)
        pos_forecast["ds"] = future_dates[: len(pos_forecast)]
        all_predictions.append(
            pos_forecast[["ds", "yhat_adjusted"]].rename(columns={"yhat_adjusted": f"yhat_adjusted_{position}"})
        )

    process_and_compare_forecasts(all_predictions, actual_2024_file_path, f"statsforecast_{model_name}_results")
