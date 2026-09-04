"""AutoARIMA / AutoETS / AutoTheta forecasts (Nixtla statsforecast), one per ball position.

Replaces the old ARIMA.py, which fit a single manually-chosen SARIMAX order
per series. Here every position gets its own AIC-searched order per model,
fit for all positions in one call. Pick which model's column to use for the
final "combined" prediction via --model (default AutoARIMA).
"""

import sys

from models.common import build_position_series, next_draw_dates
from models.statsforecast_model import adjusted_predictions, fit_predict_all
from utils.processor import load_and_preprocess, process_and_compare_forecasts

if __name__ == "__main__":
    file_path = "exported_data/final-final.csv"
    actual_2024_file_path = "exported_data/exported_data_2024.csv"
    model_name = sys.argv[1] if len(sys.argv) > 1 else "AutoARIMA"
    h = 60

    try:
        df, balls_expanded = load_and_preprocess(file_path)
        position_series = build_position_series(df, balls_expanded)
        n_columns = balls_expanded.shape[1]

        raw_forecast = fit_predict_all(position_series, h=h)
        clipped = adjusted_predictions(raw_forecast, n_columns, model_name=model_name)

        last_date = df["ds"].max()
        future_dates = next_draw_dates(last_date, h)

        all_predictions = []
        for position in range(n_columns):
            pos_forecast = clipped[clipped["unique_id"] == position].sort_values("ds").reset_index(drop=True)
            pos_forecast["ds"] = future_dates[: len(pos_forecast)]
            all_predictions.append(
                pos_forecast[["ds", "yhat_adjusted"]].rename(columns={"yhat_adjusted": f"yhat_adjusted_{position}"})
            )

        process_and_compare_forecasts(all_predictions, actual_2024_file_path, f"statsforecast_{model_name}_results/")

    except Exception as e:
        print(f"An error occurred: {e}")
