"""Fit Prophet per ball position and compare the forecasts against 2024 results.

    python -m scripts.prophet_forecast
"""

from lottery.models.common import DEFAULT_DATA_PATH, build_position_series
from lottery.models.prophet_model import forecast_position
from lottery.utils.processor import load_and_preprocess, process_and_compare_forecasts

if __name__ == "__main__":
    actual_2024_file_path = "exported_data/exported_data_2024.csv"

    df, balls_expanded = load_and_preprocess(DEFAULT_DATA_PATH)
    position_series = build_position_series(df, balls_expanded)
    n_columns = balls_expanded.shape[1]

    all_predictions = []
    for position, temp_df in position_series.items():
        # holidays are opt-in (forecast_position(holidays=lottery.constants.COLOMBIA_HOLIDAYS)) and
        # off here: a public holiday has no causal effect on which ball comes out.
        result = forecast_position(
            temp_df, position, n_columns, periods=120, run_cross_validation=True,
        )
        print(f"{result['label']} performance metrics:")
        print(result["performance"].head())

        forecast = result["forecast"]
        all_predictions.append(
            forecast[["ds", "yhat_adjusted"]].rename(columns={"yhat_adjusted": f"yhat_adjusted_{position}"})
        )

    process_and_compare_forecasts(all_predictions, actual_2024_file_path, "prophet_results")
