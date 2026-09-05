"""XGBoost per ball position, evaluated on a held-out chronological tail.

See lottery/models/xgboost_model.py for the fix to the previous version's random
(shuffled) train/test split, which was leaking future draws into training.

Note what this script produces: out-of-sample predictions for draws that
already happened, which is what comparing against the actual 2024 results
requires. For a prediction of the *next* draw use
lottery.models.xgboost_model.forecast_next (what the dashboard's Forecast tab calls).

    python -m scripts.xgboost_forecast
"""

from lottery.models.common import DEFAULT_DATA_PATH, build_position_series
from lottery.models.xgboost_model import train_predict
from lottery.utils.processor import load_and_preprocess, process_and_compare_forecasts

if __name__ == "__main__":
    actual_2024_file_path = "exported_data/exported_data_2024.csv"

    df, balls_expanded = load_and_preprocess(DEFAULT_DATA_PATH)
    position_series = build_position_series(df, balls_expanded)
    n_columns = balls_expanded.shape[1]

    all_predictions = []
    for position, temp_df in position_series.items():
        result = train_predict(temp_df, position, n_columns)
        all_predictions.append(
            result["test"][["ds", "yhat_adjusted"]]
            .rename(columns={"yhat_adjusted": f"yhat_adjusted_{position}"})
        )

    process_and_compare_forecasts(all_predictions, actual_2024_file_path, "xgboost_results")
