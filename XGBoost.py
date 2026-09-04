"""XGBoost forecast per ball position, evaluated on a held-out chronological tail.

See models/xgboost_model.py for the fix to the previous version's random
(shuffled) train/test split, which was leaking future draws into training.
"""

from matplotlib import pyplot as plt

from models.common import build_position_series
from models.xgboost_model import train_predict
from utils.processor import load_and_preprocess, process_and_compare_forecasts


def plot_results(original, forecast, label):
    plt.figure(figsize=(14, 7))
    plt.plot(original["ds"], original["y"], label="Actual")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", alpha=0.7)
    plt.legend()
    plt.title(f"XGBoost Forecast vs Actuals - {label}")
    plt.show()


if __name__ == "__main__":
    file_path = "exported_data/final-final.csv"
    actual_2024_file_path = "exported_data/exported_data_2024.csv"

    try:
        df, balls_expanded = load_and_preprocess(file_path)
        position_series = build_position_series(df, balls_expanded)
        n_columns = balls_expanded.shape[1]

        all_predictions = []
        for position, temp_df in position_series.items():
            result = train_predict(temp_df, position, n_columns)
            forecast = result["test"][["ds", "yhat_adjusted"]]
            all_predictions.append(
                forecast.rename(columns={"yhat_adjusted": f"yhat_adjusted_{position}"})
            )

        process_and_compare_forecasts(all_predictions, actual_2024_file_path, "xgboost_results/")

    except Exception as e:
        print(f"An error occurred: {e}")
