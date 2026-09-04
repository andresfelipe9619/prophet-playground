"""Prophet 1.4 forecasting per ball position.

Compared to the previous version, this drops the hand-tuned custom
seasonalities (weekly/biweekly/yearly Fourier terms) that were being fit on
a series that only has observations on draw days (Wed/Sat) — there is no
real periodic signal there to capture, so those terms were just fitting
noise and inflating the model's confidence. Seasonality/holidays are now
opt-in flags so you can turn them on for experimentation, but they're off
by default. See analysis/randomness.py for whether a given position shows
*any* evidence of non-randomness before trusting a forecast from here.
"""

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from contants import COLOMBIA_HOLIDAYS  # noqa: F401 — opt-in regressor, see forecast_position(holidays=...)
from models.common import (
    build_position_series,
    clip_to_range,
    infer_draw_weekdays,
    next_draw_dates,
    series_label,
)
from utils.processor import load_and_preprocess, process_and_compare_forecasts


def define_and_fit_model(series, holidays=None, weekly_seasonality=False, yearly_seasonality=False,
                          changepoint_prior_scale=0.05):
    m = Prophet(
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
    )
    m.fit(series)
    return m


def make_predictions(model, position, n_columns, periods, history=None):
    """Forecast the next `periods` draws.

    Future rows land on the real draw calendar (Mon/Wed/Sat, inferred from
    the history when available) instead of an evenly spaced frequency —
    draw days are 2 and 3 days apart, so no single `freq` fits them.
    """
    history = model.history if history is None else history
    weekdays = infer_draw_weekdays(history["ds"])
    future_dates = next_draw_dates(history["ds"].max(), periods, weekdays=weekdays)
    future = pd.concat([
        history[["ds"]],
        pd.DataFrame({"ds": pd.to_datetime(future_dates)}),
    ], ignore_index=True)
    forecast = model.predict(future)
    forecast["yhat_adjusted"] = forecast["yhat"].apply(lambda x: clip_to_range(x, position, n_columns))
    return forecast


def predict_at_dates(model, position, n_columns, dates):
    """Predict at specific known draw dates instead of an evenly-spaced future range."""
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    forecast = model.predict(future)
    forecast["yhat_adjusted"] = forecast["yhat"].apply(lambda x: clip_to_range(x, position, n_columns))
    return forecast


def evaluate_model_performance(model, initial, period, horizon):
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_p = performance_metrics(df_cv)
    return df_cv, df_p


def forecast_position(position_series, position, n_columns, periods=8, holidays=None,
                       run_cross_validation=False, cv_initial="730 days", cv_period="90 days",
                       cv_horizon="60 days"):
    """Fit + forecast one ball position. Used directly by the dashboard, and by __main__ below."""
    model = define_and_fit_model(position_series, holidays=holidays)
    forecast = make_predictions(model, position, n_columns, periods)
    result = {"label": series_label(position, n_columns), "model": model, "forecast": forecast}
    if run_cross_validation:
        df_cv, df_p = evaluate_model_performance(model, cv_initial, cv_period, cv_horizon)
        result["cv"] = df_cv
        result["performance"] = df_p
    return result


if __name__ == "__main__":
    file_path = "exported_data/final-final.csv"
    actual_2024_file_path = "exported_data/exported_data_2024.csv"

    try:
        df, balls_expanded = load_and_preprocess(file_path)
        position_series = build_position_series(df, balls_expanded)
        n_columns = balls_expanded.shape[1]

        all_predictions = []
        for position, temp_df in position_series.items():
            # holidays=COLOMBIA_HOLIDAYS is available but off by default: a public
            # holiday has no causal effect on which ball comes out of the machine.
            result = forecast_position(
                temp_df, position, n_columns, periods=120, run_cross_validation=True,
            )
            print(f"{result['label']} performance metrics:")
            print(result["performance"].head())

            forecast = result["forecast"]
            all_predictions.append(
                forecast[["ds", "yhat_adjusted"]].rename(columns={"yhat_adjusted": f"yhat_adjusted_{position}"})
            )

        process_and_compare_forecasts(all_predictions, actual_2024_file_path, "prophet_results/")

    except Exception as e:
        print(f"An error occurred: {e}")
