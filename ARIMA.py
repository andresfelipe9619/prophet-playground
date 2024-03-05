import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils.processor import load_actual_2024_data, load_and_preprocess, compare_numbers_by_date, \
    check_actual_in_past_predictions, process_and_compare_forecasts


# Function to define and fit the SARIMA model
def define_and_fit_model(series, seasonal_order=(1, 0, 1, 2)):
    """
    Fits a SARIMA model to the series.

    :param series: Pandas Series, the time series to model.
    :param seasonal_order: tuple, the (P, D, Q, S) order of the seasonal component of the model.
    """
    series = series.dropna().astype(float)  # Ensure no NaN values and data is float
    model = SARIMAX(series, order=(1, 2, 1), seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)  # disp=False to reduce output
    return model_fit


# Function to make predictions with ARIMA
def make_predictions(model_fit, steps, max_number):
    # Generate forecast values
    forecast = model_fit.forecast(steps=steps)

    # Adjust forecast values to be within the specified range
    adjusted_forecast = [min(max(1, round(value)), max_number) for value in forecast]

    return adjusted_forecast


# Main code execution
if __name__ == "__main__":
    file_path = 'exported_data/final-final.csv'
    actual_2024_file_path = 'exported_data/exported_data_2024.csv'

    try:
        df, balls_expanded = load_and_preprocess(file_path)
        all_forecasts = []  # This will store all the forecast DataFrames

        for i in range(balls_expanded.shape[1]):  # Assuming multiple columns for different number sets
            temp_df = df[['ds']].copy()  # Ensure 'ds' is your date column after copy
            temp_df['y'] = balls_expanded.iloc[:, i]
            temp_df = temp_df.dropna()  # Remove rows where 'y' could be NaN

            # Convert 'ds' column to datetime explicitly
            temp_df['ds'] = pd.to_datetime(temp_df['ds'], format='%Y-%m-%d')

            # Define and fit ARIMA model
            arima_model = define_and_fit_model(temp_df['y'])
            forecast = make_predictions(arima_model, 60, 16 if i == balls_expanded.shape[1] - 1 else 43)

            # Create a range of forecast dates starting from the day after the last date in temp_df
            forecast_dates = pd.date_range(start=temp_df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=len(forecast),
                                           freq='D')

            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                f'yhat_adjusted_{i}': forecast
            })
            all_forecasts.append(forecast_df)

        # Process and compare forecasts with actual data
        process_and_compare_forecasts(all_forecasts, actual_2024_file_path)

    except Exception as e:
        print(f"An error occurred: {e}")
