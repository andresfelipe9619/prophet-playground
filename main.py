import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from itertools import chain


# Function to load and preprocess data
def load_and_preprocess(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    df = pd.read_csv(path)
    # Validate expected columns
    expected_columns = ['Ball', 'Date']
    if not all(column in df.columns for column in expected_columns):
        missing = list(set(expected_columns) - set(df.columns))
        raise ValueError(f"Missing columns in the dataset: {missing}")
    # Data preprocessing
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.rename(columns={'Date': 'ds', 'Ball': 'ball'}, inplace=True)
    df['y'] = df['ball'].apply(lambda x: int(x.split('-')[0]) if '-' in x else int(x))
    return df, [int(number) for number in chain.from_iterable(df['ball'].str.split('-'))]


# Function to define and fit the Prophet model
def define_and_fit_model(df, holidays):
    m = Prophet(changepoint_prior_scale=0.05, interval_width=0.95, holidays=holidays)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(df)
    return m


# Function for making predictions
def make_predictions(custom_model, periods, max_number):
    future = custom_model.make_future_dataframe(periods=periods)
    my_forecast = custom_model.predict(future)
    my_forecast['yhat_adjusted'] = my_forecast['yhat'].apply(lambda x: min(max(1, round(x)), max_number))
    return my_forecast


# Function for plotting results
def plot_results(forecast):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    # Original forecast
    fig1 = model.plot(forecast, ax=axs[0])
    axs[0].set_title('Original Forecast Plot')
    # Adjusted forecast
    axs[1].plot(forecast['ds'], forecast['yhat_adjusted'], label='Adjusted Forecast', color='orange')
    axs[1].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='gray')
    axs[1].legend()
    axs[1].set_title('Adjusted Forecast Plot')
    plt.show()


# Define Colombian public holidays for 2024
colombia_holidays = pd.DataFrame({
    'holiday': [
        'New Year\'s Day', 'Epiphany', 'Saint Joseph\'s Day', 'Maundy Thursday', 'Good Friday', 'Labour Day',
        'Ascension Day', 'Corpus Christi', 'Sacred Heart', 'Saints Peter and Paul\'s Day',
        'Declaration of Independence',
        'Battle of Boyacá', 'Assumption Day', 'Day of the Races', 'All Saints\' Day', 'Independence of Cartagena',
        'Immaculate Conception Day', 'Christmas Day'
    ],
    'ds': pd.to_datetime([
        '2024-01-01', '2024-01-08', '2024-03-25', '2024-03-28', '2024-03-29', '2024-05-01',
        '2024-05-13', '2024-06-03', '2024-06-10', '2024-07-01', '2024-07-20',
        '2024-08-07', '2024-08-19', '2024-10-14', '2024-11-04', '2024-11-11',
        '2024-12-08', '2024-12-25'
    ]),
    'lower_window': 0,
    'upper_window': 1,
})

# Main code execution
if __name__ == "__main__":
    file_path = 'exported_data/final-final.csv'  # Update this to your actual file path
    try:
        df, all_numbers = load_and_preprocess(file_path)
        global_max_number = max(all_numbers)
        model = define_and_fit_model(df[['ds', 'y']], colombia_holidays)
        forecast = make_predictions(model, 365, global_max_number)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'yhat_adjusted']].to_csv('forecasted_values.csv',
                                                                                     index=False)
        plot_results(forecast)
        # Optional: Perform cross-validation
        df_cv = cross_validation(model, initial='1095 days', period='180 days', horizon='30 days')
        df_p = performance_metrics(df_cv)
        print(df_p.head())
        fig = plot_cross_validation_metric(df_cv, metric='mae')
    except Exception as e:
        print(f"An error occurred: {e}")
