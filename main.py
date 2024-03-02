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
                   'Ascension Day', 'Corpus Christi', 'Sacred Heart', 'Saint Peter and Saint Paul',
                   'Declaration of Independence',
                   'Battle of Boyacá', 'Assumption of Mary', 'Day of the Race', 'All Saints’ Day',
                   'Independence of Cartagena',
                   'Immaculate Conception', 'Christmas Day',
                   # Repeat for each year as needed
               ] * 8,  # Repeated for 8 years
    'ds': pd.to_datetime([
        # 2017
        '2017-01-01', '2017-01-09', '2017-03-20', '2017-04-13', '2017-04-14', '2017-05-01',
        '2017-05-29', '2017-06-19', '2017-06-23', '2017-07-03', '2017-07-20',
        '2017-08-07', '2017-08-21', '2017-10-16', '2017-11-06', '2017-11-13',
        '2017-12-08', '2017-12-25',
        # 2018
        '2018-01-01', '2018-01-08', '2018-03-19', '2018-03-29', '2018-03-30', '2018-05-01',
        '2018-05-14', '2018-06-04', '2018-06-08', '2018-07-02', '2018-07-20',
        '2018-08-07', '2018-08-20', '2018-10-15', '2018-11-05', '2018-11-12',
        '2018-12-08', '2018-12-25',
        # 2019
        '2019-01-01', '2019-01-07', '2019-03-25', '2019-04-18', '2019-04-19', '2019-05-01',
        '2019-05-27', '2019-06-17', '2019-06-21', '2019-07-01', '2019-07-22',
        '2019-08-07', '2019-08-19', '2019-10-14', '2019-11-04', '2019-11-11',
        '2019-12-08', '2019-12-25',
        # 2020
        '2020-01-01', '2020-01-06', '2020-03-23', '2020-04-09', '2020-04-10', '2020-05-01',
        '2020-05-25', '2020-06-15', '2020-06-19', '2020-07-03', '2020-07-20',
        '2020-08-07', '2020-08-17', '2020-10-12', '2020-11-02', '2020-11-09',
        '2020-12-08', '2020-12-25',
        # 2021
        '2021-01-01', '2021-01-04', '2021-03-29', '2021-04-01', '2021-04-02', '2021-05-01',
        '2021-05-31', '2021-06-21', '2021-06-25', '2021-07-05', '2021-07-19',
        '2021-08-09', '2021-08-16', '2021-10-11', '2021-11-01', '2021-11-08',
        '2021-12-08', '2021-12-25',
        # 2022
        '2022-01-01', '2022-01-10', '2022-04-11', '2022-04-14', '2022-04-15', '2022-05-01',
        '2022-05-30', '2022-06-20', '2022-06-24', '2022-07-04', '2022-07-18',
        '2022-08-08', '2022-08-15', '2022-10-10', '2022-11-07', '2022-11-14',
        '2022-12-08', '2022-12-25',
        # 2023
        '2023-01-01', '2023-01-09', '2023-04-03', '2023-04-06', '2023-04-07', '2023-05-01',
        '2023-05-29', '2023-06-19', '2023-06-23', '2023-07-03', '2023-07-17',
        '2023-08-07', '2023-08-21', '2023-10-09', '2023-11-06', '2023-11-13',
        '2023-12-08', '2023-12-25',

        # 2024
        '2024-01-01', '2024-01-08', '2024-03-25', '2024-03-28', '2024-03-29', '2024-05-01',
        '2024-05-13', '2024-06-03', '2024-06-10', '2024-07-01', '2024-07-20',
        '2024-08-07', '2024-08-19', '2024-10-14', '2024-11-04', '2024-11-11',
        '2024-12-08', '2024-12-25'
    ])
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
