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
    df.rename(columns={'Date': 'ds'}, inplace=True)
    # Splitting 'Ball' into multiple rows
    balls_expanded = df['Ball'].str.split('-', expand=True).apply(pd.to_numeric)
    return df, balls_expanded


# Function to load the actual 2024 data
def load_actual_2024_data(path):
    actual_df = pd.read_csv(path)
    actual_df['Date'] = pd.to_datetime(actual_df['Date'], dayfirst=True)
    actual_df.rename(columns={'Date': 'ds', 'Ball Number': 'numbers'}, inplace=True)
    return actual_df


def compare_numbers_by_date(actual_df, predicted_df):
    actual_df['ds'] = pd.to_datetime(actual_df['ds'])
    predicted_df['ds'] = pd.to_datetime(predicted_df['ds'])

    # Rename for clarity if necessary, depending on what your actual and predicted DataFrames contain
    merged_df = pd.merge(actual_df, predicted_df, on='ds', how='inner', suffixes=('_actual', '_predicted'))

    # Ensure there is a 'numbers' column after merging; check or add defensive programming
    if 'numbers_actual' not in merged_df or 'numbers_predicted' not in merged_df:
        raise ValueError("Missing 'numbers' in actual or predicted data.")

    merged_df['matches'] = merged_df.apply(lambda row:
                                           len(set(str(row['numbers_actual']).split('-')) &
                                               set(str(row['numbers_predicted']).split('-'))),
                                           axis=1)
    return merged_df


def check_actual_in_past_predictions(actual_df, predicted_df):
    results = []
    for _, actual_row in actual_df.iterrows():
        actual_date = actual_row['ds']
        actual_numbers = set(str(actual_row['numbers_actual']).split('-'))
        past_predictions = predicted_df[predicted_df['ds'] < actual_date]  # Only consider past predictions

        # Initialize a dictionary to keep count of dates with corresponding number of matches
        # Start from 3 as we're skipping 1 and 2 matches
        match_counts = {i: [] for i in range(3, len(actual_numbers) + 1)}  # Assuming max matches can be total actual numbers

        for _, pred_row in past_predictions.iterrows():
            predicted_numbers = set(str(pred_row['numbers_predicted']).split('-'))
            num_matches = len(actual_numbers & predicted_numbers)  # Count of matching numbers

            # If there are three or more matches, append the date to the corresponding list in the dictionary
            if num_matches >= 3:
                match_counts[num_matches].append(pred_row['ds'].strftime('%Y-%m-%d'))  # Formatting date for readability

        # Prepare the result for this row of actual numbers
        result = {
            'actual_date': actual_date.strftime('%Y-%m-%d'),
            'actual_numbers': '-'.join(actual_numbers),
        }
        # Update the result with match counts, only include if list is not empty
        result.update({f'{i}_matches_dates': ', '.join(match_counts[i]) for i in match_counts if match_counts[i]})

        results.append(result)

    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame(results)
    return results_df


# Function to define and fit the Prophet model
def define_and_fit_model(df, holidays):
    m = Prophet(changepoint_prior_scale=0.1, interval_width=0.95, holidays=holidays, n_changepoints=25)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_country_holidays(country_name='CO')
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
    file_path = 'exported_data/final-final.csv'  # Path to your historical data
    actual_2024_file_path = 'exported_data/exported_data_2024.csv'  # Path to your actual 2024 results

    try:
        # Load and preprocess historical data for predictions
        df, balls_expanded = load_and_preprocess(file_path)
        all_forecasts = []  # Will store all adjusted predictions

        # Iterate over each series of numbers (column in balls_expanded)
        for i in range(balls_expanded.shape[1]):
            # Create a new DataFrame for this specific number
            temp_df = df[['ds']].copy()
            temp_df['y'] = balls_expanded[i]
            temp_df = temp_df.dropna()  # Remove rows where 'y' could be NaN

            # Define and train the Prophet model for this specific series
            model = define_and_fit_model(temp_df, colombia_holidays)
            forecast = make_predictions(model, 60, max(balls_expanded[i]))  # 60 days ahead

            # Add the adjusted predictions for final compilation
            all_forecasts.append(
                forecast[['ds', 'yhat_adjusted']].rename(columns={'yhat_adjusted': f'yhat_adjusted_{i}'}))

        # Compile all adjusted predictions into a single DataFrame
        final_combined = pd.concat(all_forecasts, axis=1)
        final_combined = final_combined.loc[:, ~final_combined.columns.duplicated()]  # Remove duplicated 'ds' columns
        final_combined['combined'] = final_combined.filter(like='yhat_adjusted').apply(
            lambda row: '-'.join(row.dropna().astype(str)), axis=1)
        final_combined[['ds', 'combined']].to_csv('final_combined_forecast.csv', index=False)

        # Load and preprocess actual 2024 data
        actual_2024_df = load_actual_2024_data(actual_2024_file_path)

        # Prepare actual data for comparison
        actual_2024_df['numbers'] = actual_2024_df['numbers'].astype(str)  # Ensure numbers are strings for splitting

        if 'numbers' not in actual_2024_df.columns:
            raise Exception("Actual data missing 'numbers' column.")
        if 'combined' not in final_combined.columns:  # Assuming 'combined' is your predicted numbers column
            raise Exception("Predicted data missing 'combined' column.")

        # Convert 'combined' to 'numbers_predicted' if that's what your compare function expects
        final_combined.rename(columns={'combined': 'numbers_predicted'}, inplace=True)
        actual_2024_df.rename(columns={'numbers': 'numbers_actual'}, inplace=True)

        # Compare the actual 2024 numbers with the predicted numbers
        comparison_df = compare_numbers_by_date(actual_2024_df, final_combined)
        comparison_df[["ds", "numbers_actual", "numbers_predicted", "matches"]].to_csv('matched_numbers_by_date_2024.csv', index=False)

        # Check if the actual 2024 numbers appear in any past predictions
        cross_date_matches_df = check_actual_in_past_predictions(actual_2024_df, final_combined)
        cross_date_matches_df.to_csv('actual_in_past_predictions_2024.csv', index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
