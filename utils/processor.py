import os
import pandas as pd


# Función para cargar y preprocesar datos
def load_and_preprocess(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    df = pd.read_csv(path)
    # Validar columnas esperadas
    expected_columns = ['Ball', 'Date']
    if not all(column in df.columns for column in expected_columns):
        missing = list(set(expected_columns) - set(df.columns))
        raise ValueError(f"Missing columns in the dataset: {missing}")
    # Preprocesamiento de datos
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.rename(columns={'Date': 'ds'}, inplace=True)
    # Dividir 'Ball' en múltiples filas
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
        # Now including keys for 5 and 6 matches
        match_counts = {i: [] for i in range(3, 7)}  # Adjust range up to 7 to include checks for 5 and 6 matches

        for _, pred_row in past_predictions.iterrows():
            predicted_numbers = set(str(pred_row['numbers_predicted']).split('-'))
            num_matches = len(actual_numbers & predicted_numbers)  # Count of matching numbers

            # Append the date to the corresponding list in the dictionary if there are three or more matches
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


def process_and_compare_forecasts(all_forecasts, actual_2024_file_path, prefix=""):
    # Compile all adjusted predictions into a single DataFrame
    final_combined = pd.concat(all_forecasts, axis=1)
    final_combined = final_combined.loc[:, ~final_combined.columns.duplicated()]  # Remove duplicated 'ds' columns
    final_combined['combined'] = final_combined.filter(like='yhat_adjusted').apply(
        lambda row: '-'.join(row.dropna().astype(str)), axis=1)
    final_combined[['ds', 'combined']].to_csv(prefix + 'final_combined_forecast.csv', index=False)

    # Load and preprocess actual 2024 data
    actual_2024_df = load_actual_2024_data(actual_2024_file_path)

    # Prepare actual data for comparison
    actual_2024_df['numbers'] = actual_2024_df['numbers'].astype(str)  # Ensure numbers are strings for splitting

    if 'numbers' not in actual_2024_df.columns:
        raise Exception("Actual data missing 'numbers' column.")
    if 'combined' not in final_combined.columns:  # Assuming 'combined' is your predicted numbers column
        raise Exception("Predicted data missing 'combined' column.")

    # Convert 'combined' to 'numbers_predicted' and 'numbers' to 'numbers_actual'
    final_combined.rename(columns={'combined': 'numbers_predicted'}, inplace=True)
    actual_2024_df.rename(columns={'numbers': 'numbers_actual'}, inplace=True)

    # Compare the actual 2024 numbers with the predicted numbers
    comparison_df = compare_numbers_by_date(actual_2024_df, final_combined)
    comparison_df[["ds", "numbers_actual", "numbers_predicted", "matches"]].to_csv(
        prefix + 'matched_numbers_by_date_2024.csv', index=False)

    # Check if the actual 2024 numbers appear in any past predictions
    cross_date_matches_df = check_actual_in_past_predictions(actual_2024_df, final_combined)
    cross_date_matches_df.to_csv(prefix + 'actual_in_past_predictions_2024.csv', index=False)

    return comparison_df, cross_date_matches_df  # Optionally return resulting DataFrames
