"""Owner of the data contract: Date (dd/mm/yyyy) + Ball (dash-separated, superbalota last).

Every entry point parses draws through `preprocess_draws` — the CSV loader
below, the dashboard's upload path, and the synthetic sample data — so the
format cannot drift between them.
"""

import os
import warnings

import pandas as pd

from models.common import (
    MAIN_BALLS_DRAWN,
    MAIN_BALL_RANGE,
    SUPER_BALL_RANGE,
    main_positions,
    super_position,
)


def format_violations(balls_expanded):
    """Boolean Series: rows whose numbers cannot have come from the current game.

    Baloto changed shape in April 2017 — the old game drew 6 balls from 1-45
    with no superbalota, and both eras are still published as six
    dash-separated numbers, so a merged file has the same column count
    throughout and nothing about its *shape* gives the mix away. Only the
    values do:

    - a main ball above 43, or the last ball above 16 (impossible now, routine
      in the old game, where the sixth number ran to 45);
    - a repeated main ball (the five are drawn without replacement, but the
      superbalota is independent, so it may legitimately equal one of them).

    This flags rows that are *provably* not current-format. It cannot flag an
    old-era draw whose numbers happen to fit the current bounds — for that use
    `current_format_mask`, which cuts at the era boundary instead.
    """
    n_columns = balls_expanded.shape[1]
    if n_columns != MAIN_BALLS_DRAWN + 1:
        return pd.Series(False, index=balls_expanded.index)

    mains = balls_expanded[list(main_positions(n_columns))]
    supers = balls_expanded[super_position(n_columns)]

    out_of_range_main = ((mains < MAIN_BALL_RANGE[0]) | (mains > MAIN_BALL_RANGE[1])).any(axis=1)
    out_of_range_super = (supers < SUPER_BALL_RANGE[0]) | (supers > SUPER_BALL_RANGE[1])
    repeated_main = mains.nunique(axis=1) < MAIN_BALLS_DRAWN

    return out_of_range_main | out_of_range_super | repeated_main


def current_format_mask(df, balls_expanded):
    """Boolean Series: draws that belong to the current era of the game.

    A row is kept only if it is itself well formed *and* dated after the last
    provable violation. The second condition is the important one: the era
    boundary is a date, not a per-row property, so every draw up to the last
    impossible one is discarded even where its numbers would pass on their own.
    Keeping those would leave a tail of old-game draws mixed into the history,
    which is precisely the contamination this exists to remove.
    """
    violations = format_violations(balls_expanded)
    if not violations.any():
        return pd.Series(True, index=balls_expanded.index)

    last_bad_date = df.loc[violations.reindex(df.index, fill_value=False), "ds"].max()
    return (df["ds"] > last_bad_date) & ~violations


def check_draw_format(df, balls_expanded):
    """Report on how much of a loaded history is not current-format Baloto.

    Returns counts, the era boundary and a ready-to-print message, or None
    when everything checks out.
    """
    violations = format_violations(balls_expanded)
    n_bad = int(violations.sum())
    if not n_bad:
        return None

    keep = current_format_mask(df, balls_expanded)
    bad_dates = df.loc[violations.reindex(df.index, fill_value=False), "ds"]
    return {
        "n_draws": len(df),
        "n_violations": n_bad,
        "n_current_format": int(keep.sum()),
        "n_dropped_by_cutoff": int((~keep).sum()),
        "first_violation": bad_dates.min(),
        "last_violation": bad_dates.max(),
        "current_era_starts": df.loc[keep, "ds"].min() if keep.any() else None,
        "message": (
            f"{n_bad} of {len(df)} draws contain numbers the current game cannot produce "
            f"(main ball > {MAIN_BALL_RANGE[1]}, superbalota > {SUPER_BALL_RANGE[1]}, or a repeated "
            f"main ball), between {bad_dates.min():%Y-%m-%d} and {bad_dates.max():%Y-%m-%d}. This "
            "history most likely mixes the pre-2017 game (6 balls from 1-45, no superbalota) with the "
            f"current one. Only {int(keep.sum())} draws are current-format; analysing the rest together "
            "mixes two different games. Filter with utils.processor.current_format_mask()."
        ),
    }


def preprocess_draws(df, validate=True):
    expected_columns = ['Ball', 'Date']
    if not all(column in df.columns for column in expected_columns):
        missing = list(set(expected_columns) - set(df.columns))
        raise ValueError(f"Missing columns in the dataset: {missing}")

    df = df.copy()
    df['ds'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.drop(columns=['Date'], inplace=True)
    balls_expanded = df['Ball'].str.split('-', expand=True).apply(pd.to_numeric)

    # Warn rather than raise: the rows are real draws, just from a different
    # game, and a caller may deliberately want the full history. Silence is the
    # one option ruled out — mixed eras corrupt every downstream statistic
    # without changing anything visible about the frames.
    if validate:
        report = check_draw_format(df, balls_expanded)
        if report:
            warnings.warn(report["message"], stacklevel=2)

    return df, balls_expanded


def load_and_preprocess(path, validate=True, current_format_only=False):
    """Load the CSV and parse it into (df, balls_expanded).

    `current_format_only=True` drops everything before the current era of the
    game (see current_format_mask) — use it when the file spans the 2017 rule
    change and the analysis assumes today's rules, which all of them do.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    df, balls_expanded = preprocess_draws(pd.read_csv(path), validate=validate)
    if current_format_only:
        keep = current_format_mask(df, balls_expanded)
        df = df[keep].reset_index(drop=True)
        balls_expanded = balls_expanded[keep].reset_index(drop=True)
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


def process_and_compare_forecasts(all_forecasts, actual_2024_file_path, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)

    # Compile all adjusted predictions into a single DataFrame
    final_combined = pd.concat(all_forecasts, axis=1)
    final_combined = final_combined.loc[:, ~final_combined.columns.duplicated()]  # Remove duplicated 'ds' columns
    final_combined['combined'] = final_combined.filter(like='yhat_adjusted').apply(
        lambda row: '-'.join(row.dropna().astype(str)), axis=1)
    final_combined[['ds', 'combined']].to_csv(os.path.join(out_dir, 'final_combined_forecast.csv'), index=False)

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
        os.path.join(out_dir, 'matched_numbers_by_date_2024.csv'), index=False)

    # Check if the actual 2024 numbers appear in any past predictions
    cross_date_matches_df = check_actual_in_past_predictions(actual_2024_df, final_combined)
    cross_date_matches_df.to_csv(os.path.join(out_dir, 'actual_in_past_predictions_2024.csv'), index=False)

    return comparison_df, cross_date_matches_df  # Optionally return resulting DataFrames
