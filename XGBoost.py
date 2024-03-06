import xgboost as xgb
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from utils.processor import load_and_preprocess, process_and_compare_forecasts


# Function to create features for XGBoost
def create_features(df):
    """
    Create time series features based on time index.
    """
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week  # Changed line to fix warning
    return df


# Function to train XGBoost model
def xgb_forecast(train, test, y_label):
    # Exclude 'ds' column for XGBoost training and prediction
    train_x = train.drop(columns=[y_label, 'ds'])  # Removed 'ds' from training data
    train_y = train[y_label]
    test_x = test.drop(columns=['ds'])  # Removed 'ds' from test data

    # Create DMatrix for xgboost
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)

    # Set xgboost parameters
    params = {'max_depth': 6, 'eta': 0.1, 'objective': 'reg:squarederror'}
    num_round = 100  # number of boosting rounds

    # Train xgboost model
    bst = xgb.train(params, dtrain, num_round)

    # Make prediction
    preds = bst.predict(dtest)
    test['yhat'] = preds  # Attach predictions back to the original test dataset
    return test


# Function for plotting results
def plot_results(original, forecast):
    plt.figure(figsize=(14, 7))
    plt.plot(original['ds'], original['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', alpha=0.7)
    plt.legend()
    plt.title('XGBoost Forecast vs Actuals')
    plt.show()


if __name__ == "__main__":
    file_path = 'exported_data/final-final.csv'
    actual_2024_file_path = 'exported_data/exported_data_2024.csv'

    try:
        # Load and preprocess historical data for predictions
        df, balls_expanded = load_and_preprocess(file_path)
        all_predictions = []  # Will store all XGBoost predictions

        # Iterate over each series of numbers (column in balls_expanded)
        for i in range(balls_expanded.shape[1]):
            # Create a new DataFrame for this specific number
            temp_df = df[['ds']].copy()
            temp_df['y'] = balls_expanded[i]
            temp_df = temp_df.dropna()  # Remove rows where 'y' could be NaN
            temp_df = create_features(temp_df)  # Add time series features

            # Split data into training and testing for validation
            train_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=42)

            # Train the XGBoost model
            forecast = xgb_forecast(train_df, test_df, 'y')

            # Store forecasted results
            forecast = forecast[['ds', 'yhat']]  # Keep only necessary columns
            all_predictions.append(
                forecast.rename(columns={'yhat': f'yhat_adjusted_{i}'}))

        # Here you could compare forecasts with actual results as in your original code
        process_and_compare_forecasts(all_predictions, actual_2024_file_path)

    except Exception as e:
        print(f"An error occurred: {e}")
