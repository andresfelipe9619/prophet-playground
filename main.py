# Import necessary libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# Load the dataset
file_path = 'exported_data/final-final.csv'  # Update this to your actual file path
df = pd.read_csv(file_path)

# We want to find the maximum number across all these lists

# Create a list of all numbers by splitting the 'Ball' column and flattening the list of lists
all_numbers = [int(number) for sublist in df['Ball'].str.split('-') for number in sublist]

# Find the global maximum number from all the lists
global_max_number = max(all_numbers)

# Convert the 'Date' column to datetime format and rename it to 'ds'
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.rename(columns={'Date': 'ds'}, inplace=True)

# Split the 'Ball' column and extract the first number, convert it to numeric type
df['y'] = df['Ball'].apply(lambda x: int(x.split('-')[0]))

# Now, df has the correct format with 'ds' and 'y'
df = df[['ds', 'y']]

# Define Colombian public holidays for 2024
holidays = pd.DataFrame({
  'holiday': [
    'New Year\'s Day', 'Epiphany', 'Saint Joseph\'s Day', 'Maundy Thursday', 'Good Friday', 'Labour Day',
    'Ascension Day', 'Corpus Christi', 'Sacred Heart', 'Saints Peter and Paul\'s Day', 'Declaration of Independence',
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


# Initialize and fit the Prophet model
m = Prophet(changepoint_prior_scale=0.05, interval_width=0.95, holidays=holidays)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

m.fit(df)

# Create a future DataFrame for the next 365 days
future = m.make_future_dataframe(periods=365)

# Use the model to make predictions
forecast = m.predict(future)

# Post-process forecasted values to ensure they fall within the specified range (1 to global_max_number)
forecast['yhat_adjusted'] = forecast['yhat'].apply(lambda x: min(max(1, round(x)), global_max_number))

# Export forecasted values to a CSV file
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'yhat_adjusted']].to_csv('forecasted_values.csv', index=False)

# Plot the original forecast
fig1 = m.plot(forecast)
plt.title('Original Forecast Plot')
plt.show()

# Plot the adjusted forecast
plt.figure(figsize=(10, 6))
plt.plot(forecast['ds'], forecast['yhat_adjusted'], label='Adjusted Forecast', color='orange')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='gray')
plt.legend()
plt.title('Adjusted Forecast Plot')
plt.show()

# (Optional) Perform cross-validation
# Adjust 'initial', 'period', and 'horizon' based on your dataset size and needs
df_cv = cross_validation(m, initial='1095 days', period='180 days', horizon='30 days')  # 3 years initial, sliding
# every 6 months, for a 1-year horizon

# Calculate and display performance metrics
df_p = performance_metrics(df_cv)
print(df_p.head())

fig = plot_cross_validation_metric(df_cv, metric='mae')
