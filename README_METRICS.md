# Prophet Model Performance Metrics

This document explains the performance metrics generated by the Prophet forecasting model used in our time series analysis.

## Metrics Overview

The performance of the Prophet model is evaluated using several standard metrics. Understanding these metrics helps in assessing the accuracy and reliability of the forecasts produced by the model.

### Horizon

- **Description**: The forecast horizon for the performance metric, indicating how far into the future the predictions were made.
- **Interpretation**: Used as a reference for the other metrics, not a performance indicator by itself.

### MSE (Mean Squared Error)

- **Description**: The average of the squares of the errors between forecasted and actual values.
- **Interpretation**: Lower values indicate better model performance. A lower MSE means the forecasted values are closer to the actual values.

### RMSE (Root Mean Squared Error)

- **Description**: The square root of the MSE, providing error terms in the same units as the forecasted values.
- **Interpretation**: Like MSE, a lower RMSE is better. It's more interpretable as it is in the same units as the data.

### MAE (Mean Absolute Error)

- **Description**: The average of the absolute differences between the forecasted and actual values.
- **Interpretation**: Lower values are better. Provides an idea of how big the errors are expected to be.

### MAPE (Mean Absolute Percentage Error)

- **Description**: The average of the absolute percentage errors between the forecasted and actual values.
- **Interpretation**: Lower values are better. Useful for understanding the magnitude of prediction errors in percentage terms.

### MDAPE (Median Absolute Percentage Error)

- **Description**: The median of the absolute percentage errors between the forecasted and actual values.
- **Interpretation**: More robust to outliers than MAPE. Lower values indicate better predictive accuracy.

### SMAPE (Symmetric Mean Absolute Percentage Error)

- **Description**: An adjustment to MAPE that handles cases where the actual values are zero or close to zero better.
- **Interpretation**: Lower values are better. Provides a symmetric measure of the percentage error.

### Coverage

- **Description**: The percentage of actual values that fall within the predictive intervals.
- **Interpretation**: A value close to the confidence interval of the predictive intervals (e.g., 80% for an 80% interval) indicates good model calibration.

## Usage

These metrics are used to evaluate the performance of the Prophet forecasting model. By analyzing these metrics, one can understand the accuracy, reliability, and calibration of the model's forecasts. Lower values for MSE, RMSE, MAE, MAPE, MDAPE, and SMAPE indicate more accurate forecasts, while the Coverage metric should be close to the chosen confidence level of the predictive intervals.

For example, if you were using 95% predictive intervals and your coverage was 0.8, it would suggest that 80% of your actual data points fell within these 95% intervals. Ideally, for well-calibrated forecast intervals, you would want the coverage to be close to the confidence level of the intervals — in this case, close to 95%. If the coverage is significantly lower (like 80% in this case), it might indicate that your predictive intervals are too narrow or the model is underestimating the uncertainty of the forecasts.

On the other hand, if the coverage is much higher than the confidence level, it might suggest that your intervals are too wide, meaning your model might be overestimating uncertainty. In practice, finding the right balance and understanding why your model may be underperforming or overperforming in terms of coverage can help in tuning the model for better accuracy and reliability.