## Notes
**Data Preparation:** Make sure your actual dataset is formatted similarly to the data dictionary, with ds and y columns for dates and values, respectively.

**Model Fitting:** This script fits the model to your historical data.

**Future Predictions:** The script forecasts the next 365 days based on the model.

**Visualization:** It plots the forecast and its components.

**Cross-validation:** This is optional and useful for evaluating your model's performance. Adjust the parameters (initial, period, horizon) based on the size of your dataset. You need a substantial amount of historical data to use this feature effectively.

**Performance Metrics:** The script prints the head of the performance metrics DataFrame, showing errors like MAE and RMSE, which help in evaluating the forecast's accuracy.

This code is a comprehensive start for forecasting with Prophet. Depending on your specific needs or data characteristics, you might want to explore additional features of Prophet, such as incorporating holidays, adjusting seasonality, or adding regressors.

## Retraining From Scratch
The straightforward method is to retrain your model including the new data. This approach ensures that your model is up-to-date and incorporates all available information. Here's how you can approach it:

**Combine Your Data:** Each day (or at whatever frequency new data becomes available), you append the new observations to your existing dataset.

**Retrain the Model:** Use the updated dataset to fit a new Prophet model. This involves running the same code you used initially to fit the model, but on the updated dataset.

**Forecast:** Make new forecasts using the retrained model.
This method is simple and effective, ensuring your model's forecasts are as accurate as possible with the latest data. However, it might be computationally expensive for very large datasets or if you need to update your forecasts very frequently.

### Incremental Updating (Not Directly Supported)
While incremental learning is a concept where models update themselves with new data without needing to be retrained from scratch, Prophet does not support this directly due to its underlying statistical methodologies. However, for some models, especially those dealing with large datasets or requiring frequent updates, exploring models or systems designed with incremental learning in mind might be beneficial.

### Best Practices for Retraining
**Automate the Process:** Automate the data appending, model retraining, and forecasting processes as much as possible to save time and reduce errors. This can be done using scheduled scripts or workflows.
**Monitor Performance:** Regularly evaluate the performance of your model against recent historical data to ensure its predictions remain accurate over time. Adjust your model as needed based on these performance metrics.
**Consider Data Changes:** Be mindful of any significant changes in your data or in the external environment that could impact your model's assumptions. This might include changes in trends, seasonality, or the impact of external events. In such cases, you may need to adjust your model beyond simply retraining with new data.

### Conclusion
In summary, while Prophet does not learn incrementally in a strict sense, retraining the model periodically with the full dataset is a practical and effective way to keep your forecasts up-to-date. Depending on the frequency at which your data updates and the computational resources available, you can determine the best retraining schedule for your needs, whether that's daily, weekly, or at another regular interval.