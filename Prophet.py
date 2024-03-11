from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
from contants import COLOMBIA_HOLIDAYS
from utils.processor import load_and_preprocess, process_and_compare_forecasts


# Function to define and fit the Prophet model
def define_and_fit_model(series):
    m = Prophet(
        holidays=COLOMBIA_HOLIDAYS,
        changepoint_prior_scale=0.05,  # Ajustado para flexibilidad en los cambios de tendencia
        seasonality_prior_scale=10,  # Permitir que el modelo ajuste la estacionalidad más libremente
        n_changepoints=25,  # Aumentado debido a la longitud de la serie temporal y la volatilidad
        weekly_seasonality=False,  # Desactivar la estacionalidad semanal predeterminada para definir una personalizada
        yearly_seasonality='auto',  # Activar la estacionalidad anual para datos que abarcan varios años
        daily_seasonality=False  # Desactivar ya que no tienes datos diarios
    )

    # Añade estacionalidades personalizadas para adaptarse a tus datos específicos
    m.add_seasonality(name='midweek_weekend', period=7,
                      fourier_order=3)  # Estacionalidad semanal personalizada para miércoles y sábados
    m.add_seasonality(name='biweekly', period=14,
                      fourier_order=5)  # Para patrones que podrían repetirse cada dos semanas
    m.add_seasonality(name='yearly', period=365.25,
                      fourier_order=10)  # Estacionalidad anual para capturar patrones que ocurren a lo largo de los

    m.fit(series)
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


# Function to perform cross-validation and calculate performance metrics
def evaluate_model_performance(model, initial, period, horizon):
    # Perform cross-validation
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)

    # Calculate performance metrics
    df_p = performance_metrics(df_cv)

    return df_cv, df_p  # Return both the cross-validation results and performance metrics


# Main code execution
if __name__ == "__main__":
    file_path = 'exported_data/final-final.csv'  # Path to your historical data
    actual_2024_file_path = 'exported_data/exported_data_2024.csv'  # Path to your actual 2024 results

    try:
        # Load and preprocess historical data for predictions
        df, balls_expanded = load_and_preprocess(file_path)
        all_predictions = []  # Will store all adjusted predictions

        # Iterate over each series of numbers (column in balls_expanded)
        for i in range(balls_expanded.shape[1]):
            # Create a new DataFrame for this specific number
            temp_df = df[['ds']].copy()
            temp_df['y'] = balls_expanded[i]
            temp_df = temp_df.dropna()  # Remove rows where 'y' could be NaN

            # Define and train the Prophet model for this specific series
            model = define_and_fit_model(temp_df)

            # Evaluate model performance using cross-validation
            initial = '730 days'  # Adjust based on your dataset size and time frame
            period = '90 days'  # Adjust as needed
            horizon = '60 days'  # Forecast horizon for evaluation
            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
            df_p = performance_metrics(df_cv)
            print(df_p.head())  # Print the performance metrics for review
            # df_p.to_csv(f'performance_metrics_{i}.csv', index=False)  # Save performance metrics for each ball series

            # Make future predictions and add them to the compilation for final analysis
            forecast = make_predictions(model, 120,
                                        16 if i == balls_expanded.shape[1] - 1 else 43)
            all_predictions.append(
                forecast[['ds', 'yhat_adjusted']].rename(columns={'yhat_adjusted': f'yhat_adjusted_{i}'}))

        process_and_compare_forecasts(all_predictions, actual_2024_file_path, "prophet_results/")

    except Exception as e:
        print(f"An error occurred: {e}")
