import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Constants
FIRST_FIVE_BALLS_RANGE = range(1, 44)
LAST_BALL_RANGE = range(1, 17)
DATE_FORMAT = '%d/%m/%Y'

# Utility functions for lottery draws
def extract_numbers(ball_str, position='first'):
    """Extract numbers from a lottery draw string."""
    numbers = list(map(int, ball_str.split('-')))
    return numbers[:5] if position == 'first' else [numbers[-1]]

def calculate_date_differences(dates):
    """Calculate differences between consecutive dates."""
    sorted_dates = sorted(dates)
    return [(sorted_dates[i] - sorted_dates[i - 1]).days for i in range(1, len(sorted_dates))]

def create_summary_dataframe(draws_data, column_name):
    """Create a summary DataFrame for lottery draw data."""
    summary_data = []
    number_range = FIRST_FIVE_BALLS_RANGE if column_name == 'FirstFive' else LAST_BALL_RANGE
    for number in number_range:
        relevant_draws = draws_data[draws_data[column_name].apply(lambda x: number in x)]
        repetitions = len(relevant_draws)
        last_date = relevant_draws['Date'].max()
        date_differences = calculate_date_differences(relevant_draws['Date'])
        summary_data.append({
            'Number': number,
            'Repetitions': repetitions,
            'Last Date': last_date,
            'Average Days': None if not date_differences else sum(date_differences) / len(date_differences),
            'Min Days': None if not date_differences else min(date_differences),
            'Max Days': None if not date_differences else max(date_differences),
            'Std Dev Days': None if not date_differences else pd.Series(date_differences).std(),
            'Expected Date': None if last_date is None or not date_differences else last_date + timedelta(days=round(sum(date_differences) / len(date_differences)))
        })
    return pd.DataFrame(summary_data)

# Data processing
draws_data = pd.read_csv('exported_data/final-2023.csv')
draws_data['Date'] = pd.to_datetime(draws_data['Date'], format=DATE_FORMAT)
draws_data['FirstFive'] = draws_data['Ball'].apply(lambda x: extract_numbers(x, 'first'))
draws_data['Last'] = draws_data['Ball'].apply(lambda x: extract_numbers(x, 'last'))

# Generate summary data
summary_df_first_five = create_summary_dataframe(draws_data, 'FirstFive')
summary_df_last = create_summary_dataframe(draws_data, 'Last')

# Save to CSV
csv_file_path_first_five = 'summary/lottery_ball_summary_first_five.csv'
csv_file_path_last = 'summary/lottery_ball_summary_last.csv'
summary_df_first_five.to_csv(csv_file_path_first_five, index=False)
summary_df_last.to_csv(csv_file_path_last, index=False)
print(f"CSV file for the first five balls saved: {csv_file_path_first_five}")
print(f"CSV file for the last ball saved: {csv_file_path_last}")


def plot_frequency_side_by_side(summary_df_first_five, summary_df_last, title1, title2, x_label, y_label, fig_size=(28, 8)):
    """Plot frequency of first five and last lottery numbers side by side with solid colors."""
    fig, axes = plt.subplots(1, 2, figsize=fig_size)  # 1 fila, 2 columnas
    
    # Asignar una columna constante para usar en 'hue'
    summary_df_first_five['Color'] = 'FirstFive'
    summary_df_last['Color'] = 'Last'

    # Gráfico para los primeros cinco números
    sns.barplot(ax=axes[0], data=summary_df_first_five, x='Number', y='Repetitions', hue='Color', dodge=False, palette=['royalblue'])
    axes[0].set_title(title1, fontsize=18)
    axes[0].set_xlabel(x_label, fontsize=15)
    axes[0].set_ylabel(y_label, fontsize=15)
    axes[0].tick_params(labelsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend([],[], frameon=False)  # Ocultar leyenda

    # Gráfico para el último número
    sns.barplot(ax=axes[1], data=summary_df_last, x='Number', y='Repetitions', hue='Color', dodge=False, palette=['royalblue'])
    axes[1].set_title(title2, fontsize=18)
    axes[1].set_xlabel(x_label, fontsize=15)
    axes[1].set_ylabel(y_label, fontsize=15)
    axes[1].tick_params(labelsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend([],[], frameon=False)  # Ocultar leyenda

    plt.tight_layout()
    plt.show()

# Llamar a la función para crear los gráficos lado a lado
plot_frequency_side_by_side(summary_df_first_five, summary_df_last, 'Frequency of First Five Balls', 'Frequency of Last Ball', 'Number', 'Repetitions')

# Análisis de Intervalos (Histogram) para el último número
plt.figure(figsize=(12, 6))
plt.title('Gap Analysis for the Last Ball')
sns.histplot(data=summary_df_last, x='Average Days', bins=15, kde=True)  # Menos bins si hay menos datos
plt.show()

# Boxplots para Análisis de Intervalos
plt.figure(figsize=(14, 8))
plt.title('Gap Analysis for the First Five Balls with Boxplots', fontsize=18)
sns.boxplot(data=summary_df_first_five, x='Number', y='Average Days', palette='Spectral', hue='Number', legend=False)
plt.xlabel('Number', fontsize=15)
plt.ylabel('Average Days Between Draws', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Gráficos de Distribución Combinados
plt.figure(figsize=(14, 8))
plt.title('Combined Distribution Analysis for First Five Balls', fontsize=18)
sns.histplot(data=summary_df_first_five, x='Average Days', bins=30, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Average Days Between Draws', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Gráficos de barras para Min Days, Max Days y Average Days para los primeros cinco números
plt.figure(figsize=(18, 12))

# Min Days Between Draws for First Five
plt.subplot(3, 1, 1)  # 3 filas, 1 columna, posición 1
sns.barplot(data=summary_df_first_five, x='Number', y='Min Days', palette='Blues', hue='Number', legend=False)
plt.title('Minimum Days Between Draws for First Five Balls', fontsize=16)
plt.xlabel('Number', fontsize=14)
plt.ylabel('Min Days', fontsize=14)

# Max Days Between Draws for First Five
plt.subplot(3, 1, 2)  # 3 filas, 1 columna, posición 2
sns.barplot(data=summary_df_first_five, x='Number', y='Max Days', hue='Number', legend=False, palette='Reds')
plt.title('Maximum Days Between Draws for First Five Balls', fontsize=16)
plt.xlabel('Number', fontsize=14)
plt.ylabel('Max Days', fontsize=14)

# Std Dev Days Between Draws for First Five
plt.subplot(3, 1, 3)  # 3 filas, 1 columna, posición 3
sns.barplot(data=summary_df_first_five, x='Number', y='Std Dev Days', hue='Number', palette='Purples', legend=False)
plt.title('Standard Deviation of Days Between Draws for First Five Balls', fontsize=16)
plt.xlabel('Number', fontsize=14)
plt.ylabel('Std Dev Days', fontsize=14)

plt.tight_layout()
plt.legend([], [], frameon=False)
plt.show()

# Gráficos combinados de días entre sorteos
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)  # subplot para los primeros cinco números
sns.violinplot(data=summary_df_first_five, x='Number', y='Average Days', palette='viridis', hue='Number', legend=False)
plt.title('Distribution of Average Days Between Draws for First Five Balls')
plt.xlabel('Number')
plt.ylabel('Average Days Between Draws')

plt.subplot(1, 2, 2)  # subplot para el último número
sns.violinplot(data=summary_df_last, x='Number', y='Average Days', palette='viridis', hue='Number', legend=False)
plt.title('Distribution of Average Days Between Draws for Last Ball')
plt.xlabel('Number')
plt.ylabel('Average Days Between Draws')

plt.tight_layout()
plt.show()

# Visualización de datos mejorada
plt.figure(figsize=(18, 6))

# Histogramas de frecuencia mejorados con densidad
plt.subplot(1, 2, 1)
sns.histplot(summary_df_first_five, x="Number", weights="Repetitions", bins=44, kde=True, color='blue')
plt.title('Density and Frequency of First Five Numbers')
plt.xlabel('Number')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(summary_df_last, x="Number", weights="Repetitions", bins=16, kde=True, color='red')
plt.title('Density and Frequency of Last Number')
plt.xlabel('Number')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Mapa de calor para la variación en días entre sorteos (Ejemplo)
# Deberías calcular esto con datos reales
plt.figure(figsize=(12, 9))
sns.heatmap(summary_df_first_five.pivot("Number", "Last Date", "Average Days"), cmap="YlGnBu", annot=True, fmt=".0f")
plt.title('Heatmap of Average Days Between Draws for First Five Numbers')
plt.xlabel('Last Date of Draw')
plt.ylabel('Number')
plt.show()

# Asumiendo que 'draws_data' tiene una columna 'Date' con las fechas de los sorteos
plt.figure(figsize=(15, 5))
# Solo mostramos los 50 sorteos más recientes para evitar sobrecargar el gráfico
last_draws = draws_data.sort_values('Date', ascending=False).head(50)
plt.plot_date(last_draws['Date'], last_draws.index, linestyle='solid', fmt="o")  # Cambiado para evitar la advertencia
plt.title('Timeline of the Last 50 Lottery Draws')
plt.xlabel('Date')
plt.ylabel('Draw Number')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate()  # Rotación de fechas para mejor lectura
plt.grid(True)
plt.show()

# Agrupando los datos por mes y año para visualización
draws_data['Year'] = draws_data['Date'].dt.year
draws_data['Month'] = draws_data['Date'].dt.month_name()
monthly_counts = draws_data.groupby(['Year', 'Month']).size().unstack(fill_value=0)

# Visualización de los sorteos por mes en cada año
plt.figure(figsize=(15, 8))
monthly_counts.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Monthly Lottery Draws Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Draws')
plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Creación de nuevas columnas para día de la semana y mes
draws_data['Weekday'] = draws_data['Date'].dt.day_name()
draws_data['Month'] = draws_data['Date'].dt.month

# Creando una tabla pivot para visualizar los conteos de sorteos
pivot_table = draws_data.pivot_table(index='Weekday', columns='Month', aggfunc='size', fill_value=0)

# Ordenar los días de la semana de lunes a domingo
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_table = pivot_table.reindex(weekday_order)

# Heatmap de sorteos
plt.figure(figsize=(12, 7))
sns.heatmap(summary_df_first_five.pivot(index="Number", columns="Last Date", values="Average Days"), cmap="YlGnBu", annot=True, fmt=".0f")
plt.title('Lottery Draws Heatmap by Weekday and Month')
plt.xlabel('Month')
plt.ylabel('Day of the Week')
plt.show()