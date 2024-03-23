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
draws_data = pd.read_csv('exported_data/final-final.csv')
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


# Función general para graficar las distribuciones y frecuencias
def plot_data_side_by_side(df1, df2, title1, title2, x_label, y_label, kind, fig_size=(28, 8), bins=None, palette=['royalblue']):
    """Plot side by side plots for given data."""
    fig, axes = plt.subplots(1, 2, figsize=fig_size)  # 1 fila, 2 columnas
    
    # Asignar una columna constante para usar en 'hue'
    df1['Set'] = 'First Set'
    df2['Set'] = 'Second Set'

    # Configurar el gráfico dependiendo del tipo
    if kind == 'bar':
        sns.barplot(ax=axes[0], data=df1, x='Number', y='Repetitions', hue='Set', dodge=False, palette=palette)
        sns.barplot(ax=axes[1], data=df2, x='Number', y='Repetitions', hue='Set', dodge=False, palette=palette)
    elif kind == 'hist':
        sns.histplot(ax=axes[0], data=df1, x=x_label, bins=bins, kde=True, color=palette[0])
        sns.histplot(ax=axes[1], data=df2, x=x_label, bins=bins, kde=True, color=palette[0])
    elif kind == 'box':
        sns.boxplot(ax=axes[0], data=df1, x='Number', y=y_label, palette=palette)
        sns.boxplot(ax=axes[1], data=df2, x='Number', y=y_label, palette=palette)
    elif kind == 'violin':
        sns.violinplot(ax=axes[0], data=df1, x='Number', y=y_label, palette=palette)
        sns.violinplot(ax=axes[1], data=df2, x='Number', y=y_label, palette=palette)

    # Configurar títulos y etiquetas para ambos gráficos
    axes[0].set_title(title1, fontsize=18)
    axes[0].set_xlabel(x_label, fontsize=15)
    axes[0].set_ylabel(y_label, fontsize=15)
    axes[0].tick_params(labelsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend([],[], frameon=False)  # Ocultar leyenda para gráficos de barras

    axes[1].set_title(title2, fontsize=18)
    axes[1].set_xlabel(x_label, fontsize=15)
    axes[1].set_ylabel(y_label, fontsize=15)
    axes[1].tick_params(labelsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend([],[], frameon=False)  # Ocultar leyenda para gráficos de barras

    plt.tight_layout()
    plt.show()

def format_heatmap_dates(ax, dates):
    """Format the dates on the heatmap axis."""
    # Intentar convertir las etiquetas de fecha de string a datetime, luego formatear a string de nuevo
    date_format = "%d-%m-%Y"
    try:
        new_labels = [pd.to_datetime(date).strftime(date_format) for date in dates]
        ax.set_xticklabels(new_labels, rotation=45)  # Rota las etiquetas para mejor legibilidad
    except ValueError:
        # Si la conversión falla, mantén las etiquetas originales pero aún rota para mejor legibilidad
        ax.set_xticklabels(dates, rotation=45)

def plot_heatmaps_side_by_side(df1, df2, title1, title2, fig_size=(28, 10), cmap='YlGnBu'):
    """Plot side by side heatmaps for given data."""
    fig, axes = plt.subplots(1, 2, figsize=fig_size)  # 1 fila, 2 columnas
    
    # Preparar los datos para los heatmaps
    data1 = df1.pivot('Number', 'Last Date', 'Average Days')
    data2 = df2.pivot('Number', 'Last Date', 'Average Days')
    
    # Heatmap para el primer conjunto de datos
    sns.heatmap(data=data1, ax=axes[0], cmap=cmap, annot=True, fmt=".0f")
    axes[0].set_title(title1, fontsize=18)
    format_heatmap_dates(axes[0], data1.columns)  # Formatear fechas
    
    # Heatmap para el segundo conjunto de datos
    sns.heatmap(data=data2, ax=axes[1], cmap=cmap, annot=True, fmt=".0f")
    axes[1].set_title(title2, fontsize=18)
    format_heatmap_dates(axes[1], data2.columns)  # Formatear fechas
    
    plt.tight_layout()
    plt.show()

# Función para graficar Histogramas de frecuencia mejorados con densidad
def plot_density_histograms_side_by_side(df1, df2, title1, title2, x_label, y_label, fig_size=(28, 8), bins=None, color1='blue', color2='red'):
    """Plot side by side density and frequency histograms."""
    fig, axes = plt.subplots(1, 2, figsize=fig_size)  # 1 fila, 2 columnas

    # Histograma para el primer conjunto de datos
    sns.histplot(ax=axes[0], data=df1, x=x_label, weights=y_label, bins=bins, kde=True, color=color1)
    axes[0].set_title(title1, fontsize=18)
    axes[0].set_xlabel('Number', fontsize=15)
    axes[0].set_ylabel('Frequency', fontsize=15)
    axes[0].tick_params(labelsize=12)
    
    # Histograma para el segundo conjunto de datos
    sns.histplot(ax=axes[1], data=df2, x=x_label, weights=y_label, bins=bins, kde=True, color=color2)
    axes[1].set_title(title2, fontsize=18)
    axes[1].set_xlabel('Number', fontsize=15)
    axes[1].set_ylabel('Frequency', fontsize=15)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()


# Frecuencias
plot_data_side_by_side(summary_df_first_five, summary_df_last, 'Frequency of First Five Balls', 'Frequency of Last Ball', 'Number', 'Repetitions', kind='bar')

# Análisis de intervalos
plot_data_side_by_side(summary_df_first_five, summary_df_last, 'Gap Analysis for First Five Balls', 'Gap Analysis for Last Ball', 'Average Days', 'Count', kind='hist', bins=15)

# Boxplots para Análisis de Intervalos
plot_data_side_by_side(summary_df_first_five, summary_df_last, 'Boxplot Analysis for First Five Balls', 'Boxplot Analysis for Last Ball', 'Number', 'Average Days', kind='box')

# Gráficos de violín
plot_data_side_by_side(summary_df_first_five, summary_df_last, 'Violin Plot for First Five Balls', 'Violin Plot for Last Ball', 'Number', 'Average Days', kind='violin')

# Gráficos de Heatmap
plot_heatmaps_side_by_side(summary_df_first_five, summary_df_last, 'Heatmap for First Five Balls', 'Heatmap for Last Ball')

# Gráficos de Histograma
plot_density_histograms_side_by_side(summary_df_first_five, summary_df_last, 'Density and Frequency of First Five Numbers', 'Density and Frequency of Last Number', 'Number', 'Repetitions', bins=44)

# Gráficos de Caja para 'Average Days'
plt.subplot(1, 2, 2)  # Este es el segundo
sns.boxplot(data=summary_df_last, y='Average Days')
plt.title('Box Plot of Average Days Between Draws for Last Ball', fontsize=16)
plt.tight_layout()
plt.show()

# Gráficos de Densidad para 'Average Days'
plt.figure(figsize=(14, 8))
sns.kdeplot(data=summary_df_first_five, x='Average Days', shade=True, label='First Five Balls')
sns.kdeplot(data=summary_df_last, x='Average Days', shade=True, label='Last Ball')
plt.title('Density Plot of Average Days Between Draws', fontsize=18)
plt.xlabel('Average Days Between Draws', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()
