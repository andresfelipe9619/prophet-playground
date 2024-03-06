import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# Function to extract the first five numbers from the draw
def extract_first_five(ball_str):
    return list(map(int, ball_str.split('-')[:5]))


# Function to extract the last number from the draw
def extract_last(ball_str):
    return [int(ball_str.split('-')[-1])]


# Function to count the repetitions of each number
def count_repetitions(number, draws_data, column):
    return sum(draws_data[column].apply(lambda x: number in x))


# Function to calculate the average number of days between appearances of a number
def calculate_average_days(number, draws_data, column):
    dates = draws_data[draws_data[column].apply(lambda x: number in x)]['Date']
    if len(dates) < 2:
        return None  # Can't calculate if there are fewer than two dates
    sorted_dates = sorted(dates)
    differences = [(sorted_dates[i] - sorted_dates[i - 1]).days for i in range(1, len(sorted_dates))]
    return sum(differences) / len(differences) if differences else None


# Function to calculate the next expected date based on the last date and average days
def calculate_expected_date(last_date, average_days):
    if last_date is None or average_days is None:
        return None
    return last_date + timedelta(days=round(average_days))


# Function to create a summary DataFrame for each number with additional statistics
def create_summary_dataframe(draws_data, column_name):
    summary_data = []
    number_range = range(1, 44) if column_name == 'FirstFive' else range(1, 17)  # Adjust this range according to
    # your numbers
    for number in number_range:
        repetitions = count_repetitions(number, draws_data, column_name)
        last_date = draws_data[draws_data[column_name].apply(lambda x: number in x)]['Date'].max()
        dates = draws_data[draws_data[column_name].apply(lambda x: number in x)]['Date']
        if len(dates) < 2:
            average_days = None
            min_days = None
            max_days = None
            std_dev_days = None
        else:
            sorted_dates = sorted(dates)
            differences = [(sorted_dates[i] - sorted_dates[i - 1]).days for i in range(1, len(sorted_dates))]
            average_days = sum(differences) / len(differences)
            min_days = min(differences)
            max_days = max(differences)
            std_dev_days = pd.Series(differences).std()
        expected_date = calculate_expected_date(last_date, average_days)
        summary_data.append({
            'Number': number,
            'Repetitions': repetitions,
            'Last Date': last_date,
            'Average Days': average_days,
            'Min Days': min_days,
            'Max Days': max_days,
            'Std Dev Days': std_dev_days,
            'Expected Date': expected_date
        })
    return pd.DataFrame(summary_data)


# Assuming 'draws_data' is your DataFrame of lottery data
draws_data = pd.read_csv('exported_data/final-2023.csv')
draws_data['Date'] = pd.to_datetime(draws_data['Date'], format='%d/%m/%Y')
draws_data['FirstFive'] = draws_data['Ball'].apply(extract_first_five)
draws_data['Last'] = draws_data['Ball'].apply(extract_last)

# Create the summary DataFrame for the first five balls and for the last ball
summary_df_first_five = create_summary_dataframe(draws_data, 'FirstFive')
csv_file_path_first_five = 'summary/lottery_ball_summary_first_five.csv'
summary_df_first_five.to_csv(csv_file_path_first_five, index=False)

summary_df_last = create_summary_dataframe(draws_data, 'Last')
csv_file_path_last = 'summary/lottery_ball_summary_last.csv'
summary_df_last.to_csv(csv_file_path_last, index=False)

print(f"CSV file for the first five balls saved: {csv_file_path_first_five}")
print(f"CSV file for the last ball saved: {csv_file_path_last}")

# Análisis de Frecuencia (Bar Chart) para el último número
plt.figure(figsize=(14, 8))
plt.title('Frequency of First Five Balls', fontsize=18)
sns.barplot(data=summary_df_first_five, x='Number', y='Repetitions', palette='Blues', hue='Number', legend=False)
plt.xlabel('Number', fontsize=15)
plt.ylabel('Repetitions', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

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