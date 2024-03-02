import pandas as pd
import os

# The path to the directory containing your CSV files
folder_name = 'exported_data'

# The path for the merged output CSV file
output_file = 'exported_data/merged_file.csv'
# Get the current directory where the script is located
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the full path to the folder
folder_path = os.path.join(current_dir, folder_name)

# List to hold data from each CSV file
data_frames = []

# Get a list of all CSV filenames in the directory
csv_filenames = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv')]

# Sort the list of filenames alphabetically
csv_filenames.sort()

# Loop through all files in the sorted list
for filename in csv_filenames:
    # Construct full file path
    file_path = os.path.join(folder_path, filename)
    # Read the CSV file and append it to the list of dataframes
    df = pd.read_csv(file_path, index_col=None, header=0)
    data_frames.append(df)

# Concatenate all dataframes in the list into a single dataframe
merged_df = pd.concat(data_frames, axis=0, ignore_index=True)

# Export the merged dataframe to a new CSV file, saving it in the same directory as the script
merged_df.to_csv(os.path.join(current_dir, output_file), index=False)

print(f'Merged {len(data_frames)} files into {output_file}')