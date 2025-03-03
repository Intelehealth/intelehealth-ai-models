import os
import pandas as pd

# Directory containing the CSV files
input_folder = 'tmp_merge'

# Output file
output_file = '03_03_2025_gemini_2_flash_nas_combined_ayu_inference_evaluated_merged_final.csv'

# List to hold dataframes
dataframes = []

# Iterate over all files in the directory
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        # Read each CSV file
        df = pd.read_csv(file_path)
        # Append the dataframe to the list
        dataframes.append(df)

# Concatenate all dataframes
merged_df = pd.concat(dataframes, ignore_index=True)

# Write the merged dataframe to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"All CSV files have been merged into {output_file}")