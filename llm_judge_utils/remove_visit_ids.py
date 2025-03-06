import pandas as pd

# Input and output file paths
main_file = 'gemini_2_flash_nas_v2_combined_ayu_inference_final.csv'
filter_file = 'filtered_cases_result.csv'
output_file = 'gemini_2_flash_nas_v2_filtered_output.csv'

# Read the CSV files
print(f"Reading main CSV file: {main_file}")
main_df = pd.read_csv(main_file)
print(f"Main data shape: {main_df.shape}")

print(f"Reading filter CSV file: {filter_file}")
filter_df = pd.read_csv(filter_file)
print(f"Filter data shape: {filter_df.shape}")

# Get the list of visit_ids to remove
visit_ids_to_remove = filter_df['visit_id'].tolist()
print(f"Number of visit_ids to remove: {len(visit_ids_to_remove)}")

# Filter the main dataframe to remove rows with visit_ids in the filter file
filtered_df = main_df[~main_df['visit_id'].isin(visit_ids_to_remove)]

# Display filtering results
print(f"Filtered data shape: {filtered_df.shape}")
print(f"Removed {main_df.shape[0] - filtered_df.shape[0]} rows")

# Save the filtered data to a new CSV file
filtered_df.to_csv(output_file, index=False)
print(f"Filtered data saved to: {output_file}")

# Display some statistics
print("\nFiltering Statistics:")
print(f"Original number of rows: {main_df.shape[0]}")
print(f"Number of rows after filtering: {filtered_df.shape[0]}")
print(f"Percentage of data retained: {filtered_df.shape[0]/main_df.shape[0]*100:.2f}%") 