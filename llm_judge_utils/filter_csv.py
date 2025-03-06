import pandas as pd

# Input and output file paths
input_file = 'filtered_cases_with_llm_analysis_two_llms.csv'
output_file = 'filtered_cases_result.csv'

# Read the CSV file
print(f"Reading CSV file: {input_file}")
df = pd.read_csv(input_file)

# Display initial information
print(f"Original data shape: {df.shape}")

# Filter out rows where both "Top 1 ddx hit" and "Top 5 ddx hit" are 0 AND "GT_Diagnosis_Accuracy" is "No"
filtered_df = df[~((df['Top 1 ddx hit'] == 0) & 
                   (df['Top 5 ddx hit'] == 0) & 
                   (df['GT_Diagnosis_Accuracy'] == 'No'))]

# Display filtering results
print(f"Filtered data shape: {filtered_df.shape}")
print(f"Removed {df.shape[0] - filtered_df.shape[0]} rows")

# Save the filtered data to a new CSV file
filtered_df.to_csv(output_file, index=False)
print(f"Filtered data saved to: {output_file}")

# Display some statistics about the filtering
print("\nFiltering Statistics:")
print(f"Rows with 'Top 1 ddx hit' = 0: {df[df['Top 1 ddx hit'] == 0].shape[0]}")
print(f"Rows with 'Top 5 ddx hit' = 0: {df[df['Top 5 ddx hit'] == 0].shape[0]}")
print(f"Rows with 'GT_Diagnosis_Accuracy' = 'No': {df[df['GT_Diagnosis_Accuracy'] == 'No'].shape[0]}")
print(f"Rows meeting all filter conditions: {df[(df['Top 1 ddx hit'] == 0) & (df['Top 5 ddx hit'] == 0) & (df['GT_Diagnosis_Accuracy'] == 'No')].shape[0]}") 

