import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
file_path = 'merged_llm_concordance_results.csv'
df = pd.read_csv(file_path)

# Filter out rows where GT diagnosis doesn't match history
df = df[
    ~(
        ((df['OpenAI GT Diagnosis Matches History'] == 'No') | 
         (df['Gemini GT Diagnosis Matches History'] == 'No'))
    )
]

# Convert Top 1 ddx hit column to numeric, coercing errors to NaN
df['Top 1 ddx hit'] = pd.to_numeric(df['Top 1 ddx hit'], errors='coerce')

# Filter for failing diagnoses (Top 1 ddx hit == 0)
failing_df = df[df['Top 1 ddx hit'] == 0]

# Count occurrences of each unique diagnosis
diagnosis_counts = failing_df['Diagnosis'].value_counts()

# Print the counts of each failing diagnosis
print("\nFailing Diagnoses (Top 1 ddx hit == 0) Counts:")
print(diagnosis_counts)

# Plot all failing diagnoses as a horizontal bar chart
plt.figure(figsize=(12, 20))  # Swap dimensions for horizontal orientation
diagnosis_counts.plot(kind='barh')  # Use 'barh' for horizontal bars
plt.title('Failing DDX Case Counts')
plt.xlabel('Count')  # Swap x and y labels
plt.ylabel('Diagnosis')
plt.yticks(fontsize=9)  # Adjust font size for readability
plt.tight_layout()
plt.savefig('failing_ddx_case_counts.png', dpi=300)
plt.show()

# Save the failing diagnoses counts to a CSV file
diagnosis_counts.to_frame().reset_index().rename(columns={'index': 'Diagnosis', 0: 'Count'}).to_csv('failing_diagnoses_counts.csv', index=False)

# Print total number of failing diagnoses
print(f"\nTotal number of failing diagnoses: {len(failing_df)}")
print(f"Total number of unique failing diagnoses: {len(diagnosis_counts)}")
print(f"Results saved to 'failing_ddx_case_counts.png' and 'failing_diagnoses_counts.csv'")