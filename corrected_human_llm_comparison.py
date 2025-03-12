import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
file_path = 'merged_llm_concordance_results_converted.csv'
df = pd.read_csv(file_path)
print(f"Original data shape: {df.shape}")

# Filter out rows where GT diagnosis doesn't match history
df = df[
    ~(
        ((df['OpenAI GT Diagnosis Matches History'] == 'No') | 
         (df['Gemini GT Diagnosis Matches History'] == 'No'))
    )
]
print(f"Data shape after filtering: {df.shape}")

# Convert rank columns to numeric, coercing errors to NaN
df['Diagnosis Match Rank'] = pd.to_numeric(df['Diagnosis Match Rank'], errors='coerce')
df['OpenAI Match Rank'] = pd.to_numeric(df['OpenAI Match Rank'], errors='coerce')
df['Gemini Match Rank'] = pd.to_numeric(df['Gemini Match Rank'], errors='coerce')
df['Top 1 ddx hit'] = pd.to_numeric(df['Top 1 ddx hit'], errors='coerce')
df['Top 5 ddx hit'] = pd.to_numeric(df['Top 5 ddx hit'], errors='coerce')

# Calculate total number of cases
total_cases = len(df)
print(f"Total cases: {total_cases}")

# CORRECTED COUNTING FOR HUMAN MATCHES
# Count rank 1 matches for human (Top 1 ddx hit = 1)
human_rank1_count = sum(df['Top 1 ddx hit'] == 1)

# Count rank 2-5 matches for human (Top 5 ddx hit values 2,3,4,5)
human_rank2to5_count = sum((df['Top 5 ddx hit'] >= 2) & (df['Top 5 ddx hit'] <= 5))

# Total human matches (ranks 1-5 combined)
human_top1to5_count = human_rank1_count + human_rank2to5_count

print(f"Human rank 1 match count: {human_rank1_count}")
print(f"Human rank 2-5 match count: {human_rank2to5_count}")
print(f"Human total top 1-5 match count: {human_top1to5_count}")

# Count LLM matches by rank
gemini_rank1_count = sum(df['Gemini Match Rank'] == 1)
gemini_rank2_count = sum(df['Gemini Match Rank'] == 2)
gemini_rank3_count = sum(df['Gemini Match Rank'] == 3)
gemini_rank4_count = sum(df['Gemini Match Rank'] == 4)
gemini_rank5_count = sum(df['Gemini Match Rank'] == 5)
gemini_top1to5_count = sum((df['Gemini Match Rank'] >= 1) & (df['Gemini Match Rank'] <= 5))

openai_rank1_count = sum(df['OpenAI Match Rank'] == 1)
openai_rank2_count = sum(df['OpenAI Match Rank'] == 2)
openai_rank3_count = sum(df['OpenAI Match Rank'] == 3)
openai_rank4_count = sum(df['OpenAI Match Rank'] == 4)
openai_rank5_count = sum(df['OpenAI Match Rank'] == 5)
openai_top1to5_count = sum((df['OpenAI Match Rank'] >= 1) & (df['OpenAI Match Rank'] <= 5))

print(f"Gemini total top 1-5 match count: {gemini_top1to5_count}")
print(f"OpenAI total top 1-5 match count: {openai_top1to5_count}")

# Calculate percentages of total cases
human_top1to5_percent = (human_top1to5_count / total_cases) * 100
gemini_top1to5_percent = (gemini_top1to5_count / total_cases) * 100
openai_top1to5_percent = (openai_top1to5_count / total_cases) * 100

# Create the corrected plot
plt.figure(figsize=(12, 8))

# Plot bars for cumulative counts
models = ['Human\nModels', 'Gemini\nModels', 'OpenAI\nModels']
cumulative_counts = [human_top1to5_count, gemini_top1to5_count, openai_top1to5_count]
colors = ['blue', 'red', 'green']

bars = plt.bar(models, cumulative_counts, color=colors, alpha=0.7)

# Add count labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom', fontsize=12)

# Add percentage labels inside bars
for i, (bar, percent) in enumerate(zip(bars, [human_top1to5_percent, gemini_top1to5_percent, openai_top1to5_percent])):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
             f'{percent:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=14)

# Add labels and title
plt.xlabel('Models', fontsize=14)
plt.ylabel('Count of Top 1-5 Rank Matches Combined', fontsize=14)
plt.title('Comparison of Top 1-5 Rank Matches Combined Across Models', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(cumulative_counts) * 1.1)  # Add some space at the top

# Save the figure
plt.tight_layout()
plt.savefig('corrected_top1to5_rank_match_comparison.png', dpi=300)

print("Corrected plot has been generated and saved successfully.") 