import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read the CSV file
file_path = 'merged_llm_concordance_results_converted.csv'
df = pd.read_csv(file_path)
print(df.shape)

# Filter out rows where GT diagnosis doesn't match history
df = df[
    ~(
        ((df['OpenAI GT Diagnosis Matches History'] == 'No') | 
         (df['Gemini GT Diagnosis Matches History'] == 'No'))
    )
]

# Convert rank columns to numeric, coercing errors to NaN
df['Diagnosis Match Rank'] = pd.to_numeric(df['Diagnosis Match Rank'], errors='coerce')
df['OpenAI Match Rank'] = pd.to_numeric(df['OpenAI Match Rank'], errors='coerce')
df['Gemini Match Rank'] = pd.to_numeric(df['Gemini Match Rank'], errors='coerce')
df['Top 1 ddx hit'] = pd.to_numeric(df['Top 1 ddx hit'], errors='coerce')

# Count the number of cases with no rank match (rank = 0) for each model
human_no_match_count = sum(df['Top 1 ddx hit'] == 0)
gemini_no_match_count = sum(df['Gemini Match Rank'] == 0)
openai_no_match_count = sum(df['OpenAI Match Rank'] == 0)

print(f"Human no rank match count: {human_no_match_count}")
print(f"Gemini no rank match count: {gemini_no_match_count}")
print(f"OpenAI no rank match count: {openai_no_match_count}")

# Calculate total number of cases
total_cases = len(df)
print(f"Total cases: {total_cases}")

# Calculate percentages of total cases
human_no_match_percent = (human_no_match_count / total_cases) * 100
gemini_no_match_percent = (gemini_no_match_count / total_cases) * 100
openai_no_match_percent = (openai_no_match_count / total_cases) * 100

print(f"Human no rank match percentage: {human_no_match_percent:.2f}%")
print(f"Gemini no rank match percentage: {gemini_no_match_percent:.2f}%")
print(f"OpenAI no rank match percentage: {openai_no_match_percent:.2f}%")

# Filter for cases where human has no rank match
human_no_match_df = df[df['Top 1 ddx hit'] == 0]

# Count how many of these cases also have no rank match in Gemini and OpenAI
gemini_overlap_count = sum(human_no_match_df['Gemini Match Rank'] == 0)
openai_overlap_count = sum(human_no_match_df['OpenAI Match Rank'] == 0)

# Calculate percentages of overlap with human no match cases
gemini_overlap_percent = (gemini_overlap_count / human_no_match_count) * 100
openai_overlap_percent = (openai_overlap_count / human_no_match_count) * 100

print(f"\nOverlap analysis:")
print(f"Cases where both Human and Gemini have no rank match: {gemini_overlap_count} ({gemini_overlap_percent:.2f}%)")
print(f"Cases where both Human and OpenAI have no rank match: {openai_overlap_count} ({openai_overlap_percent:.2f}%)")

# Create all plots without showing them immediately
# This ensures all plots are generated and saved even if interactive display is interrupted

# Plot 1: Count comparison
fig1 = plt.figure(figsize=(10, 6))
models = ['Human', 'Gemini', 'OpenAI']
no_match_counts = [human_no_match_count, gemini_no_match_count, openai_no_match_count]

# Plot bars for counts
bars1 = plt.bar(models, no_match_counts, color=['blue', 'red', 'green'], alpha=0.7)

# Add count labels on top of bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Count of No Rank Match (Rank = 0)')
plt.title('Comparison of No Rank Match Counts Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('no_rank_match_comparison_counts.png', dpi=300)
plt.close(fig1)

# Plot 2: Percentage overlap
fig2 = plt.figure(figsize=(10, 6))

# Plot bars for overlap percentages
llm_models = ['Gemini', 'OpenAI']
overlap_percents = [gemini_overlap_percent, openai_overlap_percent]

bars2 = plt.bar(llm_models, overlap_percents, color=['red', 'green'], alpha=0.7)

# Add percentage labels on top of bars
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom')

# Add labels and title
plt.xlabel('LLM Models')
plt.ylabel('Percentage of Human No-Rank-Match Cases (%)')
plt.title('Percentage of Human No-Rank-Match Cases Also Identified by LLMs')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a reference line at 100%
plt.axhline(y=100, color='blue', linestyle='--', label='100% of Human No-Rank-Match Cases')
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_no_rank_match_overlap_percent.png', dpi=300)
plt.close(fig2)

# Plot 3: Count overlap
fig3 = plt.figure(figsize=(10, 6))

# Plot bars for overlap counts
overlap_counts = [gemini_overlap_count, openai_overlap_count, human_no_match_count]
count_labels = ['Gemini & Human\nOverlap', 'OpenAI & Human\nOverlap', 'Human Total\nNo Rank Match']
colors = ['red', 'green', 'blue']

bars3 = plt.bar(count_labels, overlap_counts, color=colors, alpha=0.7)

# Add count labels on top of bars
for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Model Overlap Categories')
plt.ylabel('Count of No Rank Match Cases')
plt.title('Count of LLM No-Rank-Match Cases that Overlap with Human No-Rank-Match Cases')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_no_rank_match_overlap_counts.png', dpi=300)
plt.close(fig3)

print("All plots have been generated and saved successfully.")