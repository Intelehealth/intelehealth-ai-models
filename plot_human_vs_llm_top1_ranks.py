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

# Count the number of cases with top rank match (rank = 1) for each model
human_top_match_count = sum(df['Top 1 ddx hit'] == 1)
gemini_top_match_count = sum(df['Gemini Match Rank'] == 1)
openai_top_match_count = sum(df['OpenAI Match Rank'] == 1)

print(f"Human top rank match count: {human_top_match_count}")
print(f"Gemini top rank match count: {gemini_top_match_count}")
print(f"OpenAI top rank match count: {openai_top_match_count}")

# Calculate total number of cases
total_cases = len(df)
print(f"Total cases: {total_cases}")

# Calculate percentages of total cases
human_top_match_percent = (human_top_match_count / total_cases) * 100
gemini_top_match_percent = (gemini_top_match_count / total_cases) * 100
openai_top_match_percent = (openai_top_match_count / total_cases) * 100

print(f"Human top rank match percentage: {human_top_match_percent:.2f}%")
print(f"Gemini top rank match percentage: {gemini_top_match_percent:.2f}%")
print(f"OpenAI top rank match percentage: {openai_top_match_percent:.2f}%")

# Filter for cases where human has top rank match
human_top_match_df = df[df['Top 1 ddx hit'] == 1]

# Count how many of these cases also have top rank match in Gemini and OpenAI
gemini_overlap_count = sum(human_top_match_df['Gemini Match Rank'] == 1)
openai_overlap_count = sum(human_top_match_df['OpenAI Match Rank'] == 1)

# Calculate percentages of overlap with human top match cases
gemini_overlap_percent = (gemini_overlap_count / human_top_match_count) * 100
openai_overlap_percent = (openai_overlap_count / human_top_match_count) * 100

print(f"\nOverlap analysis:")
print(f"Cases where both Human and Gemini have top rank match: {gemini_overlap_count} ({gemini_overlap_percent:.2f}%)")
print(f"Cases where both Human and OpenAI have top rank match: {openai_overlap_count} ({openai_overlap_percent:.2f}%)")

# Create all plots without showing them immediately
# This ensures all plots are generated and saved even if interactive display is interrupted

# Plot 1: Count comparison
fig1 = plt.figure(figsize=(10, 6))
models = ['Human', 'Gemini', 'OpenAI']
top_match_counts = [human_top_match_count, gemini_top_match_count, openai_top_match_count]

# Plot bars for counts
bars1 = plt.bar(models, top_match_counts, color=['blue', 'red', 'green'], alpha=0.7)

# Add count labels on top of bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Count of Top Rank Match (Rank = 1)')
plt.title('Comparison of Top Rank Match Counts Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top_rank_match_comparison_counts.png', dpi=300)
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
plt.ylabel('Percentage of Human Top Rank Match Cases (%)')
plt.title('Percentage of Human Top Rank Match Cases Also Identified by LLMs')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a reference line at 100%
plt.axhline(y=100, color='blue', linestyle='--', label='100% of Human Top Rank Match Cases')
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_top_rank_match_overlap_percent.png', dpi=300)
plt.close(fig2)

# Plot 3: Count overlap
fig3 = plt.figure(figsize=(10, 6))

# Plot bars for overlap counts
overlap_counts = [gemini_overlap_count, openai_overlap_count, human_top_match_count]
count_labels = ['Gemini & Human\nOverlap', 'OpenAI & Human\nOverlap', 'Human Total\nTop Rank Match']
colors = ['red', 'green', 'blue']

bars3 = plt.bar(count_labels, overlap_counts, color=colors, alpha=0.7)

# Add count labels on top of bars
for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Model Overlap Categories')
plt.ylabel('Count of Top Rank Match Cases')
plt.title('Count of LLM Top Rank Match Cases that Overlap with Human Top Rank Match Cases')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_top_rank_match_overlap_counts.png', dpi=300)
plt.close(fig3)

print("All plots have been generated and saved successfully.") 