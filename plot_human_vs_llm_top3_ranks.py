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
df['Top 5 ddx hit'] = pd.to_numeric(df['Top 5 ddx hit'], errors='coerce')

# Count the number of cases with top 3 rank match (Top 5 ddx hit = 3) for human
# For LLMs, we'll count cases where the rank is 3
human_top3_match_count = sum(df['Top 5 ddx hit'] == 3)
gemini_top3_match_count = sum((df['Gemini Match Rank'] == 3))
openai_top3_match_count = sum((df['OpenAI Match Rank'] == 3))

print(f"Human top 3 rank match count: {human_top3_match_count}")
print(f"Gemini top 3 rank match count: {gemini_top3_match_count}")
print(f"OpenAI top 3 rank match count: {openai_top3_match_count}")

# Calculate total number of cases
total_cases = len(df)
print(f"Total cases: {total_cases}")

# Calculate percentages of total cases
human_top3_match_percent = (human_top3_match_count / total_cases) * 100
gemini_top3_match_percent = (gemini_top3_match_count / total_cases) * 100
openai_top3_match_percent = (openai_top3_match_count / total_cases) * 100

print(f"Human top 3 rank match percentage: {human_top3_match_percent:.2f}%")
print(f"Gemini top 3 rank match percentage: {gemini_top3_match_percent:.2f}%")
print(f"OpenAI top 3 rank match percentage: {openai_top3_match_percent:.2f}%")

# Filter for cases where human has top 3 rank match
human_top3_match_df = df[df['Top 5 ddx hit'] == 3]

# Count how many of these cases also have top 3 rank match in Gemini and OpenAI
gemini_overlap_count = sum((human_top3_match_df['Gemini Match Rank'] == 3))
openai_overlap_count = sum((human_top3_match_df['OpenAI Match Rank'] == 3))

# Calculate percentages of overlap with human top 3 match cases
gemini_overlap_percent = (gemini_overlap_count / human_top3_match_count) * 100 if human_top3_match_count > 0 else 0
openai_overlap_percent = (openai_overlap_count / human_top3_match_count) * 100 if human_top3_match_count > 0 else 0

print(f"\nOverlap analysis:")
print(f"Cases where both Human and Gemini have top 3 rank match: {gemini_overlap_count} ({gemini_overlap_percent:.2f}%)")
print(f"Cases where both Human and OpenAI have top 3 rank match: {openai_overlap_count} ({openai_overlap_percent:.2f}%)")

# Create all plots without showing them immediately
# This ensures all plots are generated and saved even if interactive display is interrupted

# Plot 1: Count comparison
fig1 = plt.figure(figsize=(10, 6))
models = ['Human', 'Gemini', 'OpenAI']
top3_match_counts = [human_top3_match_count, gemini_top3_match_count, openai_top3_match_count]

# Plot bars for counts
bars1 = plt.bar(models, top3_match_counts, color=['blue', 'red', 'green'], alpha=0.7)

# Add count labels on top of bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Count of Top 3 Rank Matches')
plt.title('Comparison of Top 3 Rank Match Counts Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top3_rank_match_comparison_counts.png', dpi=300)
plt.close(fig1)

# Plot 2: Percentage comparison
fig2 = plt.figure(figsize=(10, 6))
models = ['Human', 'Gemini', 'OpenAI']
top3_match_percents = [human_top3_match_percent, gemini_top3_match_percent, openai_top3_match_percent]

# Plot bars for percentages
bars2 = plt.bar(models, top3_match_percents, color=['blue', 'red', 'green'], alpha=0.7)

# Add percentage labels on top of bars
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Percentage of Total Cases (%)')
plt.title('Percentage of Cases with Top 3 Rank Matches Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top3_rank_match_comparison_percents.png', dpi=300)
plt.close(fig2)

# Plot 3: Overlap percentage
fig3 = plt.figure(figsize=(10, 6))

# Plot bars for overlap percentages
llm_models = ['Gemini', 'OpenAI']
overlap_percents = [gemini_overlap_percent, openai_overlap_percent]

bars3 = plt.bar(llm_models, overlap_percents, color=['red', 'green'], alpha=0.7)

# Add percentage labels on top of bars
for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom')

# Add labels and title
plt.xlabel('LLM Models')
plt.ylabel('Percentage of Human Top-3-Rank-Match Cases (%)')
plt.title('Percentage of Human Top-3-Rank-Match Cases Also Identified by LLMs')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a reference line at 100%
plt.axhline(y=100, color='blue', linestyle='--', label='100% of Human Top-3-Rank-Match Cases')
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_top3_rank_match_overlap_percent.png', dpi=300)
plt.close(fig3)

# Plot 4: Count overlap
fig4 = plt.figure(figsize=(10, 6))

# Plot bars for overlap counts
overlap_counts = [gemini_overlap_count, openai_overlap_count, human_top3_match_count]
count_labels = ['Gemini & Human\nOverlap', 'OpenAI & Human\nOverlap', 'Human Total\nTop 3 Rank Match']
colors = ['red', 'green', 'blue']

bars4 = plt.bar(count_labels, overlap_counts, color=colors, alpha=0.7)

# Add count labels on top of bars
for bar in bars4:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Model Overlap Categories')
plt.ylabel('Count of Top 3 Rank Match Cases')
plt.title('Count of LLM Top-3-Rank-Match Cases that Overlap with Human Top-3-Rank-Match Cases')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_top3_rank_match_overlap_counts.png', dpi=300)
plt.close(fig4)

# Plot 5: Combined analysis of ranks 1-3 for LLMs
fig5 = plt.figure(figsize=(12, 8))

# Count rank 1, 2, and 3 matches for each LLM
gemini_rank1_count = sum(df['Gemini Match Rank'] == 1)
gemini_rank2_count = sum(df['Gemini Match Rank'] == 2)
gemini_rank3_count = sum(df['Gemini Match Rank'] == 3)
openai_rank1_count = sum(df['OpenAI Match Rank'] == 1)
openai_rank2_count = sum(df['OpenAI Match Rank'] == 2)
openai_rank3_count = sum(df['OpenAI Match Rank'] == 3)

# Create stacked bar chart
llm_models = ['Gemini', 'OpenAI']
rank1_counts = [gemini_rank1_count, openai_rank1_count]
rank2_counts = [gemini_rank2_count, openai_rank2_count]
rank3_counts = [gemini_rank3_count, openai_rank3_count]

# Set up the bar positions
x = np.arange(len(llm_models))
width = 0.25

# Plot grouped bars
plt.bar(x - width, rank1_counts, width, color='darkblue', alpha=0.7, label='Rank 1 Match')
plt.bar(x, rank2_counts, width, color='blue', alpha=0.7, label='Rank 2 Match')
plt.bar(x + width, rank3_counts, width, color='lightblue', alpha=0.7, label='Rank 3 Match')

# Add count labels on top of bars
for i, count in enumerate(rank1_counts):
    plt.text(i - width, count + 5, f'{count}', ha='center', va='bottom')
for i, count in enumerate(rank2_counts):
    plt.text(i, count + 5, f'{count}', ha='center', va='bottom')
for i, count in enumerate(rank3_counts):
    plt.text(i + width, count + 5, f'{count}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('LLM Models')
plt.ylabel('Count of Rank Matches')
plt.title('Distribution of Rank 1, 2, and 3 Matches for LLM Models')
plt.xticks(x, llm_models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('llm_rank1_2_3_distribution.png', dpi=300)
plt.close(fig5)

# Plot 6: Cumulative analysis - top 3 ranks combined
fig6 = plt.figure(figsize=(10, 6))

# Calculate cumulative counts (ranks 1-3 combined)
human_top1to3_count = sum((df['Top 5 ddx hit'] >= 1) & (df['Top 5 ddx hit'] <= 3))
gemini_top1to3_count = sum((df['Gemini Match Rank'] >= 1) & (df['Gemini Match Rank'] <= 3))
openai_top1to3_count = sum((df['OpenAI Match Rank'] >= 1) & (df['OpenAI Match Rank'] <= 3))

# Plot bars for cumulative counts
models = ['Human', 'Gemini', 'OpenAI']
cumulative_counts = [human_top1to3_count, gemini_top1to3_count, openai_top1to3_count]

bars6 = plt.bar(models, cumulative_counts, color=['blue', 'red', 'green'], alpha=0.7)

# Add count labels on top of bars
for bar in bars6:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Calculate and display percentages
cumulative_percents = [(count / total_cases) * 100 for count in cumulative_counts]
for i, (bar, percent) in enumerate(zip(bars6, cumulative_percents)):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
             f'{percent:.1f}%', ha='center', va='center', color='white', fontweight='bold')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Count of Top 1-3 Rank Matches Combined')
plt.title('Comparison of Top 1-3 Rank Matches Combined Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top1to3_rank_match_comparison.png', dpi=300)
plt.close(fig6)

print("All plots have been generated and saved successfully.") 