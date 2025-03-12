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

# Count the number of cases with top 4 rank match (Top 5 ddx hit = 4) for human
# For LLMs, we'll count cases where the rank is 4
human_top4_match_count = sum(df['Top 5 ddx hit'] == 4)
gemini_top4_match_count = sum((df['Gemini Match Rank'] == 4))
openai_top4_match_count = sum((df['OpenAI Match Rank'] == 4))

print(f"Human top 4 rank match count: {human_top4_match_count}")
print(f"Gemini top 4 rank match count: {gemini_top4_match_count}")
print(f"OpenAI top 4 rank match count: {openai_top4_match_count}")

# Calculate total number of cases
total_cases = len(df)
print(f"Total cases: {total_cases}")

# Calculate percentages of total cases
human_top4_match_percent = (human_top4_match_count / total_cases) * 100
gemini_top4_match_percent = (gemini_top4_match_count / total_cases) * 100
openai_top4_match_percent = (openai_top4_match_count / total_cases) * 100

print(f"Human top 4 rank match percentage: {human_top4_match_percent:.2f}%")
print(f"Gemini top 4 rank match percentage: {gemini_top4_match_percent:.2f}%")
print(f"OpenAI top 4 rank match percentage: {openai_top4_match_percent:.2f}%")

# Filter for cases where human has top 4 rank match
human_top4_match_df = df[df['Top 5 ddx hit'] == 4]

# Count how many of these cases also have top 4 rank match in Gemini and OpenAI
gemini_overlap_count = sum((human_top4_match_df['Gemini Match Rank'] == 4))
openai_overlap_count = sum((human_top4_match_df['OpenAI Match Rank'] == 4))

# Calculate percentages of overlap with human top 4 match cases
gemini_overlap_percent = (gemini_overlap_count / human_top4_match_count) * 100 if human_top4_match_count > 0 else 0
openai_overlap_percent = (openai_overlap_count / human_top4_match_count) * 100 if human_top4_match_count > 0 else 0

print(f"\nOverlap analysis:")
print(f"Cases where both Human and Gemini have top 4 rank match: {gemini_overlap_count} ({gemini_overlap_percent:.2f}%)")
print(f"Cases where both Human and OpenAI have top 4 rank match: {openai_overlap_count} ({openai_overlap_percent:.2f}%)")

# Create all plots without showing them immediately
# This ensures all plots are generated and saved even if interactive display is interrupted

# Plot 1: Count comparison
fig1 = plt.figure(figsize=(10, 6))
models = ['Human', 'Gemini', 'OpenAI']
top4_match_counts = [human_top4_match_count, gemini_top4_match_count, openai_top4_match_count]

# Plot bars for counts
bars1 = plt.bar(models, top4_match_counts, color=['blue', 'red', 'green'], alpha=0.7)

# Add count labels on top of bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Count of Top 4 Rank Matches')
plt.title('Comparison of Top 4 Rank Match Counts Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top4_rank_match_comparison_counts.png', dpi=300)
plt.close(fig1)

# Plot 2: Percentage comparison
fig2 = plt.figure(figsize=(10, 6))
models = ['Human', 'Gemini', 'OpenAI']
top4_match_percents = [human_top4_match_percent, gemini_top4_match_percent, openai_top4_match_percent]

# Plot bars for percentages
bars2 = plt.bar(models, top4_match_percents, color=['blue', 'red', 'green'], alpha=0.7)

# Add percentage labels on top of bars
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Percentage of Total Cases (%)')
plt.title('Percentage of Cases with Top 4 Rank Matches Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top4_rank_match_comparison_percents.png', dpi=300)
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
plt.ylabel('Percentage of Human Top-4-Rank-Match Cases (%)')
plt.title('Percentage of Human Top-4-Rank-Match Cases Also Identified by LLMs')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a reference line at 100%
plt.axhline(y=100, color='blue', linestyle='--', label='100% of Human Top-4-Rank-Match Cases')
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_top4_rank_match_overlap_percent.png', dpi=300)
plt.close(fig3)

# Plot 4: Count overlap
fig4 = plt.figure(figsize=(10, 6))

# Plot bars for overlap counts
overlap_counts = [gemini_overlap_count, openai_overlap_count, human_top4_match_count]
count_labels = ['Gemini & Human\nOverlap', 'OpenAI & Human\nOverlap', 'Human Total\nTop 4 Rank Match']
colors = ['red', 'green', 'blue']

bars4 = plt.bar(count_labels, overlap_counts, color=colors, alpha=0.7)

# Add count labels on top of bars
for bar in bars4:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Model Overlap Categories')
plt.ylabel('Count of Top 4 Rank Match Cases')
plt.title('Count of LLM Top-4-Rank-Match Cases that Overlap with Human Top-4-Rank-Match Cases')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_top4_rank_match_overlap_counts.png', dpi=300)
plt.close(fig4)

# Plot 5: Combined analysis of ranks 1-4 for LLMs
fig5 = plt.figure(figsize=(12, 8))

# Count rank 1, 2, 3, and 4 matches for each LLM
gemini_rank1_count = sum(df['Gemini Match Rank'] == 1)
gemini_rank2_count = sum(df['Gemini Match Rank'] == 2)
gemini_rank3_count = sum(df['Gemini Match Rank'] == 3)
gemini_rank4_count = sum(df['Gemini Match Rank'] == 4)
openai_rank1_count = sum(df['OpenAI Match Rank'] == 1)
openai_rank2_count = sum(df['OpenAI Match Rank'] == 2)
openai_rank3_count = sum(df['OpenAI Match Rank'] == 3)
openai_rank4_count = sum(df['OpenAI Match Rank'] == 4)

# Create stacked bar chart
llm_models = ['Gemini', 'OpenAI']
rank1_counts = [gemini_rank1_count, openai_rank1_count]
rank2_counts = [gemini_rank2_count, openai_rank2_count]
rank3_counts = [gemini_rank3_count, openai_rank3_count]
rank4_counts = [gemini_rank4_count, openai_rank4_count]

# Set up the bar positions
x = np.arange(len(llm_models))
width = 0.2  # Reduced width to accommodate 4 bars

# Plot grouped bars
plt.bar(x - 1.5*width, rank1_counts, width, color='darkblue', alpha=0.7, label='Rank 1 Match')
plt.bar(x - 0.5*width, rank2_counts, width, color='blue', alpha=0.7, label='Rank 2 Match')
plt.bar(x + 0.5*width, rank3_counts, width, color='lightblue', alpha=0.7, label='Rank 3 Match')
plt.bar(x + 1.5*width, rank4_counts, width, color='skyblue', alpha=0.7, label='Rank 4 Match')

# Add count labels on top of bars
for i, count in enumerate(rank1_counts):
    plt.text(i - 1.5*width, count + 5, f'{count}', ha='center', va='bottom')
for i, count in enumerate(rank2_counts):
    plt.text(i - 0.5*width, count + 5, f'{count}', ha='center', va='bottom')
for i, count in enumerate(rank3_counts):
    plt.text(i + 0.5*width, count + 5, f'{count}', ha='center', va='bottom')
for i, count in enumerate(rank4_counts):
    plt.text(i + 1.5*width, count + 5, f'{count}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('LLM Models')
plt.ylabel('Count of Rank Matches')
plt.title('Distribution of Rank 1, 2, 3, and 4 Matches for LLM Models')
plt.xticks(x, llm_models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('llm_rank1_2_3_4_distribution.png', dpi=300)
plt.close(fig5)

# Plot 6: Cumulative analysis - top 4 ranks combined
fig6 = plt.figure(figsize=(10, 6))

# Calculate cumulative counts (ranks 1-4 combined)
human_top1to4_count = sum((df['Top 5 ddx hit'] >= 1) & (df['Top 5 ddx hit'] <= 4))
gemini_top1to4_count = sum((df['Gemini Match Rank'] >= 1) & (df['Gemini Match Rank'] <= 4))
openai_top1to4_count = sum((df['OpenAI Match Rank'] >= 1) & (df['OpenAI Match Rank'] <= 4))

# Plot bars for cumulative counts
models = ['Human', 'Gemini', 'OpenAI']
cumulative_counts = [human_top1to4_count, gemini_top1to4_count, openai_top1to4_count]

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
plt.ylabel('Count of Top 1-4 Rank Matches Combined')
plt.title('Comparison of Top 1-4 Rank Matches Combined Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top1to4_rank_match_comparison.png', dpi=300)
plt.close(fig6)

print("All plots have been generated and saved successfully.") 