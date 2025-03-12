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

# Count the number of cases with top 2 rank match (Top 5 ddx hit = 2) for human
# For LLMs, we'll count cases where the rank is 1 or 2
human_top2_match_count = sum(df['Top 5 ddx hit'] == 2)
gemini_top2_match_count = sum((df['Gemini Match Rank'] == 2))
openai_top2_match_count = sum((df['OpenAI Match Rank'] == 2))

print(f"Human top 2 rank match count: {human_top2_match_count}")
print(f"Gemini top 2 rank match count: {gemini_top2_match_count}")
print(f"OpenAI top 2 rank match count: {openai_top2_match_count}")

# Calculate total number of cases
total_cases = len(df)
print(f"Total cases: {total_cases}")

# Calculate percentages of total cases
human_top2_match_percent = (human_top2_match_count / total_cases) * 100
gemini_top2_match_percent = (gemini_top2_match_count / total_cases) * 100
openai_top2_match_percent = (openai_top2_match_count / total_cases) * 100

print(f"Human top 2 rank match percentage: {human_top2_match_percent:.2f}%")
print(f"Gemini top 2 rank match percentage: {gemini_top2_match_percent:.2f}%")
print(f"OpenAI top 2 rank match percentage: {openai_top2_match_percent:.2f}%")

# Filter for cases where human has top 2 rank match
human_top2_match_df = df[df['Top 5 ddx hit'] == 2]

# Count how many of these cases also have top 2 rank match in Gemini and OpenAI
gemini_overlap_count = sum((human_top2_match_df['Gemini Match Rank'] == 2))
openai_overlap_count = sum((human_top2_match_df['OpenAI Match Rank'] == 2))

# Calculate percentages of overlap with human top 2 match cases
gemini_overlap_percent = (gemini_overlap_count / human_top2_match_count) * 100 if human_top2_match_count > 0 else 0
openai_overlap_percent = (openai_overlap_count / human_top2_match_count) * 100 if human_top2_match_count > 0 else 0

print(f"\nOverlap analysis:")
print(f"Cases where both Human and Gemini have top 2 rank match: {gemini_overlap_count} ({gemini_overlap_percent:.2f}%)")
print(f"Cases where both Human and OpenAI have top 2 rank match: {openai_overlap_count} ({openai_overlap_percent:.2f}%)")

# Create all plots without showing them immediately
# This ensures all plots are generated and saved even if interactive display is interrupted

# Plot 1: Count comparison
fig1 = plt.figure(figsize=(10, 6))
models = ['Human', 'Gemini', 'OpenAI']
top2_match_counts = [human_top2_match_count, gemini_top2_match_count, openai_top2_match_count]

# Plot bars for counts
bars1 = plt.bar(models, top2_match_counts, color=['blue', 'red', 'green'], alpha=0.7)

# Add count labels on top of bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Count of Top 2 Rank Matches')
plt.title('Comparison of Top 2 Rank Match Counts Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top2_rank_match_comparison_counts.png', dpi=300)
plt.close(fig1)

# Plot 2: Percentage comparison
fig2 = plt.figure(figsize=(10, 6))
models = ['Human', 'Gemini', 'OpenAI']
top2_match_percents = [human_top2_match_percent, gemini_top2_match_percent, openai_top2_match_percent]

# Plot bars for percentages
bars2 = plt.bar(models, top2_match_percents, color=['blue', 'red', 'green'], alpha=0.7)

# Add percentage labels on top of bars
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Percentage of Total Cases (%)')
plt.title('Percentage of Cases with Top 2 Rank Matches Across Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('top2_rank_match_comparison_percents.png', dpi=300)
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
plt.ylabel('Percentage of Human Top-2-Rank-Match Cases (%)')
plt.title('Percentage of Human Top-2-Rank-Match Cases Also Identified by LLMs')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a reference line at 100%
plt.axhline(y=100, color='blue', linestyle='--', label='100% of Human Top-2-Rank-Match Cases')
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_top2_rank_match_overlap_percent.png', dpi=300)
plt.close(fig3)

# Plot 4: Count overlap
fig4 = plt.figure(figsize=(10, 6))

# Plot bars for overlap counts
overlap_counts = [gemini_overlap_count, openai_overlap_count, human_top2_match_count]
count_labels = ['Gemini & Human\nOverlap', 'OpenAI & Human\nOverlap', 'Human Total\nTop 2 Rank Match']
colors = ['red', 'green', 'blue']

bars4 = plt.bar(count_labels, overlap_counts, color=colors, alpha=0.7)

# Add count labels on top of bars
for bar in bars4:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Model Overlap Categories')
plt.ylabel('Count of Top 2 Rank Match Cases')
plt.title('Count of LLM Top-2-Rank-Match Cases that Overlap with Human Top-2-Rank-Match Cases')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('llm_vs_human_top2_rank_match_overlap_counts.png', dpi=300)
plt.close(fig4)

# Plot 5: Stacked bar chart showing distribution of rank 1 and rank 2 matches for LLMs
fig5 = plt.figure(figsize=(10, 6))

# Count rank 1 and rank 2 matches for each LLM
gemini_rank1_count = sum(df['Gemini Match Rank'] == 1)
gemini_rank2_count = sum(df['Gemini Match Rank'] == 2)
openai_rank1_count = sum(df['OpenAI Match Rank'] == 1)
openai_rank2_count = sum(df['OpenAI Match Rank'] == 2)

# Create stacked bar chart
llm_models = ['Gemini', 'OpenAI']
rank1_counts = [gemini_rank1_count, openai_rank1_count]
rank2_counts = [gemini_rank2_count, openai_rank2_count]

# Plot stacked bars
plt.bar(llm_models, rank1_counts, color='darkblue', alpha=0.7, label='Rank 1 Match')
plt.bar(llm_models, rank2_counts, bottom=rank1_counts, color='lightblue', alpha=0.7, label='Rank 2 Match')

# Add total count labels on top of bars
for i, model in enumerate(llm_models):
    total = rank1_counts[i] + rank2_counts[i]
    plt.text(i, total + 5, f'Total: {total}', ha='center', va='bottom')
    
    # Add individual count labels within bars
    plt.text(i, rank1_counts[i]/2, f'Rank 1: {rank1_counts[i]}', ha='center', va='center')
    plt.text(i, rank1_counts[i] + rank2_counts[i]/2, f'Rank 2: {rank2_counts[i]}', ha='center', va='center')

# Add labels and title
plt.xlabel('LLM Models')
plt.ylabel('Count of Rank Matches')
plt.title('Distribution of Rank 1 and Rank 2 Matches for LLM Models')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('llm_rank1_vs_rank2_distribution.png', dpi=300)
plt.close(fig5)

print("All plots have been generated and saved successfully.") 