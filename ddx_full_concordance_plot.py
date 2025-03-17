import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

# Read the CSV file
file_path = 'merged_llm_concordance_results_converted.csv'
df = pd.read_csv(file_path)

# Filter out rows where GT diagnosis doesn't match history
df = df[
    ~(
        ((df['OpenAI GT Diagnosis Matches History'] == 'No') | 
         (df['Gemini GT Diagnosis Matches History'] == 'No'))
    )
]

# Get total number of rows in the CSV
total_rows = len(df)
print(f"Total rows in CSV: {total_rows}")

# Print column names to understand the data structure
print("\nColumn names:")
print(df.columns.tolist())

# Check the values in the Top 1 ddx hit and Top 5 ddx hit columns
print("\nTop 1 ddx hit value counts:")
print(df['Top 1 ddx hit'].value_counts().head())
print("\nTop 5 ddx hit value counts:")
print(df['Top 5 ddx hit'].value_counts().head())

# Check the values in the Gemini Match Rank column
print("\nGemini Match Rank value counts:")
print(df['Gemini Match Rank'].value_counts().head())

# Calculate all the specific counts for the concordance table
# Row 1 (Human rank 1)
count_top1_gemini1 = len(df[(df['Top 1 ddx hit'] == 1) & (df['Gemini Match Rank'] == 1)])
count_top1_gemini2 = len(df[(df['Top 1 ddx hit'] == 1) & (df['Gemini Match Rank'] == 2)])
count_top1_gemini3 = len(df[(df['Top 1 ddx hit'] == 1) & (df['Gemini Match Rank'] == 3)])
count_top1_gemini4 = len(df[(df['Top 1 ddx hit'] == 1) & (df['Gemini Match Rank'] == 4)])
count_top1_gemini5 = len(df[(df['Top 1 ddx hit'] == 1) & (df['Gemini Match Rank'] == 5)])
count_top1_gemini0 = len(df[(df['Top 1 ddx hit'] == 1) & (df['Gemini Match Rank'] == 0)])
count_top1_geminiNaN = len(df[(df['Top 1 ddx hit'] == 1) & (df['Gemini Match Rank'].isna())])

print(f"\nRow 1 (Human rank 1):")
print(f"Count where Top 1 ddx hit is 1 and Gemini Match Rank is 1: {count_top1_gemini1}")
print(f"Count where Top 1 ddx hit is 1 and Gemini Match Rank is 2: {count_top1_gemini2}")
print(f"Count where Top 1 ddx hit is 1 and Gemini Match Rank is 3: {count_top1_gemini3}")
print(f"Count where Top 1 ddx hit is 1 and Gemini Match Rank is 4: {count_top1_gemini4}")
print(f"Count where Top 1 ddx hit is 1 and Gemini Match Rank is 5: {count_top1_gemini5}")
print(f"Count where Top 1 ddx hit is 1 and Gemini Match Rank is 0: {count_top1_gemini0}")
print(f"Count where Top 1 ddx hit is 1 and Gemini Match Rank is NaN: {count_top1_geminiNaN}")

# Row 2 (Human rank 2)
count_top5_2_gemini1 = len(df[(df['Top 5 ddx hit'] == 2) & (df['Gemini Match Rank'] == 1)])
count_top5_2_gemini2 = len(df[(df['Top 5 ddx hit'] == 2) & (df['Gemini Match Rank'] == 2)])
count_top5_2_gemini3 = len(df[(df['Top 5 ddx hit'] == 2) & (df['Gemini Match Rank'] == 3)])
count_top5_2_gemini4 = len(df[(df['Top 5 ddx hit'] == 2) & (df['Gemini Match Rank'] == 4)])
count_top5_2_gemini5 = len(df[(df['Top 5 ddx hit'] == 2) & (df['Gemini Match Rank'] == 5)])
count_top5_2_gemini0 = len(df[(df['Top 5 ddx hit'] == 2) & (df['Gemini Match Rank'] == 0)])
count_top5_2_geminiNaN = len(df[(df['Top 5 ddx hit'] == 2) & (df['Gemini Match Rank'].isna())])

print(f"\nRow 2 (Human rank 2):")
print(f"Count where Top 5 ddx hit is 2 and Gemini Match Rank is 1: {count_top5_2_gemini1}")
print(f"Count where Top 5 ddx hit is 2 and Gemini Match Rank is 2: {count_top5_2_gemini2}")
print(f"Count where Top 5 ddx hit is 2 and Gemini Match Rank is 3: {count_top5_2_gemini3}")
print(f"Count where Top 5 ddx hit is 2 and Gemini Match Rank is 4: {count_top5_2_gemini4}")
print(f"Count where Top 5 ddx hit is 2 and Gemini Match Rank is 5: {count_top5_2_gemini5}")
print(f"Count where Top 5 ddx hit is 2 and Gemini Match Rank is 0: {count_top5_2_gemini0}")
print(f"Count where Top 5 ddx hit is 2 and Gemini Match Rank is NaN: {count_top5_2_geminiNaN}")

# Row 3 (Human rank 3)
count_top5_3_gemini1 = len(df[(df['Top 5 ddx hit'] == 3) & (df['Gemini Match Rank'] == 1)])
count_top5_3_gemini2 = len(df[(df['Top 5 ddx hit'] == 3) & (df['Gemini Match Rank'] == 2)])
count_top5_3_gemini3 = len(df[(df['Top 5 ddx hit'] == 3) & (df['Gemini Match Rank'] == 3)])
count_top5_3_gemini4 = len(df[(df['Top 5 ddx hit'] == 3) & (df['Gemini Match Rank'] == 4)])
count_top5_3_gemini5 = len(df[(df['Top 5 ddx hit'] == 3) & (df['Gemini Match Rank'] == 5)])
count_top5_3_gemini0 = len(df[(df['Top 5 ddx hit'] == 3) & (df['Gemini Match Rank'] == 0)])
count_top5_3_geminiNaN = len(df[(df['Top 5 ddx hit'] == 3) & (df['Gemini Match Rank'].isna())])

print(f"\nRow 3 (Human rank 3):")
print(f"Count where Top 5 ddx hit is 3 and Gemini Match Rank is 1: {count_top5_3_gemini1}")
print(f"Count where Top 5 ddx hit is 3 and Gemini Match Rank is 2: {count_top5_3_gemini2}")
print(f"Count where Top 5 ddx hit is 3 and Gemini Match Rank is 3: {count_top5_3_gemini3}")
print(f"Count where Top 5 ddx hit is 3 and Gemini Match Rank is 4: {count_top5_3_gemini4}")
print(f"Count where Top 5 ddx hit is 3 and Gemini Match Rank is 5: {count_top5_3_gemini5}")
print(f"Count where Top 5 ddx hit is 3 and Gemini Match Rank is 0: {count_top5_3_gemini0}")
print(f"Count where Top 5 ddx hit is 3 and Gemini Match Rank is NaN: {count_top5_3_geminiNaN}")

# Row 4 (Human rank 4)
count_top5_4_gemini1 = len(df[(df['Top 5 ddx hit'] == 4) & (df['Gemini Match Rank'] == 1)])
count_top5_4_gemini2 = len(df[(df['Top 5 ddx hit'] == 4) & (df['Gemini Match Rank'] == 2)])
count_top5_4_gemini3 = len(df[(df['Top 5 ddx hit'] == 4) & (df['Gemini Match Rank'] == 3)])
count_top5_4_gemini4 = len(df[(df['Top 5 ddx hit'] == 4) & (df['Gemini Match Rank'] == 4)])
count_top5_4_gemini5 = len(df[(df['Top 5 ddx hit'] == 4) & (df['Gemini Match Rank'] == 5)])
count_top5_4_gemini0 = len(df[(df['Top 5 ddx hit'] == 4) & (df['Gemini Match Rank'] == 0)])
count_top5_4_geminiNaN = len(df[(df['Top 5 ddx hit'] == 4) & (df['Gemini Match Rank'].isna())])

print(f"\nRow 4 (Human rank 4):")
print(f"Count where Top 5 ddx hit is 4 and Gemini Match Rank is 1: {count_top5_4_gemini1}")
print(f"Count where Top 5 ddx hit is 4 and Gemini Match Rank is 2: {count_top5_4_gemini2}")
print(f"Count where Top 5 ddx hit is 4 and Gemini Match Rank is 3: {count_top5_4_gemini3}")
print(f"Count where Top 5 ddx hit is 4 and Gemini Match Rank is 4: {count_top5_4_gemini4}")
print(f"Count where Top 5 ddx hit is 4 and Gemini Match Rank is 5: {count_top5_4_gemini5}")
print(f"Count where Top 5 ddx hit is 4 and Gemini Match Rank is 0: {count_top5_4_gemini0}")
print(f"Count where Top 5 ddx hit is 4 and Gemini Match Rank is NaN: {count_top5_4_geminiNaN}")

# Row 5 (Human rank 5)
count_top5_5_gemini1 = len(df[(df['Top 5 ddx hit'] == 5) & (df['Gemini Match Rank'] == 1)])
count_top5_5_gemini2 = len(df[(df['Top 5 ddx hit'] == 5) & (df['Gemini Match Rank'] == 2)])
count_top5_5_gemini3 = len(df[(df['Top 5 ddx hit'] == 5) & (df['Gemini Match Rank'] == 3)])
count_top5_5_gemini4 = len(df[(df['Top 5 ddx hit'] == 5) & (df['Gemini Match Rank'] == 4)])
count_top5_5_gemini5 = len(df[(df['Top 5 ddx hit'] == 5) & (df['Gemini Match Rank'] == 5)])
count_top5_5_gemini0 = len(df[(df['Top 5 ddx hit'] == 5) & (df['Gemini Match Rank'] == 0)])
count_top5_5_geminiNaN = len(df[(df['Top 5 ddx hit'] == 5) & (df['Gemini Match Rank'].isna())])

print(f"\nRow 5 (Human rank 5):")
print(f"Count where Top 5 ddx hit is 5 and Gemini Match Rank is 1: {count_top5_5_gemini1}")
print(f"Count where Top 5 ddx hit is 5 and Gemini Match Rank is 2: {count_top5_5_gemini2}")
print(f"Count where Top 5 ddx hit is 5 and Gemini Match Rank is 3: {count_top5_5_gemini3}")
print(f"Count where Top 5 ddx hit is 5 and Gemini Match Rank is 4: {count_top5_5_gemini4}")
print(f"Count where Top 5 ddx hit is 5 and Gemini Match Rank is 5: {count_top5_5_gemini5}")
print(f"Count where Top 5 ddx hit is 5 and Gemini Match Rank is 0: {count_top5_5_gemini0}")
print(f"Count where Top 5 ddx hit is 5 and Gemini Match Rank is NaN: {count_top5_5_geminiNaN}")

# Row 0 (Human rank 0)
count_no_top_gemini1 = len(df[(df['Top 1 ddx hit'].isna() & df['Top 5 ddx hit'].isna()) & (df['Gemini Match Rank'] == 1)])
count_no_top_gemini2 = len(df[(df['Top 1 ddx hit'].isna() & df['Top 5 ddx hit'].isna()) & (df['Gemini Match Rank'] == 2)])
count_no_top_gemini3 = len(df[(df['Top 1 ddx hit'].isna() & df['Top 5 ddx hit'].isna()) & (df['Gemini Match Rank'] == 3)])
count_no_top_gemini4 = len(df[(df['Top 1 ddx hit'].isna() & df['Top 5 ddx hit'].isna()) & (df['Gemini Match Rank'] == 4)])
count_no_top_gemini5 = len(df[(df['Top 1 ddx hit'].isna() & df['Top 5 ddx hit'].isna()) & (df['Gemini Match Rank'] == 5)])
count_no_top_gemini0 = len(df[(df['Top 1 ddx hit'].isna() & df['Top 5 ddx hit'].isna()) & (df['Gemini Match Rank'] == 0)])
count_no_top_geminiNaN = len(df[(df['Top 1 ddx hit'].isna() & df['Top 5 ddx hit'].isna()) & (df['Gemini Match Rank'].isna())])

print(f"\nRow 0 (Human rank 0):")
print(f"Count where neither Top 1 nor Top 5 ddx hit and Gemini Match Rank is 1: {count_no_top_gemini1}")
print(f"Count where neither Top 1 nor Top 5 ddx hit and Gemini Match Rank is 2: {count_no_top_gemini2}")
print(f"Count where neither Top 1 nor Top 5 ddx hit and Gemini Match Rank is 3: {count_no_top_gemini3}")
print(f"Count where neither Top 1 nor Top 5 ddx hit and Gemini Match Rank is 4: {count_no_top_gemini4}")
print(f"Count where neither Top 1 nor Top 5 ddx hit and Gemini Match Rank is 5: {count_no_top_gemini5}")
print(f"Count where neither Top 1 nor Top 5 ddx hit and Gemini Match Rank is 0: {count_no_top_gemini0}")
print(f"Count where neither Top 1 nor Top 5 ddx hit and Gemini Match Rank is NaN: {count_no_top_geminiNaN}")

# Create a new concordance matrix based on the direct counts
direct_concordance = np.zeros((6, 6), dtype=int)

# Fill in the matrix with the direct counts
# Row 1 (Human rank 1)
direct_concordance[1, 1] = count_top1_gemini1  # 155
direct_concordance[1, 2] = count_top1_gemini2  # 2
direct_concordance[1, 3] = count_top1_gemini3  # 2
direct_concordance[1, 4] = count_top1_gemini4  # 0
direct_concordance[1, 5] = count_top1_gemini5  # 1
direct_concordance[1, 0] = count_top1_gemini0 + count_top1_geminiNaN  # 36 + 0 = 36

# Row 2 (Human rank 2)
direct_concordance[2, 1] = count_top5_2_gemini1  # 0
direct_concordance[2, 2] = count_top5_2_gemini2  # 44
direct_concordance[2, 3] = count_top5_2_gemini3  # 0
direct_concordance[2, 4] = count_top5_2_gemini4  # 0
direct_concordance[2, 5] = count_top5_2_gemini5  # 0
direct_concordance[2, 0] = count_top5_2_gemini0 + count_top5_2_geminiNaN  # 4 + 0 = 4

# Row 3 (Human rank 3)
direct_concordance[3, 1] = count_top5_3_gemini1  # 0
direct_concordance[3, 2] = count_top5_3_gemini2  # 0
direct_concordance[3, 3] = count_top5_3_gemini3  # 22
direct_concordance[3, 4] = count_top5_3_gemini4  # 0
direct_concordance[3, 5] = count_top5_3_gemini5  # 0
direct_concordance[3, 0] = count_top5_3_gemini0 + count_top5_3_geminiNaN  # 1 + 0 = 1

# Row 4 (Human rank 4)
direct_concordance[4, 1] = count_top5_4_gemini1  # 0
direct_concordance[4, 2] = count_top5_4_gemini2  # 0
direct_concordance[4, 3] = count_top5_4_gemini3  # 0
direct_concordance[4, 4] = count_top5_4_gemini4  # 8
direct_concordance[4, 5] = count_top5_4_gemini5  # 0
direct_concordance[4, 0] = count_top5_4_gemini0 + count_top5_4_geminiNaN  # 1 + 0 = 1

# Row 5 (Human rank 5)
direct_concordance[5, 1] = count_top5_5_gemini1  # 0
direct_concordance[5, 2] = count_top5_5_gemini2  # 0
direct_concordance[5, 3] = count_top5_5_gemini3  # 0
direct_concordance[5, 4] = count_top5_5_gemini4  # 0
direct_concordance[5, 5] = count_top5_5_gemini5  # 3
direct_concordance[5, 0] = count_top5_5_gemini0 + count_top5_5_geminiNaN  # 1 + 0 = 1

# Row 0 (Human rank 0)
direct_concordance[0, 1] = count_no_top_gemini1  # 1
direct_concordance[0, 2] = count_no_top_gemini2  # 1
direct_concordance[0, 3] = count_no_top_gemini3  # 1
direct_concordance[0, 4] = count_no_top_gemini4  # 1
direct_concordance[0, 5] = count_no_top_gemini5  # 1
direct_concordance[0, 0] = count_no_top_gemini0 + count_no_top_geminiNaN  # 9 + 0 = 9

# Print the direct concordance matrix
print("\nDirect Concordance Matrix:")
print(direct_concordance)

# Create a DataFrame for better visualization
# Reorder to match the example image format (1-5, 0)
reordered_indices = ['1', '2', '3', '4', '5', '0']
reordered_columns = ['1', '2', '3', '4', '5', '0']

# Create the concordance DataFrame with the original order
direct_concordance_df_orig = pd.DataFrame(
    direct_concordance,
    index=['0', '1', '2', '3', '4', '5'],
    columns=['0', '1', '2', '3', '4', '5']
)

# Reorder the DataFrame to match the example image
direct_concordance_df = pd.DataFrame(
    index=reordered_indices,
    columns=reordered_columns
)

# Fill the reordered DataFrame
for i in reordered_indices:
    for j in reordered_columns:
        if i == '0' and j == '0':
            direct_concordance_df.loc[i, j] = direct_concordance_df_orig.loc['0', '0']
        elif i == '0':
            direct_concordance_df.loc[i, j] = direct_concordance_df_orig.loc['0', j]
        elif j == '0':
            direct_concordance_df.loc[i, j] = direct_concordance_df_orig.loc[i, '0']
        else:
            direct_concordance_df.loc[i, j] = direct_concordance_df_orig.loc[i, j]

print("\nReordered Direct Concordance Table:")
print(direct_concordance_df)

# Create a more visually appealing table plot
plt.figure(figsize=(10, 8))
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

# Define better colors
header_color = '#E6F2FF'  # Light blue
column_header_color = '#CCE5FF'  # Slightly darker blue
diagonal_color = '#E6F0E6'  # Light green for diagonal cells
regular_color = '#FFFFFF'  # White for regular cells

# Create the table with a better title
title = 'Diagnostic Ranking Concordance: Human Clinicians vs. Gemini 2.0 Flash'
plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

# Add subtitle explaining the table
subtitle = 'Numbers represent count of cases with corresponding rank combinations'
plt.figtext(0.5, 0.92, subtitle, fontsize=12, ha='center')

# Add row and column labels
row_label = 'Human Rank'
col_label = 'Gemini 2.0 Flash Rank'
plt.figtext(0.05, 0.5, row_label, fontsize=14, rotation=90, va='center', fontweight='bold')
plt.figtext(0.5, 0.05, col_label, fontsize=14, ha='center', fontweight='bold')

# Add column headers (1, 2, 3, 4, 5, 0)
column_headers = [['1', '2', '3', '4', '5', '0']]
column_header_colors = [[column_header_color for _ in range(6)]]

# Convert DataFrame to list of lists for table data
table_data = direct_concordance_df.values.tolist()

# Create cell colors with diagonal highlighting
cell_colors = []
for i in range(len(table_data)):
    row_colors = []
    for j in range(len(table_data[i])):
        if i == j:  # Diagonal cells
            row_colors.append(diagonal_color)
        else:
            row_colors.append(regular_color)
    cell_colors.append(row_colors)

# Create the table
table = ax.table(
    cellText=column_headers + table_data,
    rowLabels=[''] + reordered_indices,
    cellLoc='center',
    loc='center',
    cellColours=column_header_colors + cell_colors
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Add a note about which LLM was used
plt.figtext(0.5, 0.01, 'Analysis based on 351 clinical cases', 
         horizontalalignment='center', 
         verticalalignment='center',
         fontsize=10,
         style='italic')

plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])
plt.savefig('concordance_table_improved.png', bbox_inches='tight', dpi=300)
plt.close()

# Also create a heatmap visualization of the concordance table
# Convert the DataFrame to numeric values for the heatmap
heatmap_df = direct_concordance_df.copy()
for col in heatmap_df.columns:
    heatmap_df[col] = pd.to_numeric(heatmap_df[col])

plt.figure(figsize=(10, 8))
# Create a custom colormap that highlights the diagonal
cmap = sns.color_palette("Blues", as_cmap=True)

# Plot the heatmap
ax = sns.heatmap(heatmap_df, annot=True, fmt='d', cmap=cmap, linewidths=0.5, linecolor='lightgray')

# Add title and labels
plt.title('Diagnostic Ranking Concordance: Human Clinicians vs. Gemini 2.0 Flash', fontsize=16, fontweight='bold')
plt.xlabel('Gemini 2.0 Flash Rank', fontsize=14, fontweight='bold')
plt.ylabel('Human Rank', fontsize=14, fontweight='bold')

# Add a note at the bottom
plt.figtext(0.5, 0.01, 'Analysis based on 351 clinical cases', 
         horizontalalignment='center', 
         verticalalignment='center',
         fontsize=10,
         style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('concordance_heatmap_improved.png', dpi=300)
plt.close()

print("Improved concordance visualizations saved as 'concordance_table_improved.png' and 'concordance_heatmap_improved.png'")