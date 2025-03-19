# Diagnostic Concordance Analysis and Visualization Tool

## Overview

`ddx_full_concordance_plot.py` is a Python script designed to analyze and visualize the concordance between human clinician diagnostic rankings and Gemini 2.0 Flash model diagnostic rankings. The script processes diagnostic data, calculates statistics on ranking agreement, and generates visual representations of the concordance between these two diagnostic approaches.

## Purpose

The primary purpose of this script is to:

1. Analyze the concordance (agreement) of diagnostic rankings between human clinicians and the Gemini 2.0 Flash AI model
2. Generate statistical metrics on diagnostic agreement across different ranking positions (1-5 and 0)
3. Create visual representations of this concordance data in the form of tables and heatmaps
4. Filter out cases where ground truth diagnoses don't match patient history

## Features

- **Data Processing**: Filters and processes CSV data containing diagnostic rankings
- **Statistical Analysis**: Calculates detailed statistics on diagnostic ranking agreement
- **Concordance Matrix**: Generates a 6x6 (ranks 1-5 and 0) concordance matrix showing the distribution of rankings
- **Visualizations**:
  - Creates a styled table visualization showing counts of cases with corresponding rank combinations
  - Produces a heatmap visualization for better visual interpretation of concordance patterns
  - Includes relevant titles, labels, and explanatory text in the visualizations

## Input

The script expects a CSV file named `merged_llm_concordance_results_converted.csv` containing the following columns:
- `Top 1 ddx hit`: Indicates if the top diagnosis by human was correct (1) or not (0)
- `Top 5 ddx hit`: Indicates the rank (1-5) of the correct diagnosis in human's list, or 0 if absent
- `Gemini Match Rank`: Indicates the rank (1-5) of the correct diagnosis in Gemini's list, or 0 if absent
- `OpenAI GT Diagnosis Matches History`: Indicates if OpenAI's ground truth diagnosis matches patient history
- `Gemini GT Diagnosis Matches History`: Indicates if Gemini's ground truth diagnosis matches patient history

## Output

The script generates two visualization files:
1. `concordance_table_improved.png`: A styled table showing the count of cases for each human vs. Gemini rank combination
2. `concordance_heatmap_improved.png`: A heatmap visualization of the same concordance data for easier pattern recognition

## Requirements

The script requires the following Python libraries:
- pandas
- matplotlib
- numpy
- seaborn

## Installation

```bash
pip install pandas matplotlib numpy seaborn
```

## Usage

To run the script:

```bash
python ddx_full_concordance_plot.py
```

The script will:
1. Read the input CSV file
2. Filter out rows where ground truth diagnosis doesn't match history
3. Calculate and print various statistics about ranking concordance
4. Generate and save the visualizations

## Example Output

The script analyzes diagnostic concordance across 351 clinical cases and creates visualizations that show:
- Where human and AI rankings agree (highlighted on the diagonal of the matrix)
- How often AI ranks diagnoses higher or lower than human clinicians
- The distribution of rank 0 cases (where diagnosis was missed by either human or AI)

## Technical Details

The script performs the following key operations:
1. Loads and filters the dataset
2. Calculates counts of diagnostic ranking combinations
3. Constructs a concordance matrix with dimensions 6×6 (ranks 1-5 plus rank 0)
4. Creates a styled table visualization with:
   - Diagonal highlighting for cases of perfect agreement
   - Clear row and column labels
   - Informative title and subtitle
5. Generates a heatmap visualization using the seaborn library for better pattern recognition

## Notes

This analysis tool helps evaluate the comparative performance of human clinicians and the Gemini 2.0 Flash AI model in diagnostic tasks, providing insights into areas of strong agreement and potential improvement areas for both human and AI diagnostic approaches. 