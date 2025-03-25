import pandas as pd
import numpy as np

def analyze_ddx_matches(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by Diagnosis and calculate metrics
    results = []
    
    for diagnosis, group in df.groupby('Diagnosis'):
        # Calculate top 1 match (rank == 1)
        diagnosis_ranks = group['Gemini Match Rank'].fillna(0)
        top1_count = (diagnosis_ranks == 1).sum()
        top1_percentage = (top1_count / len(group)) * 100 if len(group) > 0 else 0
        
        # Calculate top 5 match (rank <= 5)
        top5_count = ((diagnosis_ranks >= 1) & (diagnosis_ranks <= 5)).sum()
        top5_percentage = (top5_count / len(group)) * 100 if len(group) > 0 else 0
        
        results.append({
            'Diagnosis': diagnosis,
            'Total Cases': len(group),
            'Top 1 Matches': top1_count,
            'Top 1 Match %': round(top1_percentage, 2),
            'Top 5 Matches': top5_count,
            'Top 5 Match %': round(top5_percentage, 2)
        })
    
    # Convert results to DataFrame and sort by total cases
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Total Cases', ascending=False)
    
    # Print results
    print("\nDDX Match Analysis Results:")
    print("=" * 100)
    print(f"{'Diagnosis':<40} {'Total Cases':<12} {'Top 1 Matches':<14} {'Top 1 Match %':<14} {'Top 5 Matches':<14} {'Top 5 Match %':<14}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        print(f"{row['Diagnosis']:<40} {row['Total Cases']:<12} {row['Top 1 Matches']:<14} {row['Top 1 Match %']:<14} {row['Top 5 Matches']:<14} {row['Top 5 Match %']:<14}")
    
    # Save results to CSV
    output_file = 'ddx_match_analysis.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    print(f"Total number of unique diagnoses: {len(results_df)}")
    total_cases = results_df['Total Cases'].sum()
    total_top1_matches = results_df['Top 1 Matches'].sum()
    total_top5_matches = results_df['Top 5 Matches'].sum()
    print(f"Total number of cases analyzed: {total_cases}")
    print(f"Total Top 1 Matches: {total_top1_matches} ({(total_top1_matches/total_cases)*100:.2f}%)")
    print(f"Total Top 5 Matches: {total_top5_matches} ({(total_top5_matches/total_cases)*100:.2f}%)")

if __name__ == "__main__":
    csv_path = "data/v2_results/21_03_2005_gemini_2_flash_ayu_cleaned_telemedicine_nas_v0.2_judged.csv"
    analyze_ddx_matches(csv_path) 