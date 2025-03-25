import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_and_save_plot(plot_func, filename):
    plt.figure(figsize=(10, 6))
    plot_func()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ddx_match_stats(csv_path):
    # Read the analysis results
    df = pd.read_csv(csv_path)
    
    # 1. Top 10 Diagnoses by Total Cases
    def plot_top_cases():
        top_10_cases = df.nlargest(10, 'Total Cases')
        sns.barplot(data=top_10_cases, x='Total Cases', y='Diagnosis')
        plt.title('Top 10 Diagnoses by Total Cases')
        plt.xlabel('Number of Cases')
    create_and_save_plot(plot_top_cases, 'top_10_cases.png')
    
    # 2. Top 10 Diagnoses by Top 1 Match Percentage
    def plot_top_1_matches():
        top_10_match1 = df.nlargest(10, 'Top 1 Match %')
        sns.barplot(data=top_10_match1, x='Top 1 Match %', y='Diagnosis')
        plt.title('Top 10 Diagnoses by Top 1 Match %')
        plt.xlabel('Match Percentage')
    create_and_save_plot(plot_top_1_matches, 'top_10_match1.png')
    
    # 3. Top 10 Diagnoses by Top 5 Match Percentage
    def plot_top_5_matches():
        top_10_match5 = df.nlargest(10, 'Top 5 Match %')
        sns.barplot(data=top_10_match5, x='Top 5 Match %', y='Diagnosis')
        plt.title('Top 10 Diagnoses by Top 5 Match %')
        plt.xlabel('Match Percentage')
    create_and_save_plot(plot_top_5_matches, 'top_10_match5.png')
    
    # 4. Bottom 10 Diagnoses by Top 5 Match Percentage
    def plot_bottom_5_matches():
        bottom_10_match5 = df.nsmallest(10, 'Top 5 Match %')
        sns.barplot(data=bottom_10_match5, x='Top 5 Match %', y='Diagnosis')
        plt.title('Bottom 10 Diagnoses by Top 5 Match %')
        plt.xlabel('Match Percentage')
    create_and_save_plot(plot_bottom_5_matches, 'bottom_10_match5.png')
    
    # 5. Distribution of Match Percentages
    def plot_distribution():
        sns.histplot(data=df, x='Top 1 Match %', bins=20, label='Top 1')
        sns.histplot(data=df, x='Top 5 Match %', bins=20, alpha=0.5, label='Top 5')
        plt.title('Distribution of Match Percentages')
        plt.xlabel('Match Percentage')
        plt.ylabel('Count')
        plt.legend()
    create_and_save_plot(plot_distribution, 'match_distribution.png')
    
    print("Plots have been saved as separate PNG files:")
    print("1. top_10_cases.png")
    print("2. top_10_match1.png")
    print("3. top_10_match5.png")
    print("4. bottom_10_match5.png")
    print("5. match_distribution.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    print(f"Total number of unique diagnoses: {len(df)}")
    print(f"Average Top 1 Match %: {df['Top 1 Match %'].mean():.2f}%")
    print(f"Average Top 5 Match %: {df['Top 5 Match %'].mean():.2f}%")
    print(f"Median Top 1 Match %: {df['Top 1 Match %'].median():.2f}%")
    print(f"Median Top 5 Match %: {df['Top 5 Match %'].median():.2f}%")
    
    # Print bottom 10 diagnoses statistics
    print("\nBottom 10 Diagnoses by Top 5 Match %:")
    print("=" * 80)
    bottom_10 = df.nsmallest(10, 'Top 5 Match %')
    for _, row in bottom_10.iterrows():
        print(f"{row['Diagnosis']:<60} {row['Top 5 Match %']:>6.2f}%")

if __name__ == "__main__":
    csv_path = "ddx_match_analysis.csv"
    plot_ddx_match_stats(csv_path) 