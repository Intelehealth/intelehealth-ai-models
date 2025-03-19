import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import Counter

# Read the CSV file
data_path = 'data/DDx_database-190-cases-data-cleaned-ayu.csv'
df = pd.read_csv(data_path)

# Extract the diagnosis column
diagnoses = df['diagnosis'].tolist()

# Function to standardize diagnosis names
def standardize_diagnosis(diagnosis):
    # Convert to lowercase for consistency
    diagnosis = diagnosis.lower()
    
    # Remove common suffixes/words that don't add to the category
    diagnosis = re.sub(r'\s*(syndrome|disease|disorder|infection)\s*$', '', diagnosis)
    
    # Standardize common variations
    # Normalize common variations in diagnosis names
    diagnosis = re.sub(r'acute\s+', '', diagnosis)
    diagnosis = re.sub(r'chronic\s+', '', diagnosis)
    diagnosis = re.sub(r'type\s+[12]\s+', '', diagnosis)
    diagnosis = re.sub(r'community\s+acquired\s+', '', diagnosis)
    diagnosis = re.sub(r'pulmonary\s+', '', diagnosis)

    # Remove acronyms by writing them out in full
    diagnosis = re.sub(r'\bafib\b', 'atrial fibrillation', diagnosis)
    diagnosis = re.sub(r'\bcad\b', 'coronary artery disease', diagnosis)
    diagnosis = re.sub(r'\bchf\b', 'congestive heart failure', diagnosis)
    diagnosis = re.sub(r'\bcva\b', 'cerebrovascular accident', diagnosis)
    diagnosis = re.sub(r'\bdm\b', 'diabetes mellitus', diagnosis)
    diagnosis = re.sub(r'\bhtn\b', 'hypertension', diagnosis)
    diagnosis = re.sub(r'\bmi\b', 'myocardial infarction', diagnosis)
    diagnosis = re.sub(r'\bpe\b', 'pulmonary embolism', diagnosis)
    diagnosis = re.sub(r'\bra\b', 'rheumatoid arthritis', diagnosis)
    diagnosis = re.sub(r'\bsle\b', 'systemic lupus erythematosus', diagnosis)
    diagnosis = re.sub(r'\btb\b', 'tuberculosis', diagnosis)
    diagnosis = re.sub(r'\buti\b', 'urinary tract infection', diagnosis)
    
    # Add more standardizations for common medical terms
    diagnosis_map = {
        'ca': 'cancer',
        'carcinoma': 'cancer',
        'neoplasm': 'cancer',
        'malignancy': 'cancer',
        'copd': 'chronic obstructive pulmonary',
        'renal failure': 'kidney failure',
        'cerebrovascular accident': 'stroke',
        'myocardial infarction': 'heart attack',
        'gastroenteritis': 'gastro',
        'pneumonia': 'pneumonia'
    }
    
    for key, value in diagnosis_map.items():
        if key in diagnosis:
            diagnosis = diagnosis.replace(key, value)
    
    # Capitalize first letter for display
    diagnosis = diagnosis.strip().capitalize()
    
    return diagnosis

# Standardize all diagnoses
standardized_diagnoses = [standardize_diagnosis(d) for d in diagnoses]

# Count occurrences of each standardized diagnosis
diagnosis_counts = Counter(standardized_diagnoses)

# Get top 20 diagnoses sorted by count
top_n = 20  # Adjust this number to show more or fewer categories
top_diagnoses = diagnosis_counts.most_common(top_n)
categories = [item[0] for item in top_diagnoses]
counts = [item[1] for item in top_diagnoses]

# Create a horizontal bar plot with better formatting
plt.figure(figsize=(12, 10))  # Fixed size for better readability

# Create color gradient
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(categories)))
bars = plt.barh(range(len(categories)), counts, color=colors)

# Add values at the end of each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.3, bar.get_y() + bar.get_height()/2,
             f'{int(width)}', ha='left', va='center', fontweight='bold')

# Add labels and title
plt.ylabel('Diagnosis Category', fontsize=14, fontweight='bold')
plt.xlabel('Number of Cases', fontsize=14, fontweight='bold')
plt.title(f'Top {top_n} Diagnosis Categories', fontsize=16, fontweight='bold')

# Set y-axis labels with improved formatting
plt.yticks(range(len(categories)), categories, fontsize=12)

# Add grid lines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('top_diagnosis_distribution.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Generate a summary of all diagnoses
all_diagnoses = diagnosis_counts.most_common()
print(f"\nTotal unique diagnosis categories (after standardization): {len(diagnosis_counts)}")

# Save full results to a file
with open('diagnosis_summary.txt', 'w') as f:
    f.write(f"Total unique diagnosis categories (after standardization): {len(diagnosis_counts)}\n\n")
    f.write("All diagnosis categories:\n")
    for category, count in all_diagnoses:
        f.write(f"{category}: {count}\n")

print(f"\nSaved complete diagnosis list to 'diagnosis_summary.txt'")
print(f"Saved plot to 'top_diagnosis_distribution.png'")

# Optional: Create a separate plot for "Other" categories
if len(all_diagnoses) > top_n:
    # Get the remaining categories
    other_diagnoses = all_diagnoses[top_n:top_n+20]  # Next 20 categories
    other_categories = [item[0] for item in other_diagnoses]
    other_counts = [item[1] for item in other_diagnoses]
    
    # Create another horizontal bar plot
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(other_categories)), other_counts, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(other_categories))))
    
    # Add values at the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    # Add labels and title
    plt.ylabel('Diagnosis Category', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Cases', fontsize=14, fontweight='bold')
    plt.title(f'Next {len(other_categories)} Diagnosis Categories', fontsize=16, fontweight='bold')
    
    # Set y-axis labels
    plt.yticks(range(len(other_categories)), other_categories, fontsize=12)
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('secondary_diagnosis_distribution.png', dpi=300, bbox_inches='tight') 