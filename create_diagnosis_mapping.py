import pandas as pd
import json
import re

def clean_diagnosis_text(diagnosis: str) -> str:
    """
    Clean the diagnosis text to extract just the diagnosis name.
    
    Args:
        diagnosis: The raw diagnosis text
        
    Returns:
        Cleaned diagnosis text
    """
    # Extract the main diagnosis before any colon
    if ':' in diagnosis:
        diagnosis = diagnosis.split(':')[0].strip()
    
    return diagnosis

def extract_top1_diagnosis(llm_diagnosis: str) -> str:
    """
    Extract the top 1 diagnosis from the LLM diagnosis list.
    
    Args:
        llm_diagnosis: The LLM diagnosis text containing ranked diagnoses
        
    Returns:
        The top 1 diagnosis
    """
    # The format is typically "1. Diagnosis\n2. ..."
    if pd.isna(llm_diagnosis):
        return ""
    
    # Split by newline and take the first line
    lines = llm_diagnosis.split('\n')
    if not lines:
        return ""
    
    # Extract the diagnosis after the "1. " prefix
    first_line = lines[0]
    
    # Try to match the pattern "1. Diagnosis"
    match = re.match(r'1\.\s+(.*)', first_line)
    if match:
        # Extract just the diagnosis part, not any additional text after it
        diagnosis_text = match.group(1).strip()
        # If there are numbers like "2." in the text, cut off at that point
        if " 2." in diagnosis_text:
            diagnosis_text = diagnosis_text.split(" 2.")[0].strip()
        return diagnosis_text
    
    # If the format is different, just return the first line
    return first_line.strip()

def main():
    """Main function to process the data and build the diagnosis mapping."""
    # Read the CSV file
    file_path = 'merged_llm_concordance_results_converted.csv'
    df = pd.read_csv(file_path)
    
    # Create a dictionary to store the mapping
    diagnosis_mapping = {}
    
    # First, process rows that meet our original criteria
    filtered_df = df[
        (df['Top 1 ddx hit'] == 1) & 
        (df['Top 5 ddx hit'] == 1) & 
        df['Diagnosis Match Rank'].notna()
    ]
    
    # Process each row in the filtered dataframe
    for _, row in filtered_df.iterrows():
        # Get the broad diagnosis (cleaned)
        broad_diagnosis = clean_diagnosis_text(row['Diagnosis'])
        
        # Extract the top 1 diagnosis from the LLM Diagnosis
        specific_diagnosis = extract_top1_diagnosis(row['LLM Diagnosis'])
        
        # Skip if either diagnosis is empty
        if not broad_diagnosis or not specific_diagnosis:
            continue
            
        # If this broad diagnosis is already in our mapping, append to the list
        if broad_diagnosis in diagnosis_mapping:
            # If the value is not already a list, convert it to a list
            if not isinstance(diagnosis_mapping[broad_diagnosis], list):
                diagnosis_mapping[broad_diagnosis] = [diagnosis_mapping[broad_diagnosis]]
            
            # Add the new specific diagnosis if it's not already in the list
            if specific_diagnosis not in diagnosis_mapping[broad_diagnosis]:
                diagnosis_mapping[broad_diagnosis].append(specific_diagnosis)
        else:
            # First occurrence of this broad diagnosis
            diagnosis_mapping[broad_diagnosis] = specific_diagnosis
    
    # Save the mapping to a JSON file
    with open('diagnosis_mapping.json', 'w') as f:
        json.dump(diagnosis_mapping, f, indent=4, sort_keys=True)
    
    print(f"Created mapping with {len(diagnosis_mapping)} diagnosis pairs")
    print("Saved to diagnosis_mapping.json")

if __name__ == "__main__":
    main() 