import pandas as pd
import openai
from openai import OpenAI
import time
from typing import List, Tuple
import os
from tqdm import tqdm

# Configure OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt4_judgment(gt_diagnoses: List[str], llm_diagnosis: str) -> Tuple[bool, str]:
    """
    Ask GPT-4 to judge if the LLM diagnosis is medically equivalent to any of the ground truth diagnoses.
    Returns a tuple of (is_equivalent, explanation)
    """
    prompt = f"""As a medical expert, evaluate if the following diagnosis is medically or semantically equivalent to any of the ground truth diagnoses.
    
Ground Truth Diagnoses:
{', '.join(gt_diagnoses)}

LLM Diagnosis to evaluate:
{llm_diagnosis}

Answer with ONLY 'Yes' or 'No' followed by a brief explanation of your reasoning.
Consider medical terminology variations, common abbreviations, and synonymous conditions."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical expert evaluating diagnosis equivalence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        print(result)
        print("-" * 50)  # Add separator line after each judgment
        is_equivalent = result.lower().startswith('yes')
        return is_equivalent, result
        
    except Exception as e:
        print(f"Error in GPT-4 API call: {e}")
        print("-" * 50)  # Add separator line after error message
        time.sleep(2)  # Back off on error
        return False, str(e)

def evaluate_diagnoses(input_csv: str, output_csv: str):
    """
    Evaluate medical equivalence for diagnoses in the input CSV file.
    """
    # Read rows 50-100 of the CSV file
    df = pd.read_csv(input_csv, skiprows=range(1,50), nrows=50)
    # Read the first 5 rows of the CSV file for testing
    # df = pd.read_csv(input_csv, nrows=50)
    
    # Print the relevant columns for the first 5 rows
    print("\nFirst 5 rows of the CSV:")
    print(df[['Diagnosis', 'LLM Diagnosis']])
    print("-" * 50)  # Add separator line after displaying CSV rows
    
    # Initialize new columns
    df['Top 1 ddx hit'] = 0
    df['Top 5 ddx hit'] = 0  # This will store the position (1-5) where a match is found
    df['Match Position'] = 0  # New column to store the position where match was found
    df['gpt4_explanation'] = ''
    
    # Process each row
    for idx in tqdm(df.index, desc="Evaluating diagnoses"):
        print(f"\nRow {idx+1}:")
        print(f"Ground Truth: {df.at[idx, 'Diagnosis']}")
        print(f"LLM Diagnosis: {df.at[idx, 'LLM Diagnosis']}")
        
        # Check if LLM Diagnosis is empty or NaN
        if pd.isna(df.at[idx, 'LLM Diagnosis']) or str(df.at[idx, 'LLM Diagnosis']).strip() == '':
            print("Empty or NaN LLM Diagnosis - skipping row")
            df.at[idx, 'gpt4_explanation'] = "Empty or NaN LLM Diagnosis"
            print("-" * 50)
            continue
            
        gt_diagnoses = [d.strip() for d in str(df.at[idx, 'Diagnosis']).split(',')]
        
        # Parse the numbered list from LLM Diagnosis
        llm_diagnoses = []
        try:
            for line in str(df.at[idx, 'LLM Diagnosis']).split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 6)):
                    # Extract the diagnosis after the number
                    diagnosis = line.split('.', 1)[1].strip()
                    llm_diagnoses.append(diagnosis)
            
            # If no valid diagnoses found, skip this row
            if not llm_diagnoses:
                print("No valid numbered diagnoses found - skipping row")
                df.at[idx, 'gpt4_explanation'] = "No valid numbered diagnoses found"
                print("-" * 50)
                continue
                
            # Check each position from 1 to 5
            match_position = 0  # 0 means no match found
            explanations = []  # Store all explanations
            found_match = False
            
            # Check each position
            for position, diagnosis in enumerate(llm_diagnoses, start=1):
                print(f"Checking position {position}: {diagnosis}")
                is_equivalent, explanation = get_gpt4_judgment(gt_diagnoses, diagnosis)
                explanations.append(f"Position {position}: {explanation}")  # Add position number to explanation
                
                if is_equivalent and not found_match:  # Only update match position for the first match
                    match_position = position
                    df.at[idx, 'Match Position'] = position
                    if position == 1:
                        df.at[idx, 'Top 1 ddx hit'] = 1
                    found_match = True
                    
            # Store the position where the first match was found (0 if no match)
            df.at[idx, 'Top 5 ddx hit'] = match_position
            
            # Combine all explanations with newlines
            df.at[idx, 'gpt4_explanation'] = '\n'.join(explanations)
            
            print(f"Result: Top 1 hit = {df.at[idx, 'Top 1 ddx hit']}, Top 5 hit position = {match_position}")
            print("-" * 50)  # Add separator line after each row's processing
            
        except Exception as e:
            print(f"Error processing row: {e}")
            df.at[idx, 'gpt4_explanation'] = f"Error processing row: {str(e)}"
            print("-" * 50)
            continue
        
        # Save progress periodically
        if idx % 10 == 0:
            df.to_csv(output_csv, index=False)
            
    # Save final results with position information
    df.to_csv(output_csv, index=False)
    
    # Calculate and print metrics
    top1_accuracy = df['Top 1 ddx hit'].mean() * 100
    top5_accuracy = (df['Top 5 ddx hit'] > 0).mean() * 100  # Any non-zero value means a match was found in top 5
    
    # Calculate position distribution
    position_counts = df['Match Position'].value_counts().sort_index()
    
    print("\nResults:")
    print("-" * 50)  # Add separator line before final results
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print("\nMatch Position Distribution:")
    for position, count in position_counts.items():
        if position > 0:  # Skip position 0 (no match)
            percentage = (count / len(df)) * 100
            print(f"Position {position}: {count} matches ({percentage:.2f}%)")
    print("-" * 50)  # Add separator line after final results

if __name__ == "__main__":
    input_file = "data/v2_results/gemini_2_flash_nas_combined_ayu_inference_final.csv"
    output_file = "data/v2_results/gemini_2_flash_nas_combined_ayu_inference_evaluated_50_100.csv"
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        exit(1)
        
    evaluate_diagnoses(input_file, output_file) 