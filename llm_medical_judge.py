import pandas as pd
import openai
from openai import OpenAI
import time
from typing import List, Tuple
import os
from tqdm import tqdm
import argparse
from google import genai

from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)

# Configure OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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

def get_gemini_judgment(gt_diagnoses: List[str], llm_diagnosis: str) -> Tuple[bool, str]:
    """
    Ask Gemini to judge if the LLM diagnosis is medically equivalent to any of the ground truth diagnoses.
    Returns a tuple of (is_equivalent, explanation)
    """
    prompt = f"""As a medical expert, evaluate if the following diagnosis is medically or semantically equivalent to any of the ground truth diagnoses.

    Note: Diagnosis of Acute Pharyngitis matches with Common Cold, Viral Upper Respiratory Infection (URTI) etc and are treated as same.

    Ground Truth Diagnoses:
    {', '.join(gt_diagnoses)}

    LLM Diagnosis to evaluate:
    {llm_diagnosis}

    Answer with ONLY 'Yes' or 'No' followed by a brief explanation of your reasoning.
    Consider medical terminology variations, common abbreviations, and synonymous conditions."""

    try:
        response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )

        result = response.text.strip()
        print(result)
        print("-" * 50)  # Add separator line after each judgment
        is_equivalent = result.lower().startswith('yes')
        return is_equivalent, result

    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        print("-" * 50)  # Add separator line after error message
        time.sleep(2)  # Back off on error
        return False, str(e)

def evaluate_diagnoses(input_csv: str, output_csv: str, judge_model: str = "openai", start_row: int = 1, num_rows: int = 50):
    """
    Evaluate medical equivalence for diagnoses in the input CSV file.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        judge_model: Which LLM to use as judge ("openai" or "gemini")
        start_row: Row to start processing from (1-indexed)
        num_rows: Number of rows to process
    """
    # Calculate skiprows based on start_row (convert to 0-indexed)
    skip_count = start_row - 1

    # Read the specified rows of the CSV file
    df = pd.read_csv(input_csv, skiprows=range(1, skip_count + 1) if skip_count > 0 else None, nrows=num_rows)

    # Print the relevant columns for the first 5 rows
    print(f"\nUsing {judge_model.upper()} as the judge model")
    print(f"Processing rows {start_row} to {start_row + num_rows - 1}")
    print("\nFirst 5 rows of the CSV:")
    print(df[['Diagnosis', 'LLM Diagnosis']].head())
    print("-" * 50)  # Add separator line after displaying CSV rows

    # Initialize new columns
    df['Top 1 ddx hit'] = 0
    df['Top 5 ddx hit'] = 0  # This will store the position (1-5) where a match is found
    df['Match Position'] = 0  # New column to store the position where match was found
    df['llm_explanation'] = ''

    # Select the judgment function based on the model
    get_judgment = get_gpt4_judgment if judge_model == "openai" else get_gemini_judgment
    
    # Process each row
    for idx in tqdm(df.index, desc="Evaluating diagnoses"):
        print(f"\nRow {idx+1}:")
        print(f"Ground Truth: {df.at[idx, 'Diagnosis']}")
        print(f"LLM Diagnosis: {df.at[idx, 'LLM Diagnosis']}")
        
        # Check if LLM Diagnosis is empty or NaN
        if pd.isna(df.at[idx, 'LLM Diagnosis']) or str(df.at[idx, 'LLM Diagnosis']).strip() == '':
            print("Empty or NaN LLM Diagnosis - skipping row")
            df.at[idx, 'llm_explanation'] = "Empty or NaN LLM Diagnosis"
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
                df.at[idx, 'llm_explanation'] = "No valid numbered diagnoses found"
                print("-" * 50)
                continue
                
            # Check each position from 1 to 5
            match_position = 0  # 0 means no match found
            explanations = []  # Store all explanations
            found_match = False
            
            # Check each position
            for position, diagnosis in enumerate(llm_diagnoses, start=1):
                print(f"Checking position {position}: {diagnosis}")
                is_equivalent, explanation = get_judgment(gt_diagnoses, diagnosis)
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
            df.at[idx, 'llm_explanation'] = '\n'.join(explanations)
            
            print(f"Result: Top 1 hit = {df.at[idx, 'Top 1 ddx hit']}, Top 5 hit position = {match_position}")
            print("-" * 50)  # Add separator line after each row's processing

        except Exception as e:
            print(f"Error processing row: {e}")
            df.at[idx, 'llm_explanation'] = f"Error processing row: {str(e)}"
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
    parser = argparse.ArgumentParser(description="Evaluate medical diagnoses using LLM as judge")
    parser.add_argument("--input", type=str, default="data/v2_results/gemini_2_flash_nas_combined_ayu_inference_final.csv",
                        help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/llm_as_judge_results/gemini_2_flash_nas_combined_ayu_inference_evaluated_by_gemini_2_flash.csv",
                        help="Path to output CSV file")
    parser.add_argument("--judge", type=str, choices=["openai", "gemini"], default="openai",
                        help="LLM to use as judge (openai or gemini)")
    parser.add_argument("--start-row", type=int, default=1,
                        help="Row to start processing from (1-indexed)")
    parser.add_argument("--num-rows", type=int, default=50,
                        help="Number of rows to process")
    
    args = parser.parse_args()
    
    # Configure APIs based on selected judge
    if args.judge == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("Please set the OPENAI_API_KEY environment variable")
            exit(1)
    elif args.judge == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            print("Please set the GEMINI_API_KEY environment variable")
            exit(1)
        client = genai.Client(api_key=GEMINI_API_KEY)
    
    evaluate_diagnoses(args.input, args.output, args.judge, args.start_row, args.num_rows) 