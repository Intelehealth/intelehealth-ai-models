import pandas as pd
import openai
from openai import OpenAI
import time
from typing import List, Tuple, Dict
import os
from tqdm import tqdm
import argparse
from google import genai
from dotenv import load_dotenv

load_dotenv("ops/.env")

# Configure OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def parse_llm_response(result: str) -> Tuple[int, str, int, str]:
    """
    Parse LLM response into the expected format.
    Returns (supports_diagnosis, explanation, is_symptom, rationale)
    Handles various response formats and ensures integer outputs.
    """
    try:
        # Clean and split the response
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        
        # Try to find the first line that's just a 0 or 1
        supports_diagnosis = None
        is_symptom = None
        explanation = ""
        rationale = ""
        
        for i, line in enumerate(lines):
            # Try to find the first number (0 or 1)
            if supports_diagnosis is None and line in ['0', '1']:
                supports_diagnosis = int(line)
                # Next line should be explanation
                if i + 1 < len(lines):
                    explanation = lines[i + 1]
                continue
                
            # Try to find the second number (0 or 1)
            if is_symptom is None and line in ['0', '1']:
                is_symptom = int(line)
                # Next line should be rationale
                if i + 1 < len(lines):
                    rationale = lines[i + 1]
                continue
        
        # Validate we got all required values
        if supports_diagnosis is None or is_symptom is None:
            raise ValueError("Could not find required 0/1 values in response")
            
        return supports_diagnosis, explanation, is_symptom, rationale
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {result}")
        return 0, f"Error parsing response: {str(e)}", 0, f"Error parsing response: {str(e)}"

def get_gpt4_judgment(gt_diagnoses: List[str], clinical_notes: str) -> Tuple[int, str, int, str]:
    """
    Ask GPT-4 to judge if the clinical notes support any of the ground truth diagnoses
    and if the diagnosis is a symptom or diagnosis according to ICD-11.
    Returns a tuple of (supports_diagnosis, explanation, is_symptom, rationale)
    All boolean values are returned as integers (1 or 0)
    """
    prompt = f"""As a medical expert, evaluate if the clinical notes support the given diagnosis:

    Ground Truth Diagnoses:
    {', '.join(gt_diagnoses)}

    Clinical Notes:
    {clinical_notes}

    Provide your evaluation in exactly this format, with one number or text per line:
    1
    The clinical notes show clear evidence of fever and sore throat, which are key symptoms of acute pharyngitis.
    1
    Fever is a symptom as it is a subjective indication of illness rather than a specific disease diagnosis.

    Line 1: 1 if clinical notes support the diagnosis, 0 if not
    Line 2: Brief explanation of your judgment
    Line 3: 1 if diagnosis is a symptom, 0 if it's a diagnosis
    Line 4: Brief explanation of symptom/diagnosis classification

    Consider:
    - Medical terminology variations and synonymous conditions
    - ICD-11 classification guidelines
    - A symptom is a subjective indication (1), while a diagnosis is the identification of a disease (0)
    - Focus on finding evidence in the clinical notes
    - Respond with exactly 4 lines, starting with either 0 or 1"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert. Respond with exactly 4 lines: one number (1/0), one explanation, one number (1/0), one explanation. No JSON or other formatting. First and third lines must be exactly 0 or 1."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )

        result = response.choices[0].message.content.strip()
        return parse_llm_response(result)
        
    except Exception as e:
        print(f"Error in GPT-4 API call: {e}")
        time.sleep(2)  # Back off on error
        return 0, str(e), 0, str(e)

def get_gemini_judgment(gt_diagnoses: List[str], clinical_notes: str) -> Tuple[int, str, int, str]:
    """
    Ask Gemini to judge if the clinical notes support any of the ground truth diagnoses
    and if the diagnosis is a symptom or diagnosis according to ICD-11.
    Returns a tuple of (supports_diagnosis, explanation, is_symptom, rationale)
    All boolean values are returned as integers (1 or 0)
    """
    prompt = f"""As a medical expert, evaluate if the clinical notes support the given diagnosis:

    Ground Truth Diagnoses:
    {', '.join(gt_diagnoses)}

    Clinical Notes:
    {clinical_notes}

    Provide your evaluation in exactly this format, with one number or text per line:
    1
    The clinical notes show clear evidence of fever and sore throat, which are key symptoms of acute pharyngitis.
    1
    Fever is a symptom as it is a subjective indication of illness rather than a specific disease diagnosis.

    Line 1: 1 if clinical notes support the diagnosis, 0 if not
    Line 2: Brief explanation of your judgment
    Line 3: 1 if diagnosis is a symptom, 0 if it's a diagnosis
    Line 4: Brief explanation of symptom/diagnosis classification

    Consider:
    - Medical terminology variations and synonymous conditions
    - ICD-11 classification guidelines
    - A symptom is a subjective indication (1), while a diagnosis is the identification of a disease (0)
    - Focus on finding evidence in the clinical notes
    - Respond with exactly 4 lines, starting with either 0 or 1"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=prompt,
        )

        result = response.text.strip()
        return parse_llm_response(result)

    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        time.sleep(2)  # Back off on error
        return 0, str(e), 0, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate if clinical notes support ground truth diagnoses using both GPT-4 and Gemini")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output CSV file")
    parser.add_argument("--start-row", type=int, default=1,
                        help="Row to start processing from (1-indexed)")
    parser.add_argument("--end-row", type=int, default=None,
                        help="Row to end processing at (1-indexed, inclusive)")
    
    args = parser.parse_args()
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        exit(1)
    if not os.getenv("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variable")
        exit(1)
    
    # Modify output filename to include row range
    output_base, output_ext = os.path.splitext(args.output)
    row_range = f"_rows_{args.start_row}-{args.end_row if args.end_row else 'end'}"
    output_file = f"{output_base}{row_range}{output_ext}"
    
    # Calculate skiprows based on start_row (convert to 0-indexed)
    skip_count = args.start_row - 1
    
    # Calculate nrows if end_row is specified
    nrows = None if args.end_row is None else (args.end_row - args.start_row + 1)

    # Read the specified rows of the CSV file
    df = pd.read_csv(args.input, skiprows=range(1, skip_count + 1) if skip_count > 0 else None, nrows=nrows)

    print(f"\nProcessing rows {args.start_row} to {args.end_row if args.end_row else args.start_row + len(df) - 1}")
    print("\nFirst 5 rows of the CSV:")
    print(df[['Diagnosis', 'Clinical_notes']].head())
    print("-" * 50)

    # Initialize new columns for both models
    df['GPT4_supports_diagnosis'] = 0
    df['GPT4_explanation'] = ''
    df['GPT4_is_symptom'] = 0
    df['GPT4_rationale'] = ''
    
    df['Gemini_supports_diagnosis'] = 0
    df['Gemini_explanation'] = ''
    df['Gemini_is_symptom'] = 0
    df['Gemini_rationale'] = ''
    
    # Process each row
    total_rows = len(df)
    for idx in tqdm(df.index, desc="Evaluating diagnoses", total=total_rows, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        # Check if Clinical Notes is empty or NaN
        if pd.isna(df.at[idx, 'Clinical_notes']) or str(df.at[idx, 'Clinical_notes']).strip() == '':
            df.at[idx, 'GPT4_explanation'] = "Empty or NaN Clinical Notes"
            df.at[idx, 'Gemini_explanation'] = "Empty or NaN Clinical Notes"
            df.to_csv(output_file, index=False)
            continue
            
        gt_diagnoses = [d.strip() for d in str(df.at[idx, 'Diagnosis']).split(',')]
        clinical_notes = str(df.at[idx, 'Clinical_notes']).strip()
        
        try:
            # Get GPT-4 judgment
            gpt4_supports, gpt4_explanation, gpt4_is_symptom, gpt4_rationale = get_gpt4_judgment(gt_diagnoses, clinical_notes)
            
            # Get Gemini judgment
            gemini_supports, gemini_explanation, gemini_is_symptom, gemini_rationale = get_gemini_judgment(gt_diagnoses, clinical_notes)
            
            # Store GPT-4 results
            df.at[idx, 'GPT4_supports_diagnosis'] = gpt4_supports
            df.at[idx, 'GPT4_explanation'] = gpt4_explanation
            df.at[idx, 'GPT4_is_symptom'] = gpt4_is_symptom
            df.at[idx, 'GPT4_rationale'] = gpt4_rationale
            
            # Store Gemini results
            df.at[idx, 'Gemini_supports_diagnosis'] = gemini_supports
            df.at[idx, 'Gemini_explanation'] = gemini_explanation
            df.at[idx, 'Gemini_is_symptom'] = gemini_is_symptom
            df.at[idx, 'Gemini_rationale'] = gemini_rationale

        except Exception as e:
            df.at[idx, 'GPT4_explanation'] = f"Error processing row: {str(e)}"
            df.at[idx, 'Gemini_explanation'] = f"Error processing row: {str(e)}"
            df.to_csv(output_file, index=False)
            continue

        # Save after each successful row
        df.to_csv(output_file, index=False)
        
        # Add a small delay between API calls to avoid rate limits
        time.sleep(0.5)

    # Final save (though we've been saving after each row)
    df.to_csv(output_file, index=False)

    # Calculate and print metrics for both models
    print("\nResults:")
    print("-" * 50)
    
    # GPT-4 metrics
    gpt4_support_rate = df['GPT4_supports_diagnosis'].mean() * 100
    gpt4_symptom_rate = df['GPT4_is_symptom'].mean() * 100
    
    # Gemini metrics
    gemini_support_rate = df['Gemini_supports_diagnosis'].mean() * 100
    gemini_symptom_rate = df['Gemini_is_symptom'].mean() * 100
    
    print("GPT-4 Metrics:")
    print(f"Diagnosis Support Rate: {gpt4_support_rate:.2f}%")
    print(f"Symptom Rate: {gpt4_symptom_rate:.2f}%")
    
    print("\nGemini Metrics:")
    print(f"Diagnosis Support Rate: {gemini_support_rate:.2f}%")
    print(f"Symptom Rate: {gemini_symptom_rate:.2f}%")
    
    # Calculate agreement between models
    agreement_rate = (df['GPT4_supports_diagnosis'] == df['Gemini_supports_diagnosis']).mean() * 100
    print(f"\nModel Agreement Rate: {agreement_rate:.2f}%")
    
    print("-" * 50) 