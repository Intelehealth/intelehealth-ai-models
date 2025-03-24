import pandas as pd
import re
import os
from openai import OpenAI
from google import genai
import time
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)

# Configure OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


# Function to extract diagnoses from the LLM Diagnosis column
def extract_diagnoses(llm_diagnosis_text):
    if pd.isna(llm_diagnosis_text) or llm_diagnosis_text == "":
        return []
    
    # Split by numbered list pattern (e.g., "1. Diagnosis")
    diagnoses = re.split(r'\d+\.\s+', llm_diagnosis_text)
    # Remove empty strings
    diagnoses = [d.strip() for d in diagnoses if d.strip()]
    
    return diagnoses

# Function to check if two diagnoses match semantically using OpenAI LLM
def diagnoses_match_openai(gt_diagnosis, llm_diagnosis, client):
    prompt = f"""
    Task: Determine if the following two medical diagnoses match semantically or exactly.
    
    Ground Truth Diagnosis: {gt_diagnosis}
    LLM Diagnosis: {llm_diagnosis}
    
    Do these diagnoses refer to the same medical condition? Consider synonyms, abbreviations, and different ways of describing the same condition.
    
    Note: If Ground Truth Diagnosis is Acute Pharyngitis, it matches with LLM Diagnosis of Common Cold, Viral Upper Respiratory Infection (URTI) etc and are treated as same. Mark them as Yes.

    Your response should be in the following format:
    Match: [Yes/No]
    Rationale: [Your explanation]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change this to your preferred model
            messages=[
                {"role": "system", "content": "You are a medical expert assistant that determines if two diagnoses match semantically or exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        answer_text = response.choices[0].message.content.strip()
        
        # Parse the response to extract match and rationale
        match_result = False
        rationale = ""
        
        # Extract match result
        match_line = next((line for line in answer_text.split('\n') if line.lower().startswith('match:')), None)
        if match_line:
            match_result = 'yes' in match_line.lower()
        
        # Extract rationale
        rationale_lines = []
        capture_rationale = False
        for line in answer_text.split('\n'):
            if line.lower().startswith('rationale:'):
                capture_rationale = True
                # Include the part after "Rationale:"
                rationale_part = line[line.lower().find('rationale:') + len('rationale:'):].strip()
                if rationale_part:
                    rationale_lines.append(rationale_part)
            elif capture_rationale:
                rationale_lines.append(line.strip())
        
        rationale = ' '.join(rationale_lines)
        
        # If we couldn't parse the structured format, fall back to simple check
        if not match_line:
            match_result = 'yes' in answer_text.lower()
            rationale = answer_text
        
        return match_result, rationale
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Default to a more conservative approach if API fails
        default_match = gt_diagnosis.lower() in llm_diagnosis.lower() or llm_diagnosis.lower() in gt_diagnosis.lower()
        return default_match, f"API Error fallback: {str(e)}"

# Function to check if two diagnoses match semantically using Gemini LLM
def diagnoses_match_gemini(gt_diagnosis, llm_diagnosis, client):
    prompt = f"""
    Task: Determine if the following two medical diagnoses match semantically or exactly.
    
    Ground Truth Diagnosis: {gt_diagnosis}
    LLM Diagnosis: {llm_diagnosis}
    
    Note: If Ground Truth Diagnosis is Acute Pharyngitis, it matches with LLM Diagnosis of Common Cold, Viral Upper Respiratory Infection (URTI) etc and are treated as same. Mark them as Yes.

    Do these diagnoses refer to the same medical condition? Consider synonyms, abbreviations, and different ways of describing the same condition.
    

    Your response should be in the following format:
    Match: [Yes/No]
    Rationale: [Your explanation]
    """
    
    try:
        response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )


        answer_text = response.text.strip()
        
        # Parse the response to extract match and rationale
        match_result = False
        rationale = ""
        
        # Extract match result
        match_line = next((line for line in answer_text.split('\n') if line.lower().startswith('match:')), None)
        if match_line:
            match_result = 'yes' in match_line.lower()
        
        # Extract rationale
        rationale_lines = []
        capture_rationale = False
        for line in answer_text.split('\n'):
            if line.lower().startswith('rationale:'):
                capture_rationale = True
                # Include the part after "Rationale:"
                rationale_part = line[line.lower().find('rationale:') + len('rationale:'):].strip()
                if rationale_part:
                    rationale_lines.append(rationale_part)
            elif capture_rationale:
                rationale_lines.append(line.strip())
        
        rationale = ' '.join(rationale_lines)
        
        # If we couldn't parse the structured format, fall back to simple check
        if not match_line:
            match_result = 'yes' in answer_text.lower()
            rationale = answer_text
        
        return match_result, rationale
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Default to a more conservative approach if API fails
        default_match = gt_diagnosis.lower() in llm_diagnosis.lower() or llm_diagnosis.lower() in gt_diagnosis.lower()
        return default_match, f"API Error fallback: {str(e)}"

# Function to check if two diagnoses match using both OpenAI and Gemini
def diagnoses_match(gt_diagnosis, llm_diagnosis, openai_client, client):
    # Get results from both models
    openai_match, openai_rationale = diagnoses_match_openai(gt_diagnosis, llm_diagnosis, openai_client)
    gemini_match, gemini_rationale = diagnoses_match_gemini(gt_diagnosis, llm_diagnosis, client)
    
    # Consider it a match if either model says it's a match
    # You could also implement a stricter rule requiring both to agree
    return openai_match or gemini_match

# Function to analyze if GT diagnosis matches patient history and why LLM diagnosis was chosen
def analyze_diagnosis_mismatch(gt_diagnosis, llm_diagnoses, patient_history, client, gemini_model):
    prompt = f"""
    Task: Analyze the following medical case:
    
    Patient History: {patient_history}
    Ground Truth Diagnosis: {gt_diagnosis}
    LLM Diagnoses: {', '.join(llm_diagnoses[:5]) if llm_diagnoses else 'None provided'}
    
    Please provide:
    1. Does the Ground Truth diagnosis match the patient history? Why or why not?
    2. Do the LLM diagnoses match the patient history? Why or why not?
    3. Compare the Ground Truth diagnosis with the LLM diagnoses in the context of this patient history.
    
    Your response should be concise but thorough.
    
    IMPORTANT: Start your response with either "GT_MATCH: YES" or "GT_MATCH: NO" to indicate if the Ground Truth diagnosis matches the patient history.
    """
    
    try:
        # Get analysis from OpenAI
        openai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert assistant analyzing diagnostic accuracy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        openai_analysis = openai_response.choices[0].message.content.strip()
        
        # Get analysis from Gemini
        gemini_response = gemini_model.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        gemini_analysis = gemini_response.text.strip()
        
        # Extract GT match status from OpenAI response
        openai_gt_match_status = None
        if openai_analysis.upper().startswith("GT_MATCH: YES"):
            openai_gt_match_status = "Yes"
        elif openai_analysis.upper().startswith("GT_MATCH: NO"):
            openai_gt_match_status = "No"
        
        # Extract GT match status from Gemini response
        gemini_gt_match_status = None
        if gemini_analysis.upper().startswith("GT_MATCH: YES"):
            gemini_gt_match_status = "Yes"
        elif gemini_analysis.upper().startswith("GT_MATCH: NO"):
            gemini_gt_match_status = "No"
        
        return openai_analysis, gemini_analysis, openai_gt_match_status, gemini_gt_match_status
    except Exception as e:
        print(f"Error in mismatch analysis: {e}")
        return f"Analysis error: {str(e)}", f"Analysis error: {str(e)}", None, None

# Main function to process the CSV file
def analyze_diagnoses(csv_path, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize OpenAI client
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Initialize Gemini client
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=GEMINI_API_KEY)

    
    # Initialize new columns for the ranks and rationales
    df['Diagnosis Match Rank'] = None
    df['OpenAI Match Rank'] = None
    df['Gemini Match Rank'] = None
    df['OpenAI Rationale'] = None
    df['Gemini Rationale'] = None
    df['OpenAI GT Diagnosis Matches History'] = None
    df['Gemini GT Diagnosis Matches History'] = None
    df['Open AI Mismatch Analysis'] = None
    df['Gemini Mismatch Analysis'] = None
    
    # Count valid rows for progress bar
    valid_rows = df[df['LLM Diagnosis'].notna() & df['Diagnosis'].notna() & (df['LLM Diagnosis'] != "") & (df['Diagnosis'] != "")]
    total_valid = len(valid_rows)
    
    print(f"Processing {total_valid} valid rows...")
    
    # Process each row with tqdm progress bar
    processed_count = 0
    for idx, row in tqdm(df[:].iterrows(), total=len(df), desc="Analyzing diagnoses"):
        # Skip rows with empty LLM Diagnosis or Diagnosis
        if pd.isna(row['LLM Diagnosis']) or pd.isna(row['Diagnosis']) or row['LLM Diagnosis'] == "" or row['Diagnosis'] == "":
            continue
        
        # Extract the ground truth diagnosis
        # Extract multiple ground truth diagnoses and clean them
        gt_diagnoses = []
        if ':' in row['Diagnosis']:
            # Split by comma to handle multiple diagnoses
            diagnosis_parts = row['Diagnosis'].split(',')
            for part in diagnosis_parts:
                part = part.strip()
                # Extract diagnosis name before the colon
                if ':' in part:
                    diagnosis = part.split(':')[0].strip()
                    # Remove any trailing/leading whitespace
                    diagnosis = diagnosis.strip()
                    gt_diagnoses.append(diagnosis)
                else:
                    gt_diagnoses.append(part.strip())
        else:
            gt_diagnoses = [row['Diagnosis'].strip()]
        
        # Use the first diagnosis as primary gt_diagnosis for the rest of the code
        # If there are multiple ground truth diagnoses, combine them with commas
        if len(gt_diagnoses) > 1:
            gt_diagnosis = ", ".join(gt_diagnoses)
        else:
            gt_diagnosis = gt_diagnoses[0]
            # gt_diagnosis = row['Diagnosis'].split(':')[0] if ':' in row['Diagnosis'] else row['Diagnosis']
        
        # Extract the list of LLM diagnoses
        llm_diagnoses = extract_diagnoses(row['LLM Diagnosis'])
        
        # Get patient history (assuming it's in a column called 'Patient History' or 'History')
        patient_history = row.get('Clinical Notes', row.get('Clinical Notes', 'No patient history available'))
        
        # Check for matches and find the rank for each model
        combined_match_rank = None
        openai_match_rank = None
        gemini_match_rank = None
        openai_match_rationale = None
        gemini_match_rationale = None
        
        for i, diagnosis in enumerate(llm_diagnoses, 1):
            # Add rate limiting to avoid API throttling
            if i > 1:
                time.sleep(0.5)  # Sleep for 0.5 seconds between API calls
            
            # Check OpenAI match
            openai_result, openai_rationale = diagnoses_match_openai(gt_diagnosis, diagnosis, openai_client)
            if openai_result and openai_match_rank is None:
                openai_match_rank = i
                openai_match_rationale = openai_rationale
            
            # Check Gemini match
            gemini_result, gemini_rationale = diagnoses_match_gemini(gt_diagnosis, diagnosis, client)
            print(f"Gemini result: {gemini_result}, Gemini rationale: {gemini_rationale}, gt_diagnosis: {gt_diagnosis}, diagnosis: {diagnosis}")
            if gemini_result and gemini_match_rank is None:
                gemini_match_rank = i
                gemini_match_rationale = gemini_rationale
            
            # Combined match (either model says it's a match)
            if (openai_result == gemini_result) and (openai_result !=0 and gemini_result !=0) and combined_match_rank is None:
                combined_match_rank = i
        
        # If no match was found, set rank to 0 and analyze the mismatch
        if combined_match_rank is None:
            combined_match_rank = 0
            
            # Analyze why the match failed and compare GT diagnosis to case history
            openai_mismatch_analysis, gemini_mismatch_analysis, openai_gt_match_status, gemini_gt_match_status = analyze_diagnosis_mismatch(
                gt_diagnosis, llm_diagnoses, patient_history, openai_client, client
            )
            
            # Use OpenAI's analysis as the primary rationale if available
            if openai_match_rationale is None:
                openai_match_rationale = openai_mismatch_analysis
            
            # Use Gemini's analysis as the primary rationale if available
            if gemini_match_rationale is None:
                gemini_match_rationale = gemini_mismatch_analysis
            
            # Store the mismatch analysis
            df.at[idx, 'Open AI Mismatch Analysis'] = openai_mismatch_analysis
            df.at[idx, 'Gemini Mismatch Analysis'] = gemini_mismatch_analysis
            
            # Store GT diagnosis match status from both models
            df.at[idx, 'OpenAI GT Diagnosis Matches History'] = openai_gt_match_status
            df.at[idx, 'Gemini GT Diagnosis Matches History'] = gemini_gt_match_status
        else:
            # For cases where there is a match, we still want to check if GT diagnosis matches history
            # This is a separate analysis from the match between GT and LLM diagnoses
            openai_mismatch_analysis, gemini_mismatch_analysis, openai_gt_match_status, gemini_gt_match_status = analyze_diagnosis_mismatch(
                gt_diagnosis, llm_diagnoses, patient_history, openai_client, client
            )
            df.at[idx, 'OpenAI GT Diagnosis Matches History'] = openai_gt_match_status
            df.at[idx, 'Gemini GT Diagnosis Matches History'] = gemini_gt_match_status
        
        # Update the dataframe
        df.at[idx, 'Diagnosis Match Rank'] = combined_match_rank
        df.at[idx, 'OpenAI Match Rank'] = openai_match_rank if openai_match_rank is not None else 0
        df.at[idx, 'Gemini Match Rank'] = gemini_match_rank if gemini_match_rank is not None else 0
        df.at[idx, 'OpenAI Rationale'] = openai_match_rationale
        df.at[idx, 'Gemini Rationale'] = gemini_match_rationale
        
        # Print progress
        if idx % 10 == 0:
            print(f"Processed {idx} rows")
    
    # Save the results
    df.to_csv(output_path, index=False)
    print(f"Analysis complete. Results saved to {output_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    
    # Combined results
    match_counts = df['Diagnosis Match Rank'].value_counts().sort_index()
    total_matches = match_counts.sum()
    total_analyzed = len(df[df['LLM Diagnosis'].notna() & df['Diagnosis'].notna() & (df['LLM Diagnosis'] != "") & (df['Diagnosis'] != "")])
    
    print(f"\nCombined Results (Either OpenAI or Gemini):")
    print(f"Total rows analyzed: {total_analyzed}")
    print(f"Total matches found: {total_matches} ({total_matches/total_analyzed*100:.2f}%)")
    print("\nMatch Rank Distribution:")
    for rank, count in match_counts.items():
        print(f"Rank {rank}: {count} ({count/total_matches*100:.2f}%)")
    
    # OpenAI results
    openai_match_counts = df['OpenAI Match Rank'].value_counts().sort_index()
    openai_total_matches = openai_match_counts.sum()
    
    print(f"\nOpenAI Results:")
    print(f"Total matches found: {openai_total_matches} ({openai_total_matches/total_analyzed*100:.2f}%)")
    print("\nMatch Rank Distribution:")
    for rank, count in openai_match_counts.items():
        print(f"Rank {rank}: {count} ({count/openai_total_matches*100:.2f}%)")
    
    # Gemini results
    gemini_match_counts = df['Gemini Match Rank'].value_counts().sort_index()
    gemini_total_matches = gemini_match_counts.sum()
    
    print(f"\nGemini Results:")
    print(f"Total matches found: {gemini_total_matches} ({gemini_total_matches/total_analyzed*100:.2f}%)")
    print("\nMatch Rank Distribution:")
    for rank, count in gemini_match_counts.items():
        print(f"Rank {rank}: {count} ({count/gemini_total_matches*100:.2f}%)")

if __name__ == "__main__":
    input_csv = "gemini_2_flash_nas_combined_ayu_inference_merged_latest.csv"
    output_csv = "gemini_2_flash_nas_combined_ayu_inference_with_match_ranks_llm_550_600.csv"

    input_csv = "./data/llm_as_judge_results/21_03_2005_gemini_2_flash_ayu_cleaned_telemedicine_nas_v0.2_judged_100_150.csv"
    output_csv = "./data/llm_as_judge_results/21_03_2005_gemini_2_flash_ayu_cleaned_telemedicine_nas_v0.2_judged_100_150_match_ranks.csv"
    
    analyze_diagnoses(input_csv, output_csv) 