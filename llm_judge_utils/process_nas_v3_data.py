import pandas as pd
import os
import time
from google import genai
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv("ops/.env")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Input and output file paths
input_file = "data/NAS_UnseenData_02042025_v3_medications.csv"
output_file = "data/NAS_UnseenData_02042025_v3_medications_judged_1.csv"

# Columns to keep
columns_to_keep = [
    'Visit_id',
    'Patient_id',
    'Clinical_notes',
    'Diagnosis_provided',
    'Diagnosis',
    'Primary & Provisional',
    'Secondary & Provisional',
    'Primary & Confirmed',
    'Secondary & Confirmed',
    'Medications',
    'Medicines',
    'Strength',
    'Dosage',
    'Medical_test',
    'Medical_advice'
]

def classify_diagnosis_with_gemini(diagnosis_text):
    """
    Uses the Gemini model to classify if a diagnosis is a symptom.
    Returns 1 if it's a symptom, 0 if it's a diagnosis.
    """
    if not isinstance(diagnosis_text, str) or not diagnosis_text.strip():
        return 0  # Treat empty or non-string diagnoses as not symptoms

    prompt = f"""
    Task: Determine if the following medical text represents primarily a SYMPTOM or a DISEASE DIAGNOSIS.

    Medical Text: "{diagnosis_text}"

    Also refer to SNOMED CT terminology to see if the diagnosis text is a symptom or diagnosis.

    Answer with only 'Yes' (if it is primarily a symptom) or 'No' (if it is primarily a disease diagnosis).
    """

    try:
        # Add a delay before the API call
        time.sleep(0.5)  # Delay between API calls to avoid rate limiting
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        answer = response.text.strip().lower()

        if 'yes' in answer:
            return 1  # It's a symptom
        elif 'no' in answer:
            return 0  # It's a diagnosis
        else:
            print(f"Warning: Unexpected response for '{diagnosis_text}': {response.text}. Defaulting to 0.")
            return 0  # Default to diagnosis for ambiguous answers
    except Exception as e:
        print(f"Error calling Gemini API for '{diagnosis_text}': {e}")
        return 0  # Default to diagnosis on error

def has_multiple_diagnoses(diagnosis_str):
    if pd.isna(diagnosis_str):
        return 0
    # Split by common delimiters and clean up
    diagnoses = [d.strip() for d in str(diagnosis_str).replace(';', ',').replace('|', ',').split(',')]
    # Remove empty strings and count unique diagnoses
    diagnoses = [d for d in diagnoses if d]
    return 1 if len(diagnoses) > 1 else 0

def is_skin_disorder_case(clinical_notes, diagnosis):
    """
    Use Gemini to determine if this is a skin disorder case based on both clinical notes and diagnosis.
    Returns 1 if it's a skin disorder case, 0 otherwise.
    """
    if pd.isna(clinical_notes) or not isinstance(clinical_notes, str):
        return 0
        
    prompt = f"""
    Task: Determine if this is a skin disorder case based on both the clinical notes and diagnosis.

    Clinical Notes:
    {clinical_notes}

    Diagnosis:
    {diagnosis}

    Consider the following:
    1. The clinical notes should indicate skin-related symptoms as a chief complaint
    2. The diagnosis should be related to a skin condition/disorder
    3. Both conditions must be met to classify as a skin disorder case

    Answer with only 'Yes' (if it is a skin disorder case) or 'No' (if it is not a skin disorder case).
    """

    try:
        # Add a delay before the API call
        time.sleep(0.5)  # Delay between API calls to avoid rate limiting
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        answer = response.text.strip().lower()

        if 'yes' in answer:
            return 1  # It's a skin disorder case
        elif 'no' in answer:
            return 0  # It's not a skin disorder case
        else:
            print(f"Warning: Unexpected response for clinical notes: {response.text}. Defaulting to 0.")
            return 0  # Default to not a skin disorder case for ambiguous answers
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return 0  # Default to not a skin disorder case on error

try:
    # Read the CSV file in chunks to handle large file
    chunk_size = 10000
    chunks = pd.read_csv(input_file, chunksize=chunk_size)
    
    # Get total number of rows for progress bar
    total_rows = sum(1 for _ in pd.read_csv(input_file))
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    # Process each chunk and write to output file
    first_chunk = True
    for chunk_idx, chunk in enumerate(tqdm(chunks, total=total_chunks, desc="Processing chunks")):
        # Create a copy of the chunk to avoid SettingWithCopyWarning
        processed_chunk = chunk[columns_to_keep].copy()
        
        # Add Multiple_Diagnosis_GT column
        processed_chunk.loc[:, 'Multiple_Diagnosis_GT'] = processed_chunk['Diagnosis'].apply(has_multiple_diagnoses)
        
        # Add Is_Symptom column using Gemini classification
        print("\nClassifying diagnoses as symptoms or diseases...")
        # Create progress bar for this chunk
        pbar = tqdm(total=len(processed_chunk), desc=f"Chunk {chunk_idx + 1}/{total_chunks}")
        
        # Apply classification with progress bar
        processed_chunk.loc[:, 'Is_Symptom'] = processed_chunk['Diagnosis'].apply(
            lambda x: classify_diagnosis_with_gemini(x) if pbar.update(1) is None else 0
        )
        pbar.close()
        
        # Add Is_Skin_Disorder column using Gemini
        print("\nDetecting skin disorder cases...")
        pbar = tqdm(total=len(processed_chunk), desc=f"Chunk {chunk_idx + 1}/{total_chunks}")
        processed_chunk.loc[:, 'Is_Skin_Disorder'] = processed_chunk.apply(
            lambda row: is_skin_disorder_case(row['Clinical_notes'], row['Diagnosis']) if pbar.update(1) is None else 0,
            axis=1
        )
        pbar.close()
        
        # Write to CSV
        if first_chunk:
            processed_chunk.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            processed_chunk.to_csv(output_file, index=False, mode='a', header=False)
    
    print(f"Successfully created processed file at: {output_file}")
    print(f"Number of rows processed: {sum(1 for _ in pd.read_csv(output_file))}")
    
    # Calculate and print metrics
    final_df = pd.read_csv(output_file)
    symptom_count = final_df['Is_Symptom'].sum()
    diagnosis_count = len(final_df) - symptom_count
    skin_disorder_count = final_df['Is_Skin_Disorder'].sum()
    
    print("\nResults:")
    print("-" * 50)
    print(f"Total rows processed: {len(final_df)}")
    print(f"Symptoms detected: {symptom_count} ({symptom_count/len(final_df)*100:.2f}%)")
    print(f"Disease diagnoses detected: {diagnosis_count} ({diagnosis_count/len(final_df)*100:.2f}%)")
    print(f"Skin disorder cases detected: {skin_disorder_count} ({skin_disorder_count/len(final_df)*100:.2f}%)")
    print("-" * 50)

except Exception as e:
    print(f"An error occurred: {str(e)}") 