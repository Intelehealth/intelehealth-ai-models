import pandas as pd
import os
import sys
import time
from google import genai
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configuration ---
INPUT_CSV_PATH = '/Users/bsb/work/intelehealth-ai-models/data/NAS_v2_unseen_data_clean_gt_no_fatigue_v0.1.csv'
OUTPUT_CSV_PATH = '/Users/bsb/work/intelehealth-ai-models/data/NAS_v2_unseen_data_clean_gt_symptoms_as_diagnoses_v0.1.csv'
DIAGNOSIS_COLUMN = 'Diagnosis'
OUTPUT_COLUMN = 'Is_Symptom'
API_CALL_DELAY_SECONDS = 0.5 # Delay between API calls to avoid rate limiting
GEMINI_MODEL_NAME = 'gemini-2.0-flash' # Or choose another suitable model
# --- End Configuration ---

def classify_diagnosis_with_gemini(diagnosis_text, client):
    """
    Uses the Gemini model to classify if a diagnosis is a symptom.

    Args:
        diagnosis_text (str): The diagnosis string to classify.
        model: The configured Gemini GenerativeModel instance.

    Returns:
        str: 'Yes' if judged as a symptom, 'No' otherwise. Returns 'Error' on API failure.
    """
    if not isinstance(diagnosis_text, str) or not diagnosis_text.strip():
        return 'No' # Treat empty or non-string diagnoses as not symptoms

    prompt = f"""
    Task: Determine if the following medical text represents primarily a SYMPTOM or a DISEASE DIAGNOSIS.

    Medical Text: "{diagnosis_text}"

    Is this text describing a symptom (like 'headache', 'cough', 'fever', 'pain', 'nausea', 'rash')
    rather than a specific disease diagnosis (like 'Pneumonia', 'Diabetes Mellitus', 'Hypertension', 'Appendicitis')?

    Answer with only 'Yes' (if it is primarily a symptom) or 'No' (if it is primarily a disease diagnosis).
    """

    try:
        # Add a delay before the API call
        time.sleep(API_CALL_DELAY_SECONDS)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        answer = response.text.strip().lower()

        if 'yes' in answer:
            return 'Yes'
        elif 'no' in answer:
            return 'No'
        else:
            print(f"Warning: Unexpected response for '{diagnosis_text}': {response.text}. Defaulting to 'No'.")
            return 'No' # Default to 'No' for ambiguous answers
    except Exception as e:
        print(f"Error calling Gemini API for '{diagnosis_text}': {e}")
        return 'Error' # Indicate API call failure

def process_diagnoses(input_path, output_path):
    """
    Reads the input CSV, classifies diagnoses using Gemini, and saves the result.
    """
    # --- Initialize Gemini API ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"Error: GEMINI_API_KEY environment variable not set. Please set it before running.", file=sys.stderr)
        sys.exit(1) # Exit if API key is missing

    try:
        client = genai.Client(api_key=api_key)
        print(f"Gemini API configured using model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Gemini API: {e}", file=sys.stderr)
        sys.exit(1)
    # --- End Gemini Initialization ---

    # --- Read Input CSV ---
    try:
        print(f"Reading input CSV: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        if DIAGNOSIS_COLUMN not in df.columns:
            print(f"Error: Required column '{DIAGNOSIS_COLUMN}' not found in {input_path}", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {input_path}: {e}", file=sys.stderr)
        sys.exit(1)
    # --- End Read Input CSV ---

    # --- Classify Diagnoses ---
    print("Starting diagnosis classification (this may take a while)...")
    classifications = []
    total_rows = len(df)
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Classifying diagnoses"):
        diagnosis = row[DIAGNOSIS_COLUMN]
        classification = classify_diagnosis_with_gemini(diagnosis, client)
        classifications.append(classification)
    # --- End Classify Diagnoses ---

    # --- Add results and Save Output ---
    df[OUTPUT_COLUMN] = classifications
    print(f"\nClassification complete. Saving results to {output_path}")
    try:
        df.to_csv(output_path, index=False)
        print("Output file saved successfully.")

        # Optional: Print summary
        symptom_count = df[OUTPUT_COLUMN].value_counts().get('Yes', 0)
        disease_count = df[OUTPUT_COLUMN].value_counts().get('No', 0)
        error_count = df[OUTPUT_COLUMN].value_counts().get('Error', 0)
        print("\nSummary:")
        print(f"  Symptoms detected: {symptom_count}")
        print(f"  Disease diagnoses detected: {disease_count}")
        if error_count > 0:
            print(f"  API/Processing Errors: {error_count}")

    except Exception as e:
        print(f"Error writing output file {output_path}: {e}", file=sys.stderr)
    # --- End Save Output ---

if __name__ == "__main__":
    process_diagnoses(INPUT_CSV_PATH, OUTPUT_CSV_PATH) 