import pandas as pd
import json
from tqdm.auto import tqdm
import re
import os
import traceback
from sklearn.model_selection import train_test_split

# --- Constants ---
# These can be adjusted or moved into the function/config if needed
MIN_DIAGNOSIS_LEN = 3
MIN_PROMPT_LEN = 10 # Min length for history

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def prepare_grpo_data(csv_path: str):
    """
    Loads data from a CSV, cleans it, generates placeholder reasoning,
    formats it into chosen responses, and calculates average answer length.

    Args:
        csv_path (str): Path to the input CSV file.

    Returns:
        tuple: A tuple containing:
            - list: The processed data (list of dictionaries).
            - float: The calculated average answer length.
            - str: The actual path from which the CSV was successfully loaded.
        Returns (None, None, None) if loading or processing fails critically.
    """
    print(f"Attempting to load data from {csv_path}...")
    actual_csv_path = csv_path
    try:
        # Load CSV (handling encoding/fallback paths)
        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            print("UTF-8 failed, trying latin1 encoding...")
            df = pd.read_csv(csv_path, encoding='latin1')
        except FileNotFoundError:
             alt_csv_path = "data/DDx_database-190-cases-data-cleaned-ayu.csv" # Define potential alt path
             print(f"Original path failed, trying alternative path: {alt_csv_path}")
             try:
                  df = pd.read_csv(alt_csv_path)
                  actual_csv_path = alt_csv_path # Update path if successful
             except UnicodeDecodeError:
                  print("UTF-8 failed on alt path, trying latin1 encoding...")
                  df = pd.read_csv(alt_csv_path, encoding='latin1')
                  actual_csv_path = alt_csv_path # Update path if successful
             except FileNotFoundError:
                  print(f"Error: Input CSV file not found at {csv_path} or {alt_csv_path}")
                  return None, None, None # Indicate failure

        print(f"Successfully loaded data from {actual_csv_path}")

        # --- Data Cleaning & Pre-calculation ---
        df.columns = df.columns.str.strip()

        required_cols = ['history', 'diagnosis']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in {actual_csv_path}: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return None, None, actual_csv_path # Return path for context, but indicate failure

        # Fill NaNs and convert types
        df['history'] = df['history'].fillna('').astype(str)
        df['diagnosis'] = df['diagnosis'].fillna('N/A').astype(str)

        # Filter rows
        initial_rows = len(df)
        df = df[df['diagnosis'].str.len() >= MIN_DIAGNOSIS_LEN]
        df = df[df['history'].str.len() >= MIN_PROMPT_LEN]
        print(f"Filtered rows: {initial_rows} -> {len(df)}")

        if len(df) == 0:
             print("Error: No valid rows remaining after filtering.")
             # Return empty list, 0 avg_length, and the path used
             return [], 0.0, actual_csv_path

        # Calculate Average Length from 'diagnosis' column
        avg_length = df['diagnosis'].apply(len).mean()
        print(f"Calculated Average Answer Length (AVG_LENGTH): {avg_length:.2f}")

        # --- Create GRPO Pairs (Chosen only for now) ---
        processed_data = []
        print("Processing rows to format chosen data (rejected pair creation pending)...")

        question_header = """Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
            Based on patient history, symptoms, physical exam findings, and demographics, provide top 5 differential diagnoses ranked by likelihood."""

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            prompt_text = row['history']
            ground_truth_diagnosis = row['diagnosis']

            if ground_truth_diagnosis == 'N/A':
                continue
            # Define a mapping function to create the prompt and answer fields
            def map_fn(ground_truth_diagnosis, prompt_text):
                return {
                    'prompt': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content':  prompt_text + " " + question_header}
                    ],
                    'answer': ground_truth_diagnosis,  # The reference answer
                    'question': prompt_text + " " + question_header
                }


            processed_data.append(map_fn(ground_truth_diagnosis, prompt_text))

        print(f"Formatted {len(processed_data)} chosen responses (rejected strategy pending).")
        return processed_data, avg_length, actual_csv_path

    except Exception as e:
        print(f"An unexpected error occurred during data preparation: {e}")
        traceback.print_exc()
        # Try to return the path used even if error occurred later
        return None, None, locals().get('actual_csv_path', csv_path)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Configuration for script execution
    DEFAULT_CSV_PATH = "/Users/bsb/work/intelehealth-ai-models/data/DDx_database-190-cases-data-cleaned-ayu.csv"
    TRAIN_OUTPUT_PATH = "train_grpo_data.jsonl"
    VALIDATION_OUTPUT_PATH = "validation_grpo_data.jsonl"
    TEST_OUTPUT_PATH = "test_grpo_data.jsonl"
    AVG_LENGTH_FILE = "avg_length.txt"

    # Call the preparation function
    processed_data, avg_length, loaded_csv_path = prepare_grpo_data(DEFAULT_CSV_PATH)

    # Check if processing was successful before writing output
    if processed_data is not None and avg_length is not None:
        if not processed_data:
            print("No data was processed (e.g., all rows filtered out). Creating empty split files.")
            # Create empty files
            for path in [TRAIN_OUTPUT_PATH, VALIDATION_OUTPUT_PATH, TEST_OUTPUT_PATH]:
                 try:
                     open(path, 'w').close()
                     print(f"Created empty file: {path}")
                 except Exception as e:
                     print(f"Error creating empty file {path}: {e}")
        else:
            print(f"Splitting {len(processed_data)} items into train/validation/test sets (80/10/10)...")
            # Split data: 80% train, 20% temp
            train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
            # Split temp: 50% validation (10% total), 50% test (10% total)
            validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

            print(f"Split complete: Train={len(train_data)}, Validation={len(validation_data)}, Test={len(test_data)}")

            # --- Save Split Data ---
            split_files = {
                TRAIN_OUTPUT_PATH: train_data,
                VALIDATION_OUTPUT_PATH: validation_data,
                TEST_OUTPUT_PATH: test_data
            }

            for output_path, data_split in split_files.items():
                print(f"Saving {len(data_split)} items to {output_path}...")
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for item in data_split:
                            f.write(json.dumps(item) + '\n')
                    print(f"Saved data to {output_path}")
                except Exception as e:
                    print(f"Error saving data to {output_path}: {e}")

        # --- Save Average Length ---
        print(f"Saving average length ({avg_length:.2f}) to {AVG_LENGTH_FILE}...")
        try:
            with open(AVG_LENGTH_FILE, 'w') as f:
                f.write(str(avg_length))
            print(f"Saved average length to {AVG_LENGTH_FILE}")
        except Exception as e:
            print(f"Warning: Could not save average length to {AVG_LENGTH_FILE}: {e}")

        print("Data preparation and splitting script finished successfully.")
    else:
        print("Data preparation script failed. No output files were written.") 