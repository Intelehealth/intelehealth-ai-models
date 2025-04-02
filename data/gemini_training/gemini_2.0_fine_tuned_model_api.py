import pandas as pd
import csv
from google import genai
from google.genai import types
import base64
import os
from dotenv import load_dotenv

# Configuration
load_dotenv(dotenv_path='../../ops/.env')

VERTEXAI_PROJECT = os.getenv("VERTEXAI_PROJECT")
VERTEXAI_LOCATION = os.getenv("VERTEXAI_LOCATION")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")
if not all([VERTEXAI_PROJECT, VERTEXAI_LOCATION, MODEL_ENDPOINT]):
    raise ValueError("One or more Vertex AI environment variables (VERTEXAI_PROJECT, VERTEXAI_LOCATION, MODEL_ENDPOINT) are not set in ops/.env")

INPUT_CSV_PATH = "../../data/DDx_database-190-cases-data-cleaned-ayu.csv"
OUTPUT_CSV_PATH = "gemini_results.csv"
CLINICAL_NOTES_COLUMN = "history" # Adjust if your column name is different

def initialize_client():
    """Initializes and returns the Gemini client."""
    try:
        client = genai.Client(
            vertexai=True,
            project=VERTEXAI_PROJECT,
            location=VERTEXAI_LOCATION,
        )
        print("Gemini client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        raise

def generate_text(client, note_text):
    """Generates text based on the provided clinical note using the Gemini model."""
    if not isinstance(note_text, str) or not note_text.strip():
        # print("Skipping empty or invalid note.")
        return "Error: Empty or invalid input note"

    msg_part = types.Part.from_text(text=note_text)
    contents = [types.Content(role="user", parts=[msg_part])]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=8192, # Adjust as needed
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        ],
    )

    try:
        response_chunks = client.models.generate_content_stream(
            model=MODEL_ENDPOINT,
            contents=contents,
            config=generate_content_config,
        )
        full_response = "".join(chunk.text for chunk in response_chunks)
        return full_response
    except types.generation_types.BlockedPromptException as bpe:
        print(f"Warning: Prompt blocked for note. Reason: {bpe}")
        # Truncate note for logging if it's too long
        truncated_note = note_text[:200] + "..." if len(note_text) > 200 else note_text
        print(f"Blocked Note (start): {truncated_note}")
        return "Error: Prompt blocked by safety settings"
    except Exception as e:
        print(f"Error generating content for note: {e}")
        # Truncate note for logging
        truncated_note = note_text[:200] + "..." if len(note_text) > 200 else note_text
        print(f"Problematic Note (start): {truncated_note}")
        return f"Error processing note: {e}"

def main():
    """Main function to read CSV, process notes, and write results."""
    # Construct the absolute path for the input CSV relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from script_dir to get to the project root, then down to data/
    project_root = os.path.dirname(os.path.dirname(script_dir))
    input_csv_full_path = os.path.join(project_root, 'data', os.path.basename(INPUT_CSV_PATH))

    print(f"Attempting to read clinical notes from: {input_csv_full_path}")
    try:
        df = pd.read_csv(input_csv_full_path)
        print(f"Successfully read {len(df)} rows from {input_csv_full_path}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_full_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file {input_csv_full_path}: {e}")
        return

    if CLINICAL_NOTES_COLUMN not in df.columns:
        print(f"Error: Column '{CLINICAL_NOTES_COLUMN}' not found in the input CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        return
    else:
        print(f"Found '{CLINICAL_NOTES_COLUMN}' column.")

    # Handle potential NaN values in the clinical notes column *before* processing
    df[CLINICAL_NOTES_COLUMN] = df[CLINICAL_NOTES_COLUMN].fillna('')

    # Get original headers
    original_headers = df.columns.tolist()
    output_headers = original_headers + ['Top 1 ddx FT'] # Add prediction column header

    try:
        client = initialize_client()
    except Exception:
        print("Failed to initialize Gemini client. Exiting.")
        return

    # Construct the absolute path for the output CSV relative to the script's directory
    output_csv_full_path = os.path.join(script_dir, OUTPUT_CSV_PATH)
    print(f"Preparing to write results to: {output_csv_full_path}")

    try:
        with open(output_csv_full_path, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.writer(outfile)
            # Write the new header
            csv_writer.writerow(output_headers)
            print("Output CSV header written.")

            total_rows = len(df)
            print(f"Starting processing of {total_rows} rows...")
            for index, row in df.iterrows():
                clinical_note = row[CLINICAL_NOTES_COLUMN]
                original_row_values = row.tolist() # Get all original values

                # Optional: Add a check here if you strictly want to skip rows where the note was originally NaN/empty
                # if not clinical_note:
                #    print(f"Skipping row {index + 1}/{total_rows} due to empty note.")
                #    # Write original row + skipped message or skip writing the row
                #    csv_writer.writerow(original_row_values + ["Skipped: Empty input"])
                #    continue

                print(f"Processing row {index + 1}/{total_rows}...")
                generated_output = generate_text(client, clinical_note)

                # Combine original row values with the generated output
                output_row_data = original_row_values + [generated_output]

                # Write row to output CSV immediately
                csv_writer.writerow(output_row_data)
                # Flush buffer to ensure data is written
                outfile.flush()

                # More granular progress, remove if too verbose
                # print(f"Row {index + 1} processed and saved.")

            print(f"\nProcessing complete. {total_rows} rows processed.")
            print(f"Results saved to {output_csv_full_path}")

    except Exception as e:
        print(f"An error occurred during processing or writing to CSV: {e}")
        print("Processing stopped prematurely. Check the output file for partial results.")

# Add the standard Python entry point check
if __name__ == "__main__":
    main()

# The original generate() function and its call are replaced by the main() logic.