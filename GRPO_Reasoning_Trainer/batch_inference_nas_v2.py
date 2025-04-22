import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os
import logging
import pandas as pd # Added for CSV handling
import csv # Added for CSV writing

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Define the fixed instruction part of the prompt
INSTRUCTION_PROMPT = """Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
Based on patient history, symptoms, physical exam findings, and demographics, provide top 5 differential diagnoses ranked by likelihood."""

def setup_device():
    """Sets up the device to use (MPS or CPU)."""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            logging.info("MPS device found and PyTorch built with MPS enabled. Using MPS.")
            return torch.device("mps")
        else:
            logging.warning("MPS not available because the current PyTorch install was not built with MPS enabled. Using CPU.")
            return torch.device("cpu")
    else:
        logging.warning("MPS device not found. Using CPU.")
        return torch.device("cpu")

def load_model_and_tokenizer(base_model_id, adapter_path, device):
    """Loads the base model, tokenizer, and applies the PEFT adapter."""
    logging.info(f"Loading base tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    logging.info(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map={"": device}
    )
    logging.info(f"Base model loaded onto device: {base_model.device}")

    logging.info(f"Loading PEFT adapter from: {adapter_path}")
    if not os.path.isdir(adapter_path):
         raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    try:
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        model = model.to(device)
        model.eval()
        logging.info("PEFT adapter loaded and applied successfully.")
    except Exception as e:
         logging.error(f"Error loading PEFT adapter: {e}", exc_info=True)
         raise RuntimeError(f"Could not load PEFT adapter from {adapter_path}") from e

    logging.info(f"Final model is on device: {model.device}")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, device, max_new_tokens=2000):
    """Generates a response from the model given a prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    logging.info("Applying chat template...")
    try:
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
    except Exception as e:
        logging.error(f"Error applying chat template: {e}", exc_info=True)
        logging.warning("Falling back to basic tokenization due to chat template error.")
        tokenized_chat = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    prompt_len = tokenized_chat.shape[1]
    logging.info(f"Input tokens length: {prompt_len}")

    logging.info(f"Generating response (max_new_tokens={max_new_tokens})...")
    with torch.no_grad():
        outputs = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[:, prompt_len:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    logging.info("Generation complete.")
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on clinical notes from a CSV file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen2-2.5B-GRPO-test",
        help="Path to the directory containing the PEFT adapter."
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model ID used during fine-tuning."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../data/NAS_V2_CleanUpDataset_v0.2.csv", # Default path relative to the script location
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--notes_column",
        type=str,
        default="Clinical Notes", # Name of the column containing clinical notes
        help="Name of the column containing the clinical notes in the CSV."
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="The 0-based index of the first row to process from the CSV file."
    )
    parser.add_argument(
        "--end_row",
        type=int,
        default=None, # Process until the end if not specified
        help="The 0-based index of the row to stop processing at (exclusive). If not provided, process until the end."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1000,
        help="Maximum number of new tokens to generate for each response."
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default="batch_inference_output.csv", # Default output file name
        help="Path to save the output CSV file with results."
    )

    args = parser.parse_args()

    try:
        # Setup device
        device = setup_device()

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.base_model_id, args.model_path, device)

        # Load CSV data
        logging.info(f"Loading CSV data from: {args.csv_path}")
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {args.csv_path}")
        # Read the entire CSV first
        full_df = pd.read_csv(args.csv_path)
        logging.info(f"Loaded {len(full_df)} total rows from the CSV.")

        # Select the specified range of rows
        start = args.start_row
        end = args.end_row if args.end_row is not None else len(full_df)

        if start < 0:
            logging.warning(f"start_row ({start}) is less than 0, adjusting to 0.")
            start = 0
        if end > len(full_df):
            logging.warning(f"end_row ({end}) is greater than the number of rows ({len(full_df)}), adjusting to {len(full_df)}.")
            end = len(full_df)
        if start >= end:
             raise ValueError(f"start_row ({start}) must be less than end_row ({end}). No rows to process.")


        df_slice = full_df.iloc[start:end]
        logging.info(f"Processing rows from index {start} to {end-1} (total {len(df_slice)} rows).")


        if args.notes_column not in df_slice.columns:
            raise ValueError(f"Column '{args.notes_column}' not found in the CSV file. Available columns: {df_slice.columns.tolist()}")

        # Prepare output CSV
        output_columns = df_slice.columns.tolist() + ['Generated Response']
        logging.info(f"Saving results row by row to: {args.output_csv_path}")

        try:
            with open(args.output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
                csv_writer = csv.writer(outfile)
                csv_writer.writerow(output_columns) # Write header

                # Process each row in the slice and write to CSV
                # Use df_slice.iterrows() which yields (original_index, row_data)
                for original_index, row in df_slice.iterrows():
                    # original_index corresponds to the index in the full_df
                    current_row_number = original_index + 1 # 1-based row number for logging

                    clinical_note = row[args.notes_column]
                    if pd.isna(clinical_note):
                        logging.warning(f"Skipping row {current_row_number} (original index {original_index}) due to missing clinical note.")
                        # Write original row with empty response if note is missing? Optional.
                        # original_data = row.tolist()
                        # csv_writer.writerow(original_data + ['']) # Example: write skipped row
                        continue

                    # Construct the full prompt - Combining instruction and notes
                    # Using f-string for clarity
                    full_prompt = f"{clinical_note}\n\n{INSTRUCTION_PROMPT}"


                    print("\n" + "="*20 + f" Processing Row {current_row_number} (Index {original_index}) " + "="*20)
                    # Limit printing long notes for brevity in console
                    print(f"\n[CLINICAL NOTE PREVIEW]:\n{clinical_note[:500]}...")


                    # Generate response
                    response = generate_response(model, tokenizer, full_prompt, device, args.max_new_tokens)

                    print(f"\n[GENERATED RESPONSE]:\n{response}")
                    print("\n" + "="*60) # Adjusted separator length

                    # Write row to output CSV
                    original_data = row.tolist()
                    csv_writer.writerow(original_data + [response])
                    logging.info(f"Processed and wrote row {current_row_number} (Index {original_index}) to {args.output_csv_path}")


        except IOError as e:
            logging.error(f"Error writing to output CSV file {args.output_csv_path}: {e}", exc_info=True)
            # Re-raise the exception after logging if needed, or handle appropriately
            raise

    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please ensure the specified paths exist.")
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during batch inference: {e}", exc_info=True) 