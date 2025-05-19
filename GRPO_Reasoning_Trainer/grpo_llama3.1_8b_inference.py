import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import logging
import pandas as pd # Added for CSV handling
import os # Added for file checks
import csv # Added for CSV writing
from tqdm import tqdm
import time
from datetime import datetime
import re

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the fixed instruction part of the prompt
INSTRUCTION_PROMPT = """Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
Based on patient history, symptoms, physical exam findings, and demographics, provide top 5 differential diagnoses ranked by likelihood."""

# Global cache for the model and pipeline
_model_cache = {
    'generator': None,
    'base_model_id': None,
    'adapter_model_id': None,
    'device': None
}

def setup_device():
    """Sets up the device to use (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        logging.info("CUDA device found. Using CUDA.")
        return "cuda"
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            logging.info("MPS device found and PyTorch built with MPS enabled. Using MPS.")
            return "mps"
        else:
            logging.warning("MPS available but PyTorch not built with MPS. Using CPU.")
            return "cpu"
    else:
        logging.warning("Neither CUDA nor MPS device found. Using CPU.")
        return "cpu"

def load_and_merge_model(base_model_id, adapter_model_id, device_option):
    """
    Loads and merges the base model with adapter, caching the result for reuse.
    
    Args:
        base_model_id (str): The Hugging Face model ID for the base model.
        adapter_model_id (str): The Hugging Face model ID for the adapter model.
        device_option (str): The device to run the model on.
        
    Returns:
        tuple: (generator, device_used) - The pipeline generator and the device it's running on.
    """
    global _model_cache
    
    # Check if we can use cached model
    if (_model_cache['generator'] is not None and 
        _model_cache['base_model_id'] == base_model_id and 
        _model_cache['adapter_model_id'] == adapter_model_id and 
        _model_cache['device'] == device_option):
        logging.info("Using cached model and pipeline")
        return _model_cache['generator'], device_option

    if device_option == "auto":
        selected_device_str = setup_device()
    else:
        selected_device_str = device_option

    logging.info(f"Initializing model and pipeline with device option: {selected_device_str}")
    try:
        # Set device_map based on selected device
        if selected_device_str == "cpu":
            device_map = "cpu"
        elif selected_device_str.startswith("cuda"):
            device_map = "auto"  # Let accelerate handle GPU placement
        else:
            device_map = "auto"  # Default to auto for other cases

        # Load base model with progress bar
        with tqdm(total=3, desc="Loading model", unit="step") as pbar:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map=device_map,
                torch_dtype="auto"
            )
            pbar.update(1)
            pbar.set_description("Loading base model")
            
            # Load adapter and merge it with base model
            model = PeftModel.from_pretrained(base_model, adapter_model_id)
            pbar.update(1)
            pbar.set_description("Loading adapter")
            
            model = model.merge_and_unload()  # Merge adapter weights with base model
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            pbar.update(1)
            pbar.set_description("Merging weights")
        
        # Create pipeline with the merged model
        logging.info("Creating pipeline with merged model...")
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype="auto"
        )
        
        # Cache the generator and model info
        _model_cache['generator'] = generator
        _model_cache['base_model_id'] = base_model_id
        _model_cache['adapter_model_id'] = adapter_model_id
        _model_cache['device'] = device_option
        
        logging.info("Pipeline initialized and cached successfully.")
        return generator, device_option
        
    except Exception as e:
        logging.error(f"Error initializing pipeline: {e}", exc_info=True)
        if selected_device_str != "cpu":
            logging.warning(f"Falling back to CPU due to error with device {selected_device_str}.")
            try:
                # Load base model on CPU with progress bar
                with tqdm(total=3, desc="Loading model on CPU", unit="step") as pbar:
                    # Load base model on CPU
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_id,
                        device_map="cpu",
                        torch_dtype="auto"
                    )
                    pbar.update(1)
                    pbar.set_description("Loading base model on CPU")
                    
                    # Load and merge adapter on CPU
                    model = PeftModel.from_pretrained(base_model, adapter_model_id)
                    pbar.update(1)
                    pbar.set_description("Loading adapter on CPU")
                    
                    model = model.merge_and_unload()  # Merge adapter weights with base model
                    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                    pbar.update(1)
                    pbar.set_description("Merging weights on CPU")
                
                # Create pipeline with merged model
                logging.info("Creating pipeline with merged model on CPU...")
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype="auto"
                )
                
                # Cache the generator and model info
                _model_cache['generator'] = generator
                _model_cache['base_model_id'] = base_model_id
                _model_cache['adapter_model_id'] = adapter_model_id
                _model_cache['device'] = "cpu"
                
                logging.info("Pipeline initialized and cached successfully on CPU.")
                return generator, "cpu"
            except Exception as e_cpu:
                logging.error(f"Error initializing pipeline on CPU: {e_cpu}", exc_info=True)
                raise RuntimeError(f"Could not initialize pipeline on {device_option} or CPU.") from e_cpu
        else:
            raise RuntimeError(f"Could not initialize pipeline on CPU.") from e

def generate_response(base_model_id, adapter_model_id, system_prompt, question, device_option, max_new_tokens):
    """
    Generates a response using a base model with an adapter model from Hugging Face Hub.
    Uses cached model if available.

    Args:
        base_model_id (str): The Hugging Face model ID for the base model.
        adapter_model_id (str): The Hugging Face model ID for the adapter model.
        system_prompt (str): The system prompt to contextualize the model.
        question (str): The question to ask the model.
        device_option (str): The device to run the model on ('cuda', 'mps', 'cpu', or 'auto').
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated text from the model.
    """
    # Get or load the generator
    generator, _ = load_and_merge_model(base_model_id, adapter_model_id, device_option)

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    
    # Combine the user's question with the fixed instruction prompt
    user_content = f"{question}\n\n{INSTRUCTION_PROMPT}"
    messages.append({"role": "user", "content": user_content})

    logging.info(f"Generating response (max_new_tokens={max_new_tokens})...")
    try:
        output = generator(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    except Exception as e:
        logging.error(f"Error during generation: {e}", exc_info=True)
        raise

    # output is a list of dictionaries
    if output and isinstance(output, list) and isinstance(output[0], dict) and "generated_text" in output[0]:
        generated_text = output[0]["generated_text"]
        logging.info("Generation complete.")
        return generated_text
    else:
        logging.error(f"Unexpected output format from generator: {output}")
        return "Error: Could not parse generated output."

def extract_tagged_content(text, tag):
    """
    Extract content between specified XML-style tags.
    
    Args:
        text (str): The text to search in
        tag (str): The tag name without angle brackets
        
    Returns:
        str: The content between tags, or empty string if not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a Llama model with adapter from Hugging Face Hub.")
    
    # Model loading arguments
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model ID for the base model."
    )
    parser.add_argument(
        "--adapter_model_id",
        type=str,
        required=True,
        help="Hugging Face model ID for the adapter model (e.g., 'organization/model-name-adapter')."
    )

    # Single question arguments
    parser.add_argument(
        "--question",
        type=str,
        default=None, # No default, required if not using CSV
        help="The question to ask the model. Required if --csv_path is not provided."
    )

    # CSV processing arguments
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to the input CSV file for batch processing."
    )
    parser.add_argument(
        "--notes_column",
        type=str,
        default="Clinical Notes", # Default column name
        help="Name of the column containing the text/notes in the input CSV."
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
        "--output_csv_path",
        type=str,
        default="llama_batch_inference_output.csv", # Default output file name
        help="Path to save the output CSV file with results when using --csv_path."
    )

    # Common arguments
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="""
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""",
        help="The system prompt to send to the model."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256, 
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run the model on. 'auto' will attempt CUDA, then MPS, then CPU."
    )

    args = parser.parse_args()

    try:
        if args.csv_path: # Batch processing mode
            logging.info(f"Starting batch inference from CSV: {args.csv_path}")
            if not os.path.exists(args.csv_path):
                raise FileNotFoundError(f"CSV file not found at: {args.csv_path}")

            full_df = pd.read_csv(args.csv_path)
            logging.info(f"Loaded {len(full_df)} total rows from the CSV.")

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
                raise ValueError(f"Column '{args.notes_column}' not found in the CSV. Available columns: {df_slice.columns.tolist()}")

            # Add timestamp, processing time, and parsed response columns
            output_columns = df_slice.columns.tolist() + [
                'Generated Response',
                'Reasoning',
                'Answer',
                'Processing Time (seconds)',
                'Timestamp'
            ]
            logging.info(f"Will write results to: {args.output_csv_path}")

            # Create progress bar for CSV processing
            pbar = tqdm(df_slice.iterrows(), total=len(df_slice), desc="Processing CSV rows", unit="row")
            
            # Open CSV file in append mode to ensure we don't lose data if the script is interrupted
            file_exists = os.path.exists(args.output_csv_path)
            mode = 'a' if file_exists else 'w'
            
            with open(args.output_csv_path, mode, newline='', encoding='utf-8') as outfile:
                csv_writer = csv.writer(outfile)
                
                # Write header only if this is a new file
                if not file_exists:
                    csv_writer.writerow(output_columns)
                    logging.info("Created new output CSV file with headers")
                else:
                    logging.info("Appending to existing output CSV file")

                for original_index, row in pbar:
                    current_row_number_for_logging = original_index + 1 
                    note_content = row[args.notes_column]
                    pbar.set_description(f"Processing row {current_row_number_for_logging}")
                    
                    # Start timing
                    start_time = time.time()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    try:
                        if pd.isna(note_content):
                            processing_time = time.time() - start_time
                            logging.warning(f"Skipping row {current_row_number_for_logging} (original index {original_index}) due to missing content in '{args.notes_column}'.")
                            original_data = row.tolist()
                            csv_writer.writerow(original_data + [
                                'SKIPPED - MISSING NOTE',
                                '',  # Reasoning
                                '',  # Answer
                                f"{processing_time:.2f}",
                                timestamp
                            ])
                            outfile.flush()  # Ensure immediate write
                            logging.info(f"Wrote skipped row {current_row_number_for_logging} to CSV (took {processing_time:.2f} seconds)")
                            continue
                        
                        question_for_model = str(note_content) # Ensure it's a string

                        print("\n" + "="*20 + f" Processing CSV Row {current_row_number_for_logging} (Index {original_index}) " + "="*20)
                        if args.system_prompt and args.system_prompt.strip():
                            print(f"System Prompt: {args.system_prompt}")
                        print(f"Input Note (from column '{args.notes_column}'):\n{question_for_model[:500]}...")
                        print(f"Instruction Prompt (appended to input note):\n{INSTRUCTION_PROMPT}")
                        
                        # Generate response
                        generated_text = generate_response(
                            args.base_model_id,
                            args.adapter_model_id,
                            args.system_prompt, 
                            question_for_model, 
                            args.device, 
                            args.max_new_tokens
                        )
                        
                        # Extract reasoning and answer from the generated text
                        reasoning = extract_tagged_content(generated_text, "reasoning")
                        answer = extract_tagged_content(generated_text, "answer")
                        
                        # Calculate processing time
                        processing_time = time.time() - start_time
                        
                        print(f"\n[GENERATED RESPONSE for row {current_row_number_for_logging}]:\n{generated_text}")
                        print(f"\nProcessing time: {processing_time:.2f} seconds")
                        print("\n" + "="*70)

                        # Write the row immediately after generation
                        original_data = row.tolist()
                        csv_writer.writerow(original_data + [
                            generated_text,
                            reasoning,
                            answer,
                            f"{processing_time:.2f}",
                            timestamp
                        ])
                        outfile.flush()  # Ensure immediate write to disk
                        logging.info(f"Successfully processed and wrote row {current_row_number_for_logging} to CSV (took {processing_time:.2f} seconds)")
                        
                    except Exception as row_error:
                        # Calculate processing time even for failed rows
                        processing_time = time.time() - start_time
                        # Log the error but continue processing other rows
                        logging.error(f"Error processing row {current_row_number_for_logging}: {row_error}", exc_info=True)
                        # Write error information to CSV
                        original_data = row.tolist()
                        csv_writer.writerow(original_data + [
                            f'ERROR: {str(row_error)}',
                            '',  # Reasoning
                            '',  # Answer
                            f"{processing_time:.2f}",
                            timestamp
                        ])
                        outfile.flush()  # Ensure immediate write
                        logging.info(f"Wrote error row {current_row_number_for_logging} to CSV (took {processing_time:.2f} seconds)")
                        continue  # Continue with next row
            
            logging.info(f"Batch inference complete. All results saved to {args.output_csv_path}")

        elif args.question: # Single question mode
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"User Question: {args.question}")
            if args.system_prompt and args.system_prompt.strip():
                print(f"System Prompt: {args.system_prompt}")
            print(f"Instruction Prompt (appended to user question):\n{INSTRUCTION_PROMPT}")
            
            generated_text = generate_response(
                args.base_model_id,
                args.adapter_model_id,
                args.system_prompt, 
                args.question, 
                args.device, 
                args.max_new_tokens
            )
            
            # Extract reasoning and answer from the generated text
            reasoning = extract_tagged_content(generated_text, "reasoning")
            answer = extract_tagged_content(generated_text, "answer")
            
            processing_time = time.time() - start_time
            print("\nModel Response:")
            print(generated_text)
            print("\nExtracted Reasoning:")
            print(reasoning)
            print("\nExtracted Answer:")
            print(answer)
            print(f"\nProcessing time: {processing_time:.2f} seconds")
            print(f"Timestamp: {timestamp}")
        else:
            parser.error("Either --csv_path (for batch processing) or --question (for single inference) must be provided.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}. Please ensure the specified paths exist.")
    except ValueError as e:
        logging.error(f"Configuration or data error: {e}")
    except RuntimeError as e:
        logging.error(f"A runtime error occurred during model processing: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) 