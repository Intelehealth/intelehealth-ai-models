import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.DDxQwenLocalModule import DDxQwenLocalModule
from peft import PeftModel
import argparse
import os
import logging
import pandas as pd
import csv
import dspy
import random
from utils.metric_utils import openai_qwen_local_llm_judge
from utils.lm_studio_infer import lm_studio_openai_client_qwen2_5_3b_infer
from dotenv import load_dotenv
from dspy import OpenAI


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(
    "ops/.env"
)


# --- Constants and Helper Functions from batch_inference_nas_v2.py ---

SYSTEM_PROMPT = """ # This might be implicitly handled by the model's fine-tuning
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Define the fixed instruction part of the prompt
# This instruction might be better incorporated directly into the dspy.Signature if not part of the base model's fine-tuning
INSTRUCTION_PROMPT = """Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
Based on patient history, symptoms, physical exam findings, and demographics, provide top 5 differential diagnoses ranked by likelihood."""

def setup_device():
    """Sets up the device to use (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logging.info("MPS device found and PyTorch built with MPS enabled. Using MPS.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logging.info("CUDA device found. Using CUDA.")
        return torch.device("cuda")
    else:
        logging.warning("Neither MPS nor CUDA available. Using CPU.")
        return torch.device("cpu")

def load_model_and_tokenizer(base_model_id, adapter_path, device):
    """Loads the base model, tokenizer, and applies the PEFT adapter."""
    logging.info(f"Loading base tokenizer: {base_model_id}")
    # Consider adding trust_remote_code=True if needed for the specific Qwen model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    logging.info(f"Loading base model: {base_model_id}")
    # Load the model onto the specified device directly if possible
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map={ "": device } # Try device_map first for better memory management
            # trust_remote_code=True # Uncomment if required
        )
        logging.info(f"Base model loaded using device_map onto: {device}")
    except Exception:
        logging.warning(f"Could not use device_map. Loading model to CPU first then moving to {device}.")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            # trust_remote_code=True # Uncomment if required
        ).to(device)
        logging.info(f"Base model loaded and moved to device: {base_model.device}")

    logging.info(f"Loading PEFT adapter from: {adapter_path}")
    if not os.path.isdir(adapter_path):
         raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    try:
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        # Ensure the final adapted model is on the correct device (might be redundant if base model already on device)
        model = model.to(device)
        model.eval()
        logging.info("PEFT adapter loaded and applied successfully.")
    except Exception as e:
         logging.error(f"Error loading PEFT adapter: {e}", exc_info=True)
         raise RuntimeError(f"Could not load PEFT adapter from {adapter_path}") from e

    logging.info(f"Final model is on device: {model.device}")
    # Ensure tokenizer padding settings are correct
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Often recommended for decoder-only models
        logging.info("Set tokenizer pad_token to eos_token and padding_side to left")

    return model, tokenizer


# --- Data Loading for DSPy --- 
def load_data_for_dspy(csv_path, notes_column, target_column, num_rows=None):
    """Loads data from CSV and converts it into dspy.Example objects."""
    logging.info(f"Loading data for DSPy from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path, nrows=num_rows)
    logging.info(f"Loaded {len(df)} rows from the CSV.")

    if notes_column not in df.columns:
        raise ValueError(f"Notes column '{notes_column}' not found in the CSV. Available columns: {df.columns.tolist()}")
    if target_column and target_column not in df.columns:
         raise ValueError(f"Target column '{target_column}' not found in the CSV. Available columns: {df.columns.tolist()}")

    examples = []
    for index, row in df.iterrows():
        notes = row[notes_column]
        target = row[target_column] if target_column else "" # Handle case where target might not exist/be needed

        if pd.isna(notes):
            logging.warning(f"Skipping row {index+1} due to missing clinical note.")
            continue
        if target_column and pd.isna(target):
             logging.warning(f"Skipping row {index+1} due to missing target diagnosis.")
             continue

        if target_column:
            # Create example with input and known output (for training/compilation)
            example = dspy.Example(clinical_notes=notes, differential_diagnoses=target).with_inputs('clinical_notes')
        else:
            # Create example with only input (for pure inference)
             example = dspy.Example(clinical_notes=notes).with_inputs('clinical_notes')
        examples.append(example)

    logging.info(f"Created {len(examples)} dspy.Example objects.")
    return examples


# --- Main Execution Logic --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a DSPy program for clinical notes using a local Qwen PEFT model via dspy.HFModel.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PEFT adapter directory.")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model ID.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV training data.")
    parser.add_argument("--notes_column", type=str, default="Clinical Notes", help="CSV column for clinical notes.")
    parser.add_argument("--target_column", type=str, required=True, help="CSV column for target differential diagnoses (for training).")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to load from CSV for training.")
    parser.add_argument("--output_program_path", type=str, default="compiled_ddx_program.json", help="Path to save the compiled DSPy program.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials for MIPROv2 compilation.")
    parser.add_argument("--num_candidates", type=int, default=4, help="Number of candidates per trial for MIPROv2.")
    parser.add_argument("--max_demos", type=int, default=4, help="Max labeled demos for MIPROv2.")
    # Add arguments for MIPRO parameters if needed

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    try:
        # 1 & 2: Local Model Loading (Commented out as we're using LM Studio)
        # device = setup_device()
        # model, tokenizer = load_model_and_tokenizer(...)

        # 3. Configure DSPy LM using dspy.OpenAI for LM Studio
        logging.info("Configuring dspy.OpenAI to connect to LM Studio...")

        # --- IMPORTANT: Find this identifier in LM Studio's Local Server tab ---
        # Replace this placeholder with the actual identifier shown in LM Studio after loading the model
        LM_STUDIO_MODEL_IDENTIFIER = "indhic-ai/Qwen2.5-3B-Instruct-Qlora-GGUF" # e.g., "Qwen/Qwen2.5-3B-Instruct-Qlora-GGUF" or similar

        # --- Verify this matches your LM Studio server address ---
        LM_STUDIO_API_BASE = "http://localhost:1234/v1/"

        if LM_STUDIO_MODEL_IDENTIFIER == "placeholder_from_lm_studio":
             logging.warning("Please update LM_STUDIO_MODEL_IDENTIFIER with the actual model ID from LM Studio!")
             # You might want to raise an error here or exit if it's not set
             # raise ValueError("LM_STUDIO_MODEL_IDENTIFIER not set!")

        lm = OpenAI(
            model=LM_STUDIO_MODEL_IDENTIFIER,
            api_base=LM_STUDIO_API_BASE,
            api_key='no-key-required', # Use '' or None if this doesn't work
            model_type="chat",        # Usually correct for Instruct models
            temperature=1.0,          # Adjust as needed
            # max_tokens=1024         # Optional: Add a limit
        )
        dspy.configure(lm=lm) # Configure DSPy's default LM
        logging.info(f"DSPy LM configured for LM Studio model: {LM_STUDIO_MODEL_IDENTIFIER} at {LM_STUDIO_API_BASE}")

        # 4. Load Data
        trainset = load_data_for_dspy(args.csv_path, args.notes_column, args.target_column, args.num_rows)
        print(trainset)

        if not trainset:
            raise ValueError("Trainset is empty. Check CSV path, columns, and content.")
        random.shuffle(trainset)
        logging.info(f"Loaded and shuffled {len(trainset)} training examples.")
        # print(f"First training example: {trainset[0]}")

        # 5. Define the DSPy Program to Compile
        program_to_compile = DDxQwenLocalModule()

        # 6. Define Metric
        metric_to_use = openai_qwen_local_llm_judge

        # 7. Setup Teleprompter (MIPROv2)
        # Adjust num_threads based on your system capabilities
        teleprompter = dspy.MIPROv2(metric=openai_qwen_local_llm_judge, num_threads=2, num_candidates=args.num_candidates, max_labeled_demos=args.max_demos)


        # 8. Compile
        logging.info(f"Starting compilation with {args.num_trials} trials...")
        optimized_program = teleprompter.compile(
            program_to_compile, 
            trainset=trainset, 
            num_trials=args.num_trials
        ) 
        logging.info("Compilation finished.")

        # 9. Save Compiled Program
        optimized_program.save(args.output_program_path)
        logging.info(f"Compiled program saved to {args.output_program_path}")

        # Optional: Inspect the compiled program
        # print("\n--- Compiled Program Structure ---")
        # print(compiled_program)
        # print("\n--- Example Prediction with Compiled Program ---")
        # if trainset:
        #     example_pred = compiled_program(clinical_notes=trainset[0].clinical_notes)
        #     print(f"Input Notes: {trainset[0].clinical_notes[:200]}...")
        #     print(f"Predicted Diagnoses: {example_pred.differential_diagnoses}")

    except FileNotFoundError as e:
        logging.error(f"File Error: {e}. Please ensure paths exist.")
    except ValueError as e:
        logging.error(f"Configuration or Data Error: {e}")
    except RuntimeError as e:
         logging.error(f"Runtime Error (likely CUDA/MPS OOM or model issue): {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) 