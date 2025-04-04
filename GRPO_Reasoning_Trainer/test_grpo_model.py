import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os
import logging

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
# Define the default multi-line prompt
default_clinical_prompt = """Female, 89 years
►Difficulty breathing:
• Severity - When walking with people of own age on level ground.
• Aggravating Factors - Excercise.
• Relieving Factors - Gets better after waking up.
►Fatigue & General weakness:
• Timing - At times.
►Oedema:
• Site - Localized - Chronic oedema legs 1+ .
►Palpitations:
• Additional information - She has occasional palpitations .

c. = Associated with, s. = Not associated with, h/o = History of
 Family History

 Past Medical History

• Allergies - Allergy to Sulfa drugs and penicillin ..
• Medication history - Amlodipine besylate ,5 mg/day . Calcium + Vit D , twice daily . Polyethylene glycol powder , 17 Gm in 8 oz liquid daily . Frusemide 4o mg, 40 mg /day. Escitalopram, 20mg /day.
Prednisone 10 mg/day
Omeprazole 20mg/day
Cinacalcet 30mg/day
Simvastatin 20mg at bedtime
Tiotropium oral inhalation 1 cap per inhalation device daily.
Vit B12 1000 might twice daily.
Enteric coated aspirin 81 mg/day.


 Vitals


Temp:        97.39 F        Height:        168 cm        Weight:        71 kg        BMI:        25.16        SP02:        94 %        BP:        132/84        HR:        72        RR:        20

 On Examination

General exams:
• Eyes: Jaundice-Don't know.
• Eyes: Pallor-Don't know.
• Arm-Pinch skin* - Don't know.
• Nail abnormality-Don't know.
• Nail anemia-Don't know.
• Ankle oedema-pedal oedema in both feet.
Physical Examination

No Physical Exam Images available! 
Additional Documents 

Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
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
    # Load base model directly onto the target device if possible
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto", # Automatically select appropriate dtype for MPS (bfloat16/float16)
        device_map={"": device} # Load directly to the target device
    )
    logging.info(f"Base model loaded onto device: {base_model.device}")

    logging.info(f"Loading PEFT adapter from: {adapter_path}")
    if not os.path.isdir(adapter_path):
         raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    # Load the LoRA adapter and merge it with the base model
    try:
        # is_trainable=False is important for inference to ensure weights are merged correctly
        # and unnecessary components aren't loaded.
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        # Although device_map loaded the base model, ensure the final PEFT model object is on the device
        model = model.to(device)
        model.eval() # Set the model to evaluation mode (disables dropout, etc.)
        logging.info("PEFT adapter loaded and applied successfully.")
    except Exception as e:
         logging.error(f"Error loading PEFT adapter: {e}", exc_info=True)
         raise RuntimeError(f"Could not load PEFT adapter from {adapter_path}") from e

    logging.info(f"Final model is on device: {model.device}")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, device, max_new_tokens=2000):
    """Generates a response from the model given a prompt."""
    # Use the chat template required by Qwen Instruct models
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
        # Fallback if template fails (might produce worse results)
        logging.warning("Falling back to basic tokenization due to chat template error.")
        tokenized_chat = tokenizer(prompt, return_tensors="pt").input_ids.to(device)


    prompt_len = tokenized_chat.shape[1]
    logging.info(f"Input tokens length: {prompt_len}")

    logging.info(f"Generating response (max_new_tokens={max_new_tokens})...")
    # Disable gradient calculations for inference efficiency
    with torch.no_grad():
        outputs = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # Enable sampling for more varied output
            temperature=0.1,         # Control randomness (lower = more focused)
            top_p=0.9,               # Nucleus sampling (consider top 90% probability mass)
            pad_token_id=tokenizer.eos_token_id # Important for open-ended generation
        )

    # Decode only the newly generated tokens, skipping the prompt
    generated_tokens = outputs[:, prompt_len:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    logging.info("Generation complete.")
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and test a fine-tuned Qwen model with PEFT LoRA adapter.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen2-2.5B-GRPO-test", # Default to the final save directory from training_args.output_dir
        help="Path to the directory containing the PEFT adapter (e.g., 'Qwen2-2.5B-GRPO-test' or 'Qwen2-2.5B-GRPO-test/checkpoint-10')."
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct", # Match the base model used for training
        help="Base model ID used during fine-tuning."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=default_clinical_prompt, # Use the multi-line string as default
        help="Prompt to test the model generation."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1000,
        help="Maximum number of new tokens to generate for the response."
    )

    args = parser.parse_args()

    try:
        device = setup_device()
        model, tokenizer = load_model_and_tokenizer(args.base_model_id, args.model_path, device)
        response = generate_response(model, tokenizer, args.prompt, device, args.max_new_tokens)

        print("\n" + "="*20 + " Inference Results " + "="*20)
        print(f"\n[PROMPT]:\n{args.prompt}")
        print(f"\n[GENERATED RESPONSE]:\n{response}")
        print("\n" + "="*59)

    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please ensure the path '{args.model_path}' exists and contains the adapter files (adapter_model.safetensors, adapter_config.json).")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) 