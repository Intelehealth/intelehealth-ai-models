import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
import os
import re
import json
import logging
from openai import OpenAI
from pydantic import BaseModel

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys and Client Setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

client = OpenAI(
  api_key=OPENAI_API_KEY,
  organization=OPENAI_ORG_ID,
  project=OPENAI_PROJECT_ID,
)

class DdxResponse(BaseModel):
    score: int
    rationale: str

# --- Configuration ---
dataset_path = "train_nas_v4_v0.1_data.jsonl"  # From original script
validation_dataset_path = "validation_nas_v4_v0.1_data.jsonl" # From original script
seed = 42  # From original script
beta_value = 0.005 # From original script's GRPOConfig

# --- Unsloth Model Configuration ---
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 16 # Larger rank = smarter, but slower

# --- Load Model and Tokenizer with Unsloth ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=False, # Changed to False for 8-bit LoRA / 16-bit training
    fast_inference=True, # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6, # Reduce if out of memory
)

# --- PEFT Model Setup with Unsloth ---
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth", # Enable long context finetuning
    random_state=3407,
)
model.print_trainable_parameters()


# --- LLM Judge (from original script) ---
def openai_llm_judge(gold, pred, trace=None):
    print("############## evaluating open ai llm judge ###############")
    print(gold)
    pred_diagnosis = pred
    print(pred_diagnosis)
    print("\n")
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."},
                {"role": "user", "content": f"Expected output: {gold}"},
                {"role": "user", "content": f"Predicted output: {str(pred_diagnosis)}"},
                {"role": "user", "content": '''Evaluate the semantic similarity between the predicted and expected outputs. Consider the following: 
                 1. Is the expected diagnosis present in the top 5 diagnosises predicted?
                 2. Is the core meaning preserved even if the wording differs from medical terminologies and synonyms for the matching expected and predicted diagnosis?
                 3. Are there any significant omissions or additions in the predicted output?
                 
                 Provide output as valid JSON with field `matching rank` within 1-5 for any match and 0 for no match and field `rationale` having the reasoning string for this score.'''
                }
            ],
            response_format=DdxResponse
        )
        content = response.choices[0].message.parsed
        print("Response from llm:")
        print("LLM Judge score: ", content.score)
        print("Rationale: ", content.rationale)
        return content.score
    except Exception as e:
        print(f"Error in openai_llm_judge: {e}")
        return 0 # Default to 0 score on error



# --- Dataset Loading (adapted from original script) ---
full_train_dataset = load_dataset("json", data_files=dataset_path, split="train")
validation_dataset = load_dataset("json", data_files=validation_dataset_path, split="train")

# Split dataset into train and test sets (90/10) - original script did this
# For GRPOTrainer, we usually pass the full training dataset and an eval dataset.
# The user's new snippet uses `train_dataset = dataset`. We'll call our loaded one `train_dataset`.
dataset_dict = full_train_dataset.train_test_split(test_size=0.1, seed=seed)
train_dataset = dataset_dict['train']
test_dataset = dataset_dict['test'] # Keep for potential use, though GRPOTrainer uses eval_dataset

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# --- GRPO Training Arguments (from user's Unsloth example) ---
max_prompt_length = 256 # As per Unsloth example

training_args = GRPOConfig(
    learning_rate=1e-4,
    # adam_beta1=0.9,
    # adam_beta2=0.99,
    # weight_decay=0.1,
    # warmup_ratio=0.1,
    # lr_scheduler_type="cosine",
    optim="adamw_torch",
    gradient_accumulation_steps=1, # A100: Increased further to compensate for batch_size=1
    per_device_train_batch_size=2, # A100: Decreased to 1, the minimum
    # num_train_epochs=1, # Removed, using max_steps
    max_steps=25, # User specified max steps
    bf16=True, # Keeping bf16 for performance, should work on M4
    # beta=beta_value, # Added beta parameter to control KL divergence from reference model
    # Parameters that control de data preprocessing
    # Setting max_length based on user's max sequence length 3000
    # Assuming max_prompt + max_completion <= 3000
    # Allocate proportionally or based on expected lengths.
    # Let's try allocating more to completion.
    # Reduced sequence lengths significantly to combat OOM
    max_prompt_length=768, # A100: Decreased from 1024
    max_completion_length=1280, # Increased to fit new max_seq_length
    num_generations=2,  # Set to 2 for compatibility
    report_to=["wandb"], # Explicitly enable TensorBoard (and wandb if desired/setup)
    logging_strategy="steps", # Explicitly set
    logging_steps=1, # Logging every 1 steps as per eval
    push_to_hub=True,
    save_strategy="steps",
    save_steps=2, # Save checkpoint at each eval step
    eval_strategy="steps", # User specified
    eval_steps=1, # User specified
    # max_grad_norm=0.1,
    output_dir="Llama-3.1-8B-Instruct-NAS-v4-v0.7-unsloth-GRPO", # Changed output dir
    remove_unused_columns=False, # Important for accessing 'answer' column in reward_funcs
    # Added eval strategy from original script
)





def calculate_reward(response_text, ground_truth_diagnosis):
    print("############## calculating reward ###############")
    print(response_text)
    reward = 0.0
    parsed_reasoning = None
    parsed_answer = None
    answer_len = 0
    try:
        reasoning_present = "<reasoning>" in response_text and "</reasoning>" in response_text
        answer_present = "<answer>" in response_text and "</answer>" in response_text
        if reasoning_present: reward += 0.5
        if answer_present: reward += 0.5
        if reasoning_present:
            match = re.search(r"<reasoning>(.*?)</reasoning>", response_text, re.DOTALL)
            if match: parsed_reasoning = match.group(1).strip()
        if answer_present:
            match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
            if match:
                parsed_answer = match.group(1).strip()
                answer_len = len(parsed_answer)
        position_reward = 0.0
        try:
            response = openai_llm_judge(ground_truth_diagnosis, response_text)
            llm_rank_text = response

            print(f"INFO: Simulated LLM Rank: {llm_rank_text}")

            # Parse the LLM response
            try:
                rank = int(llm_rank_text)
                if not 1 <= rank <= 5:
                    print(f"Warning: LLM returned rank {rank} outside the expected range [1, 6]. Defaulting to 6.")
                    rank = 6
            except ValueError:
                print(f"Warning: Could not parse integer rank from LLM response: '{llm_rank_text}'. Defaulting to exact match logic.")
                # Fallback to exact match if LLM response is not a valid integer
                rank = 1 if parsed_answer.lower() == ground_truth_diagnosis.lower() else 6

            # Calculate position reward based on the obtained rank
            position_reward = max(0, 1.2 - 0.2 * rank)
            reward += position_reward
        except Exception as e:
            print(f"Error during LLM call or processing: {e}. Defaulting to exact match logic.")
            # Fallback to exact match logic in case of any error during the LLM call
            rank = 1 if parsed_answer.lower() == ground_truth_diagnosis.lower() else 6

            # Calculate position reward based on the obtained rank
            position_reward = max(0, 1.2 - 0.2 * rank)
            reward += position_reward
    except Exception as e:
        print(f"Reward calc error: {e} for text: '{response_text[:50]}...'")
        return 0.0
    print("############## reward calculated ###############", reward)
    return max(0, reward)

print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Max steps: {training_args.max_steps}")


def combined_reward_func(
    prompts, completions, answer, **kwargs
):
    print(completions)
    # Extract generated responses
    responses = []
    valid_indices = []
    for idx, completion in enumerate(completions):
        try:
            generated_content = completion[0]['content'].strip()
            responses.append(generated_content)
            valid_indices.append(idx)
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error extracting content from completion {idx}: {e}")
            continue

    # Calculate rewards for valid responses
    rewards = []
    avg_length = kwargs.get('avg_length', 100)  # Default avg_length if not provided
    
    for idx, response in enumerate(responses):
        reward = calculate_reward(
            response_text=response,
            ground_truth_diagnosis=answer[valid_indices[idx]]
            # avg_length=avg_length
        )
        rewards.append(reward)

    return rewards


# --- Initialize GRPOTrainer ---
# Note: The user's example reward_funcs (xmlcount_reward_func, etc.) are not defined.
# Using combined_reward_func from the original script.
trainer = GRPOTrainer(
    model=model,
    # tokenizer=tokenizer, # Unsloth's FastLanguageModel returns tokenizer, GRPO needs it.
                         # The user's new example had `processing_class = tokenizer`
                         # `trl.GRPOTrainer` expects `tokenizer` argument. <- This comment was incorrect.
    reward_funcs=[combined_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset, # Added validation dataset
)

# --- Start Training ---
print("\n---> Starting Unsloth GRPO Training <---\n")
print(f"Using beta={training_args.beta} to control KL divergence from reference model")
trainer.train()

# --- Save Model ---
print("\n---> Saving Model <---\n")
# Unsloth recommends saving LoRA adapters separately if needed,
# or the full model via Hugging Face `save_pretrained`
# GRPOTrainer's save_model should work for PEFT models.
trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")

# For Unsloth, if you want to save with `save_pretrained` for full model merging:
# model.save_pretrained(training_args.output_dir) # This saves the full model
# tokenizer.save_pretrained(training_args.output_dir)
# If you only want to save LoRA adapters:
# model.save_pretrained_merged("merged_lora_model", tokenizer, save_method = "merged_16bit")
# model.save_pretrained_gguf("gguf_quantized_model", tokenizer, quantization_method = "q4_k_m")

print("Training complete.") 