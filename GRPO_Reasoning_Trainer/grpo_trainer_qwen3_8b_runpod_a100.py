import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model # Added PEFT imports

import re # Keep re if needed for potential future answer parsing
import json # Added json import
import os # Added os import
import logging # Add logging import
# from google import genai
# from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
  api_key = OPENAI_API_KEY,
  organization=OPENAI_ORG_ID,
  project=OPENAI_PROJECT_ID,
)

# load_dotenv(
#     "ops/.env"
# )

class DdxResponse(BaseModel):
    score: int
    rationale: str

# --- Configuration ---
model_id = "Qwen/Qwen3-8B"
dataset_path = "train_grpo_data.jsonl"
# output_dir = "grpo_qwen_trained_model"
# num_train_epochs = 1 # Removed, max_steps will be used
# Adjust batch size and gradient accumulation based on your M4's memory for Qwen
per_device_train_batch_size = 2 # For 8B model on A100, using smaller batch size
gradient_accumulation_steps = 16 # Increased for 8B model to maintain effective batch size
# learning_rate = 1.41e-5 # Updated below in GRPOConfig
beta = 0.05 # GRPO specific hyperparameter, decreased to allow more deviation
seed = 42

# --- Check for MPS device ---
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        device = "cpu"
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
        device = "cpu"
else:
    print("MPS device found.")
    device = "mps"

# --- Load Model and Tokenizer ---
print(f"Loading model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto", # device_map='auto' handles placing model parts on MPS/CPU
)

# --- PEFT LoRA Configuration ---
print("Applying PEFT LoRA configuration...")
lora_config = LoraConfig(
    r=16, # Updated rank
    lora_alpha=32, # Often set higher than r
    target_modules=[ # Common targets for Qwen models
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM" # Specify task type
)

model = get_peft_model(model, lora_config)
print("PEFT model created.")
model.print_trainable_parameters()

# --- LLM Judge ---
# Function to interact with OpenAI API
def openai_llm_judge(gold, pred, trace=None):
    print("############## evaluating open ai llm judge ###############")
    print(gold)
    pred_diagnosis = pred
    print(pred_diagnosis)


    print("\n")
    response = client.beta.chat.completions.parse(
        
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."},   
            {"role": "user", "content": f"Expected output: " + gold},
            {"role": "user", "content": f"Predicted output: " + str(pred_diagnosis) },
            {"role": "user", "content": '''Evaluate the semantic similarity between the predicted and expected outputs. Consider the following: 
             1. Is the expected diagnosis present in the top 5 diagnosises predicted?
             2. Is the core meaning preserved even if the wording differs from medical terminologies and synonyms for the matching expected and predicted diagnosis?
             3. Are there any significant omissions or additions in the predicted output?
             
             Provide output as valid JSON with field `matching rank` within 1-5 for any match and 0 for no match and field `rationale` having the reasoning string for this score.'''}
        ],
        response_format = DdxResponse
    )
    # print(response)
    # Extract the content from the first choice
    # content = response.choices[0].message.content
    content = response.choices[0].message.parsed

    print("Response from llm:")
    print("LLM Judge score: ", content.score)
    score = content.score
    rationale = content.rationale
    print("Rationale: ", content.rationale)
    

    # time.sleep(2)

    return score

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen3-8B-GRPO",
    learning_rate=3e-5, # Increased learning rate based on graph feedback
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16, # A100: Increased further to compensate for batch_size=1
    per_device_train_batch_size=2, # A100: Decreased to 1, the minimum
    # num_train_epochs=1, # Removed, using max_steps
    max_steps=25, # User specified max steps
    bf16=True, # Keeping bf16 for performance, should work on M4
    # Parameters that control de data preprocessing
    # Setting max_length based on user's max sequence length 3000
    # Assuming max_prompt + max_completion <= 3000
    # Allocate proportionally or based on expected lengths.
    # Let's try allocating more to completion.
    # Reduced sequence lengths significantly to combat OOM
    max_prompt_length=768, # Increased moderately
    max_completion_length=1536, # Increased moderately
    num_generations=2,  # A100: Reduced from 4 to reduce OOM
    # max_prompt_length=128,  # default: 512 - Replaced by above
    # Parameters related to reporting and saving
    report_to=["wandb"], # Explicitly enable TensorBoard (and wandb if desired/setup)
    logging_strategy="steps", # Explicitly set
    logging_steps=1, # Logging every 1 steps as per eval
    push_to_hub=True,
    save_strategy="steps",
    save_steps=2, # Save checkpoint at each eval step
    eval_strategy="steps", # User specified
    eval_steps=1, # User specified
    optim="adamw_torch", # Or just remove this line to use the HF default adam_w8bit bitnbuytes
    # save_steps=10, # Effective batch size is 8 (1*8). Steps per epoch varies.
    beta=0.05 # KL coefficient, decreased based on graph feedback
)



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
        if parsed_answer is not None:
            try:
                response = openai_llm_judge(ground_truth_diagnosis, parsed_answer)
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

            except Exception as e:
                print(f"Error during LLM call or processing: {e}. Defaulting to exact match logic.")
                # Fallback to exact match logic in case of any error during the LLM call
                rank = 1 if parsed_answer.lower() == ground_truth_diagnosis.lower() else 6

            # Calculate position reward based on the obtained rank
            position_reward = max(0, 1.2 - 0.2 * rank)
            reward += position_reward
        # length_penalty = 0.0
        # if parsed_answer is not None and answer_len > avg_length:
        #     length_penalty = (answer_len - avg_length) * 0.01
        #     reward -= length_penalty
    except Exception as e:
        print(f"Reward calc error: {e} for text: '{response_text[:50]}...'")
        return 0.0
    print("############## reward calculated ###############", reward)
    return max(0, reward)



dataset = load_dataset("json", data_files=dataset_path, split="train")
# Split dataset into train/validation/test sets (80/10/10)
dataset_dict = dataset.train_test_split(test_size=0.2, seed=seed)
temp_test = dataset_dict['test'].train_test_split(test_size=0.5, seed=seed)

train_dataset = dataset_dict['train']
validation_dataset = temp_test['train'] 
test_dataset = temp_test['test']

# Print dataset size
print(f"Training dataset size: {len(train_dataset)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Expected steps per epoch: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

# Print before starting training
print("\n---> Starting Training <---\n")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[combined_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset # Added validation dataset for evaluation
)

trainer.train()
trainer.save_model(training_args.output_dir) 