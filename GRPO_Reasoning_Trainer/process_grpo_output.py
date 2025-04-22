import pandas as pd
import re
import os

def extract_reasoning_answer(response):
    """
    Extracts reasoning and answer from the response string based on tags.
    """
    reasoning = None
    answer = None

    # Use non-greedy matching to find content within tags
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)

    # Fallback for structure without closing tags or slightly different patterns
    if not reasoning_match:
         reasoning_match = re.search(r'<reasoning>(.*?)(?:<answer>|$)', response, re.DOTALL | re.IGNORECASE)
    if not answer_match:
         answer_match = re.search(r'<answer>(.*)', response, re.DOTALL | re.IGNORECASE)


    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        # If answer wasn't found separately, check if it's nested or follows reasoning
        if not answer_match:
            remaining_text = response[reasoning_match.end():]
            answer_match_after_reasoning = re.search(r'<answer>(.*)', remaining_text, re.DOTALL | re.IGNORECASE)
            if answer_match_after_reasoning:
                 answer = answer_match_after_reasoning.group(1).strip()
                 # Remove potential closing tag if present
                 answer = re.sub(r'</answer>\s*$', '', answer, flags=re.IGNORECASE).strip()

    if answer_match:
         answer = answer_match.group(1).strip()
         # Remove potential closing tag if present
         answer = re.sub(r'</answer>\s*$', '', answer, flags=re.IGNORECASE).strip()
         # If reasoning wasn't found separately, assume it's before the answer tag
         if not reasoning_match:
             reasoning_text_before_answer = response[:answer_match.start()]
             reasoning_match_before_answer = re.search(r'<reasoning>(.*)', reasoning_text_before_answer, re.DOTALL | re.IGNORECASE)
             if reasoning_match_before_answer:
                 reasoning = reasoning_match_before_answer.group(1).strip()
                 # Remove potential closing tag if present
                 reasoning = re.sub(r'</reasoning>\s*$', '', reasoning, flags=re.IGNORECASE).strip()


    # Handle cases where tags might be missing entirely or format is unexpected
    if reasoning is None and answer is None:
        # Simple split strategy as a last resort (e.g., split by "Answer:")
        parts = re.split(r'Answer:', response, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            # If no clear separator, assign whole content to reasoning or handle as needed
            reasoning = response # Or set both to None/empty string based on desired handling
            answer = None


    # Clean up potential leftover tags if extraction wasn't perfect
    if reasoning:
        reasoning = re.sub(r'</?reasoning>','', reasoning, flags=re.IGNORECASE).strip()
        reasoning = re.sub(r'<answer>.*','', reasoning, flags=re.DOTALL | re.IGNORECASE).strip() # Avoid including answer tag in reasoning
    if answer:
       answer = re.sub(r'</?answer>','', answer, flags=re.IGNORECASE).strip()


    return pd.Series([reasoning, answer])

# Define file paths
input_csv_path = '/Users/bsb/work/intelehealth-ai-models/GRPO_Reasoning_Trainer/grpo_nas_v3_unseen_data_inference_output_complete.csv'
output_csv_path = '/Users/bsb/work/intelehealth-ai-models/GRPO_Reasoning_Trainer/grpo_nas_v3_unseen_data_inference_output_processed.csv'

# Check if input file exists
if not os.path.exists(input_csv_path):
    print(f"Error: Input file not found at {input_csv_path}")
    exit()

# Read the CSV file
try:
    df = pd.read_csv(input_csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Ensure the 'Generated Response' column exists
if 'Generated Response' not in df.columns:
    print(f"Error: Column 'Generated Response' not found in the CSV file.")
    exit()

# Apply the function to the 'Generated Response' column
# Handle potential NaN values in the column first
df['Generated Response'] = df['Generated Response'].fillna('')
df[['Reasoning', 'Answer']] = df['Generated Response'].apply(extract_reasoning_answer)

# Save the processed DataFrame to a new CSV file
try:
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully processed the file and saved the results to {output_csv_path}")
except Exception as e:
    print(f"Error writing processed CSV file: {e}") 