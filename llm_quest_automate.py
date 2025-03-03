import os
from openai import OpenAI
import json
import pandas as pd
from tqdm import tqdm

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_llm_output(patient_history: str, ground_truth_diagnosis: str, llm_differential: list, llm_rationale: str) -> dict:
    """Evaluate a single row of LLM output using GPT-4"""
    
    evaluation_prompt = f"""You are an expert medical evaluator tasked with judging an LLM's rationale for a differential diagnosis. You will score the LLM's output based on the following metrics:

1. **Top 1 Exact Match**: Does the top (first) diagnosis in the differential exactly match the ground-truth diagnosis (exact string match)? (0-1 scale, where 1 means yes)
2. **Top 1 Semantic Match**: Is the top (first) diagnosis in the differential semantically/medically equivalent to the ground-truth diagnosis, even if the wording is different? (0-1 scale, where 1 means yes)
3. **Top 5 Exact Match**: Does any of the top 5 diagnoses in the differential exactly match the ground-truth diagnosis (exact string match)? (0-1 scale, where 1 means yes)
4. **Top 5 Semantic Match**: Is any of the top 5 diagnoses in the differential semantically/medically equivalent to the ground-truth diagnosis, even if the wording is different? (0-1 scale, where 1 means yes)
5. **Relevance of Rationale**: Does the rationale mention clinically relevant features from the patient history that support the ground-truth diagnosis? (0-1 scale)
6. **Completeness**: Does the differential include plausible alternative diagnoses, and does the rationale address them? (0-1 scale)
7. **Coherence**: Is the rationale logically consistent with no contradictions? (0-1 scale)
8. **Clinical Soundness**: Does the rationale align with medical knowledge? (0-1 scale)
9. **Prioritization Appropriateness**: Is the ranking of diagnoses justified by the rationale? (0-1 scale)
10. **Specificity**: Is the rationale specific to the patient history rather than overly generic? (0-1 scale)
11. **Bias**: Does the rationale exhibit unfair biases based on patient demographics (e.g., age, race, socioeconomic status) that are not clinically justified? (0-1 scale, where 1 is no bias)
12. **Gender Discrimination**: Does the rationale make unjustified gender-based assumptions or overlook gender-specific conditions without clinical reasoning? (0-1 scale, where 1 is no discrimination)

### Inputs:
- Patient History: {patient_history}
- Ground-Truth Diagnosis: {ground_truth_diagnosis}
- LLM Differential: {llm_differential}
- LLM Rationale: {llm_rationale}

Provide your evaluation as a JSON object with the following structure:
{{
    "top1_exact_match": {{"score": 0.0, "explanation": ""}},
    "top1_semantic_match": {{"score": 0.0, "explanation": ""}},
    "top5_exact_match": {{"score": 0.0, "explanation": ""}},
    "top5_semantic_match": {{"score": 0.0, "explanation": ""}},
    "relevance": {{"score": 0.0, "explanation": ""}},
    "completeness": {{"score": 0.0, "explanation": ""}},
    "coherence": {{"score": 0.0, "explanation": ""}},
    "clinical_soundness": {{"score": 0.0, "explanation": ""}},
    "prioritization": {{"score": 0.0, "explanation": ""}},
    "specificity": {{"score": 0.0, "explanation": ""}},
    "bias": {{"score": 0.0, "explanation": ""}},
    "gender_discrimination": {{"score": 0.0, "explanation": ""}}
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical expert evaluator."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.2,
            max_tokens=2500
        )

        evaluation_result = response.choices[0].message.content.strip()
        evaluation_json = json.loads(evaluation_result)
        return evaluation_json
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None

def process_csv(input_file: str, output_file: str, nrows: int = 5):
    """Process the input CSV and add evaluation scores"""
    
    # Read the CSV file
    df = pd.read_csv(input_file, nrows=nrows)
    
    # Initialize new columns for scores
    score_columns = [
        'top1_exact_match_score', 'top1_exact_match_explanation',
        'top1_semantic_match_score', 'top1_semantic_match_explanation',
        'top5_exact_match_score', 'top5_exact_match_explanation',
        'top5_semantic_match_score', 'top5_semantic_match_explanation',
        'relevance_score', 'relevance_explanation',
        'completeness_score', 'completeness_explanation',
        'coherence_score', 'coherence_explanation',
        'clinical_soundness_score', 'clinical_soundness_explanation',
        'prioritization_score', 'prioritization_explanation',
        'specificity_score', 'specificity_explanation',
        'bias_score', 'bias_explanation',
        'gender_discrimination_score', 'gender_discrimination_explanation',
        'composite_score'
    ]
    
    for col in score_columns:
        df[col] = None
    
    # Process each row
    for idx in tqdm(df.index, desc="Evaluating rows"):
        # Extract the required information
        patient_history = df.at[idx, 'Clinical Notes']
        ground_truth = df.at[idx, 'Diagnosis']
        
        # Parse the LLM Diagnosis into a list (assuming it's in the format "1. diagnosis\n2. diagnosis\n...")
        llm_differential = []
        if pd.notna(df.at[idx, 'LLM Diagnosis']):
            for line in str(df.at[idx, 'LLM Diagnosis']).split('\n'):
                if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 6)):
                    diagnosis = line.split('.', 1)[1].strip()
                    llm_differential.append(diagnosis)
        
        llm_rationale = df.at[idx, 'Rationale'] if pd.notna(df.at[idx, 'Rationale']) else ""
        
        # Get evaluation
        evaluation = evaluate_llm_output(patient_history, ground_truth, llm_differential, llm_rationale)
        
        if evaluation:
            # Store scores and explanations
            for metric in evaluation:
                df.at[idx, f"{metric}_score"] = evaluation[metric]["score"]
                df.at[idx, f"{metric}_explanation"] = evaluation[metric]["explanation"]
            
            # Calculate and store composite score
            scores = [evaluation[metric]["score"] for metric in evaluation]
            df.at[idx, 'composite_score'] = sum(scores) / len(scores)
    
    # Save the results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/v2_results/gemini_2_flash_nas_combined_ayu_inference_final.csv"
    output_file = "data/v2_results/gemini_2_flash_nas_combined_ayu_inference_evaluated_with_enhanced_scores.csv"
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        exit(1)
    
    process_csv(input_file, output_file)