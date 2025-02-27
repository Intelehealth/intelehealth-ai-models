import pandas as pd
import json
import re
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import os
load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def calculate_word_count(text: str) -> int:
    """Calculate the number of words in a text."""
    return len(re.findall(r'\b\w+\b', text))


def evaluate_factual_accuracy(case: str, rationale: str) -> Tuple[float, str]:
    """Evaluate factual accuracy of the rationale compared to the case."""
    # This is a simplified implementation - in a real scenario, you would need
    # more sophisticated NLP to extract and compare facts
    score = 1.0  # Assume accurate by default
    explanation = ""
    
    # Example check: if rationale mentions symptoms not in the case
    # This is just a placeholder for demonstration
    if "not mentioned symptom" in rationale.lower() and "not mentioned symptom" not in case.lower():
        score = 0.5
        explanation = "Rationale mentions symptoms not present in the case."
    
    return score, explanation

def evaluate_coverage(case: str, rationale: str) -> Tuple[float, str]:
    """Evaluate if the rationale covers all key entities from the case."""
    # Simplified implementation
    score = 1.0
    explanation = ""
    
    # Example: Check if chief complaints are addressed
    if "chief_complaint" in case.lower() and not any(complaint.lower() in rationale.lower() 
                                                   for complaint in re.findall(r'Chief_complaint:.*?►\s*\*\*([^*]+)\*\*', case)):
        score = 0.5
        explanation = "Rationale does not address all chief complaints."
    
    return score, explanation

def evaluate_precision(case: str, rationale: str) -> Tuple[float, str]:
    """Check if rationale introduces new information not in the case."""
    # Simplified implementation
    score = 1.0
    explanation = ""
    
    # This would require sophisticated NLP to properly implement
    # For demonstration, we'll use a simple heuristic

    if len(rationale) > len(case) * 1.5:
        score = 0.8
        explanation = "Rationale may contain information not present in the case."
    
    return score, explanation

def evaluate_reasoning(rationale: str) -> Tuple[float, str]:
    """Evaluate if rationale logically connects evidence to conclusions."""
    score = 1.0
    explanation = ""
    
    reasoning_terms = ["because", "therefore", "suggests", "indicates", "due to", "as a result"]
    if not any(term in rationale.lower() for term in reasoning_terms):
        score = 0.5
        explanation = "Rationale lacks explicit reasoning terms connecting evidence to conclusions."
    
    return score, explanation

def evaluate_alternatives(rationale: str) -> Tuple[float, str]:
    """Assess if rationale considers alternative diagnoses."""
    score = 1.0
    explanation = ""
    
    # Check if multiple diagnoses are mentioned and compared
    if "differential" in rationale.lower() and "likelihood" in rationale.lower():
        # Good sign that alternatives are being considered
        pass
    else:
        score = 0.7
        explanation = "Rationale may not adequately consider alternative diagnoses."
    
    return score, explanation

def evaluate_conciseness(rationale: str) -> Tuple[float, str]:
    """Determine if rationale is appropriately detailed."""
    word_count = calculate_word_count(rationale)
    score = 1.0
    explanation = ""
    
    if word_count < 50:
        score = 0.5
        explanation = f"Rationale is too brief ({word_count} words)."
    elif word_count > 150:
        score = 0.8
        explanation = f"Rationale is verbose ({word_count} words)."
    
    return score, explanation

def evaluate_consistency(rationale: str) -> Tuple[float, str]:
    """Check for internal contradictions in the rationale."""
    # This is a complex task requiring sophisticated NLP
    # For demonstration, we'll use a simple heuristic
    score = 1.0
    explanation = ""
    
    contradictory_phrases = [
        ("present", "absent"),
        ("has", "does not have"),
        ("positive", "negative")
    ]
    
    for phrase1, phrase2 in contradictory_phrases:
        if phrase1 in rationale.lower() and phrase2 in rationale.lower():
            # This is a very simplified check and would produce false positives
            score = 0.8
            explanation = "Potential contradictions detected in rationale."
            break
    
    return score, explanation

def calculate_confidence_score(case: str, rationale: str) -> Dict[str, Any]:
    """Calculate the overall confidence score based on the seven components."""
    factual_accuracy_score, factual_accuracy_explanation = evaluate_factual_accuracy(case, rationale)
    coverage_score, coverage_explanation = evaluate_coverage(case, rationale)
    precision_score, precision_explanation = evaluate_precision(case, rationale)
    reasoning_score, reasoning_explanation = evaluate_reasoning(rationale)
    alternatives_score, alternatives_explanation = evaluate_alternatives(rationale)
    conciseness_score, conciseness_explanation = evaluate_conciseness(rationale)
    consistency_score, consistency_explanation = evaluate_consistency(rationale)
    
    # Calculate overall score
    overall_score = (factual_accuracy_score + coverage_score + precision_score + 
                     reasoning_score + alternatives_score + conciseness_score + 
                     consistency_score) / 7
    
    # Create result dictionary
    result = {
        "Factual Accuracy": {"score": factual_accuracy_score},
        "Coverage": {"score": coverage_score},
        "Precision": {"score": precision_score},
        "Reasoning": {"score": reasoning_score},
        "Alternatives": {"score": alternatives_score},
        "Conciseness": {"score": conciseness_score},
        "Consistency": {"score": consistency_score},
        "Overall Confidence Score": round(overall_score, 2)
    }
    
    # Add explanations for scores less than 1
    if factual_accuracy_score < 1:
        result["Factual Accuracy"]["explanation"] = factual_accuracy_explanation
    if coverage_score < 1:
        result["Coverage"]["explanation"] = coverage_explanation
    if precision_score < 1:
        result["Precision"]["explanation"] = precision_explanation
    if reasoning_score < 1:
        result["Reasoning"]["explanation"] = reasoning_explanation
    if alternatives_score < 1:
        result["Alternatives"]["explanation"] = alternatives_explanation
    if conciseness_score < 1:
        result["Conciseness"]["explanation"] = conciseness_explanation
    if consistency_score < 1:
        result["Consistency"]["explanation"] = consistency_explanation
    
    return result

def evaluate_diagnosis_match_with_llm(actual_diagnoses, predicted_diagnoses, k=5):
    """
    Use LLM to evaluate if the predicted diagnoses match the actual diagnoses.
    
    Args:
        actual_diagnoses: List or string of actual diagnoses
        predicted_diagnoses: List or string of predicted diagnoses
        k: Number of top predictions to consider
        
    Returns:
        Dictionary with recall@1 and recall@5 scores and explanations
        recall@1 will be 0 if the top-ranked LLM diagnosis doesn't match any of the ground truth diagnoses
        recall@5 will be 0 if none of the top 5 LLM diagnoses match any of the ground truth diagnoses
    """
    import os
    from openai import OpenAI
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Ensure diagnoses are in string format
    if isinstance(actual_diagnoses, list):
        actual_diagnoses = ", ".join(actual_diagnoses)
    if isinstance(predicted_diagnoses, list):
        predicted_diagnoses = ", ".join(predicted_diagnoses)
    
    # Construct the prompt
    prompt = f"""
    Evaluate if the predicted diagnoses match the actual diagnoses.
    
    ACTUAL DIAGNOSES:
    {actual_diagnoses}
    
    PREDICTED DIAGNOSES (in order of confidence):
    {predicted_diagnoses}
    
    Please determine:
    1. Recall@1: Does the top predicted diagnosis match any of the actual diagnoses? Consider semantic equivalence, not just exact string matching. Score should be 1.0 if there's a match, and 0.0 if there's no match.
    2. Recall@5: Do any of the top 5 predicted diagnoses match any of the actual diagnoses? Consider semantic equivalence, not just exact string matching. Score should be 1.0 if there's at least one match, and 0.0 if there's no match.
    
    Format your response as a JSON object with the following structure:
    {{
        "recall_at_1": {{
            "score": float (1.0 if match, 0.0 if no match),
            "explanation": "string explaining the reasoning"
        }},
        "recall_at_5": {{
            "score": float (1.0 if match, 0.0 if no match),
            "explanation": "string explaining the reasoning"
        }}
    }}
    
    Ensure your response is a valid JSON object and nothing else.
    """
    
    try:
        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical evaluation assistant that responds only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # Parse the response as JSON
        result = json.loads(response.choices[0].message.content)
        
        return result
    
    except Exception as e:
        print(f"Error calling GPT-4o-mini API for diagnosis evaluation: {e}")
        # Fallback to a basic response in case of API failure
        return {
            "recall_at_1": {
                "score": 0.0,
                "explanation": "API error occurred"
            },
            "recall_at_5": {
                "score": 0.0,
                "explanation": "API error occurred"
            }
        }

def process_csv(file_path: str) -> List[Dict[str, Any]]:
    """Process the CSV file and calculate confidence scores for each row using LLM evaluation."""
    df = pd.read_csv(file_path)
    results = []
    
    # Track recall metrics
    recall_at_1_count = 0
    recall_at_5_count = 0
    total_evaluated = 0
    
    # Create output CSV file with headers
    output_csv_path = "llm_confidence_scores.csv"
    with open(output_csv_path, 'w') as f:
        f.write("visit_id,overall_confidence_score,recall_at_1,recall_at_5,recall_at_1_explanation,recall_at_5_explanation,factual_accuracy_score,coverage_score,precision_score,reasoning_score,alternatives_score,conciseness_score,consistency_score,factual_accuracy_explanation,coverage_explanation,precision_explanation,reasoning_explanation,alternatives_explanation,conciseness_explanation,consistency_explanation,matched_facts\n")
    
    # Create output JSON file
    output_json_path = "llm_confidence_scores.json"
    with open(output_json_path, 'w') as f:
        f.write("[\n")  # Start JSON array
    
    for idx, row in df.iterrows():
        case = row['Clinical Notes']
        rationale = row['Rationale']
        
        # Skip rows with missing data
        if pd.isna(case) or pd.isna(rationale):
            continue
        
        print(f"Processing row {idx+1}/{len(df)}...")
        
        # Use LLM to evaluate the rationale
        confidence_score = evaluate_with_llm(case, rationale)
        
        # Initialize diagnosis evaluation results
        diagnosis_evaluation = {
            "recall_at_1": {"score": 0.0, "explanation": "No diagnosis data available"},
            "recall_at_5": {"score": 0.0, "explanation": "No diagnosis data available"}
        }
        
        # Calculate recall metrics if diagnosis columns exist
        if 'Diagnosis' in row and 'LLM Diagnosis' in row and not pd.isna(row['Diagnosis']) and not pd.isna(row['LLM Diagnosis']):
            actual_diagnoses = str(row['Diagnosis'])
            llm_diagnoses = str(row['LLM Diagnosis'])
            
            # Use LLM to evaluate diagnosis matches
            diagnosis_evaluation = evaluate_diagnosis_match_with_llm(actual_diagnoses, llm_diagnoses)
            
            # Update counts for overall metrics
            recall_at_1_count += diagnosis_evaluation["recall_at_1"]["score"]
            recall_at_5_count += diagnosis_evaluation["recall_at_5"]["score"]
            total_evaluated += 1

        result = {
            "visit_id": row['visit_id'],
            "confidence_score": confidence_score,
            "diagnosis_evaluation": diagnosis_evaluation
        }

        print(result)
        results.append(result)
        
        # Append to CSV immediately
        flat_result = {"visit_id": row['visit_id']}
        
        # Add overall score
        flat_result["overall_confidence_score"] = confidence_score.get("Overall Confidence Score", 0)
        
        # Add recall metrics
        flat_result["recall_at_1"] = diagnosis_evaluation["recall_at_1"]["score"]
        flat_result["recall_at_5"] = diagnosis_evaluation["recall_at_5"]["score"]
        flat_result["recall_at_1_explanation"] = diagnosis_evaluation["recall_at_1"]["explanation"].replace('"', '""')
        flat_result["recall_at_5_explanation"] = diagnosis_evaluation["recall_at_5"]["explanation"].replace('"', '""')
        
        # Add individual scores and explanations
        for category in ["Factual Accuracy", "Coverage", "Precision", "Reasoning", 
                         "Alternatives", "Conciseness", "Consistency"]:
            if category in confidence_score:
                flat_result[f"{category.lower().replace(' ', '_')}_score"] = confidence_score[category].get("score", 0)
                if "explanation" in confidence_score[category]:
                    flat_result[f"{category.lower().replace(' ', '_')}_explanation"] = confidence_score[category]["explanation"].replace('"', '""')
        
        # Add matched facts as a string
        if "Matched Facts" in confidence_score:
            flat_result["matched_facts"] = "; ".join(confidence_score["Matched Facts"]).replace('"', '""')
        
        # Append to CSV
        with open(output_csv_path, 'a') as f:
            csv_line = []
            for field in ["visit_id", "overall_confidence_score", "recall_at_1", "recall_at_5", 
                         "recall_at_1_explanation", "recall_at_5_explanation", 
                         "factual_accuracy_score", "coverage_score", "precision_score", 
                         "reasoning_score", "alternatives_score", "conciseness_score", 
                         "consistency_score", "factual_accuracy_explanation", 
                         "coverage_explanation", "precision_explanation", 
                         "reasoning_explanation", "alternatives_explanation", 
                         "conciseness_explanation", "consistency_explanation", "matched_facts"]:
                if field in flat_result:
                    value = flat_result[field]
                    if isinstance(value, str):
                        csv_line.append(f'"{value}"')
                    else:
                        csv_line.append(str(value))
                else:
                    csv_line.append("")
            f.write(",".join(csv_line) + "\n")
        
        # Append to JSON
        with open(output_json_path, 'a') as f:
            if idx > 0:  # Add comma for all but the first entry
                f.write(",\n")
            f.write(json.dumps(result, indent=2))
    
    # Close the JSON array
    with open(output_json_path, 'a') as f:
        f.write("\n]")
    
    # Calculate overall recall metrics
    overall_recall_at_1 = recall_at_1_count / total_evaluated if total_evaluated > 0 else 0.0
    overall_recall_at_5 = recall_at_5_count / total_evaluated if total_evaluated > 0 else 0.0
    
    print(f"\nOverall Recall@1: {overall_recall_at_1:.4f}")
    print(f"Overall Recall@5: {overall_recall_at_5:.4f}")
    print(f"Total evaluated diagnoses: {total_evaluated}")
    
    return results

def evaluate_with_llm(case: str, rationale: str) -> Dict[str, Any]:
    """Use GPT-4o-mini to evaluate the rationale against the case."""
    import os
    import json
    from openai import OpenAI
    
    # Set up the OpenAI API with your API key
    # You should store this in an environment variable for security
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Construct the prompt
    prompt = f"""
    Evaluate the following medical rationale against the provided clinical case:
    
    CLINICAL CASE:
    {case}
    
    RATIONALE:
    {rationale}
    
    Please evaluate on the following criteria and provide a score from 0.0 to 1.0 for each:
    1. Factual Accuracy: Does the rationale contain only facts present in the case?
    2. Coverage: Does the rationale address all key information from the case?
    3. Precision: Does the rationale avoid introducing new information not in the case?
    4. Reasoning: Does the rationale logically connect evidence to conclusions?
    5. Alternatives: Does the rationale consider alternative diagnoses?
    6. Conciseness: Is the rationale appropriately detailed without being too verbose?
    7. Consistency: Is the rationale free from internal contradictions?
    
    Also, extract and list all the key facts from the case that are correctly mentioned in the rationale.
    
    Format your response as a JSON object with the following structure:
    {{
        "Factual Accuracy": {{"score": float, "explanation": "string"}},
        "Coverage": {{"score": float, "explanation": "string"}},
        "Precision": {{"score": float, "explanation": "string"}},
        "Reasoning": {{"score": float, "explanation": "string"}},
        "Alternatives": {{"score": float, "explanation": "string"}},
        "Conciseness": {{"score": float, "explanation": "string"}},
        "Consistency": {{"score": float, "explanation": "string"}},
        "Overall Confidence Score": float,
        "Matched Facts": ["fact1", "fact2", ...]
    }}
    
    Ensure your response is a valid JSON object and nothing else.
    """
    
    try:
        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical evaluation assistant that responds only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # Parse the response as JSON
        result = json.loads(response.choices[0].message.content)

        print(result)
        
        # Calculate overall score if not provided by the model
        if "Overall Confidence Score" not in result:
            scores = [result[key]["score"] for key in result if key != "Matched Facts" and isinstance(result[key], dict) and "score" in result[key]]
            if scores:
                result["Overall Confidence Score"] = round(sum(scores) / len(scores), 2)
            else:
                result["Overall Confidence Score"] = 0.0
        
        # Ensure Matched Facts exists
        if "Matched Facts" not in result:
            result["Matched Facts"] = []
        
        return result
    
    except Exception as e:
        print(f"Error calling GPT-4o-mini API: {e}")
        # Fallback to a basic response in case of API failure
        return {
            "Factual Accuracy": {"score": 0.5, "explanation": "API error occurred"},
            "Coverage": {"score": 0.5, "explanation": "API error occurred"},
            "Precision": {"score": 0.5, "explanation": "API error occurred"},
            "Reasoning": {"score": 0.5, "explanation": "API error occurred"},
            "Alternatives": {"score": 0.5, "explanation": "API error occurred"},
            "Conciseness": {"score": 0.5, "explanation": "API error occurred"},
            "Consistency": {"score": 0.5, "explanation": "API error occurred"},
            "Overall Confidence Score": 0.5,
            "Matched Facts": ["API error occurred"]
        }

def main():
    file_path = "./gemini_2_flash_nas_combined_ayu_inference_final.csv"
    results = process_csv(file_path)
    
    # Print results in the requested format
    for result in results:
        print(f"Visit ID: {result['visit_id']}")
        print(json.dumps(result['confidence_score'], indent=2))
        print("Diagnosis Evaluation:")
        print(json.dumps(result['diagnosis_evaluation'], indent=2))
        print("\n" + "-"*50 + "\n")
    
    print(f"Results have been saved to llm_confidence_scores.csv and llm_confidence_scores.json")

if __name__ == "__main__":
    main()