import csv
import pandas as pd
import json
import re
import os
import sys
from google import genai
from openai import OpenAI
import time

from google.api_core.exceptions import GoogleAPIError
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Literal, Optional

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Gemini client
client = ""
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully")

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI API configured successfully")

# Define Pydantic model for structured response
class DiagnosisAnalysis(BaseModel):
    is_accurate: bool = Field(..., description="Whether the diagnosis is accurate based on the clinical notes")
    confidence: Literal["High", "Medium", "Low"] = Field(..., description="Confidence level in the assessment")
    rationale: str = Field(..., description="Detailed rationale for the assessment")

def llm_judge_diagnosis(clinical_notes, diagnosis):
    """
    Function that uses Gemini Flash 2.0 to evaluate if the ground truth diagnosis
    is accurate based on the patient's clinical notes.
    Uses structured prompt for JSON response.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Create a structured prompt for the LLM
            prompt = f"""
You are a medical expert tasked with evaluating the accuracy of a diagnosis based on clinical notes relevant to rural Indian healthcare settings.

CLINICAL NOTES:
{clinical_notes}

DIAGNOSIS:
{diagnosis}

Based on the clinical notes, evaluate whether the diagnosis is accurate. Consider the symptoms, physical examination findings, medical history, and other relevant information.

IMPORTANT: Return ONLY a valid JSON object with exactly the following structure:
{{
  "is_accurate": true/false,
  "confidence": "High"|"Medium"|"Low",
  "rationale": "Your detailed rationale here"
}}

Your response must only contain this JSON object and nothing else.
"""

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': DiagnosisAnalysis,
                },
            )
            
            # Extract text from response
            response_text = response.text
            
            # Try to parse JSON directly
            try:
                response_json = json.loads(response_text)
                # Validate against our model
                analysis = DiagnosisAnalysis(**response_json)
                return analysis.model_dump()
            except (json.JSONDecodeError, ValueError) as e:
                # If direct parsing failed, try to extract JSON from the response
                json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                json_matches = re.findall(json_pattern, response_text)
                
                if json_matches:
                    # Try each JSON match
                    for json_str in json_matches:
                        try:
                            # Clean up common JSON formatting issues
                            json_str = json_str.replace("'", '"')
                            json_str = json_str.replace("True", "true").replace("False", "false")
                            
                            response_json = json.loads(json_str)
                            analysis = DiagnosisAnalysis(**response_json)
                            return analysis.model_dump()
                        except:
                            continue
                
                # If we're here, we couldn't extract valid JSON
                if attempt < max_retries - 1:
                    print(f"Retry {attempt+1}/{max_retries} for Gemini - Invalid JSON response")
                    time.sleep(2)  # Add delay between retries
                    continue
                else:
                    # Final attempt, use manual extraction
                    print("Attempting manual extraction of values")
                    
                    # Check for accuracy indication
                    is_accurate = "true" in response_text.lower() and not "false" in response_text.lower()
                    
                    # Check for confidence level
                    confidence = "Medium"  # Default
                    if "high" in response_text.lower():
                        confidence = "High"
                    elif "low" in response_text.lower():
                        confidence = "Low"
                    
                    # Extract some text for rationale
                    rationale = "Extracted from unstructured response. " + response_text[:200] + "..."
                    
                    return {
                        "is_accurate": is_accurate,
                        "confidence": confidence,
                        "rationale": rationale
                    }
                
        except GoogleAPIError as e:
            print(f"Gemini API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Add delay between retries
            else:
                return {
                    "is_accurate": True,
                    "confidence": "Low",
                    "rationale": f"Unable to analyze due to API error: {str(e)}"
                }
        except Exception as e:
            print(f"Unexpected error in Gemini LLM analysis (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Add delay between retries
            else:
                return {
                    "is_accurate": True,
                    "confidence": "Low",
                    "rationale": f"Unable to analyze due to unexpected error: {str(e)}"
                }

def openai_judge_diagnosis(clinical_notes, diagnosis):
    """
    Function that uses OpenAI to evaluate if the ground truth diagnosis
    is accurate based on the patient's clinical notes.
    Uses the latest OpenAI API.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            if not openai_client:
                return {
                    "is_accurate": True,
                    "confidence": "Low",
                    "rationale": "OpenAI client not configured. Please set OPENAI_API_KEY environment variable."
                }
            
            # Create a structured prompt for the LLM
            system_prompt = """You are a medical expert tasked with evaluating the accuracy of diagnoses based on clinical notes in rural Indian healthcare settings. 
You will provide your assessment in a valid JSON format with the following structure exactly:
{
  "is_accurate": true/false,
  "confidence": "High"|"Medium"|"Low",
  "rationale": "Your detailed rationale here"
}
Your response must ONLY contain this JSON object and nothing else. No markdown formatting, no explanations outside the JSON."""

            user_prompt = f"""
CLINICAL NOTES:
{clinical_notes}

DIAGNOSIS:
{diagnosis}

Based on the clinical notes, evaluate whether the diagnosis is accurate. Consider the symptoms, physical examination findings, medical history, and other relevant information.

Return ONLY a JSON object with the specified structure.
"""

            # Call the OpenAI API
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # Parse the structured response
            response_text = response.choices[0].message.content.strip()
            
            try:
                response_json = json.loads(response_text)
                analysis = DiagnosisAnalysis(**response_json)
                return analysis.model_dump()
            except (json.JSONDecodeError, ValueError) as e:
                # If direct parsing failed, try to extract JSON from the response
                json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                json_matches = re.findall(json_pattern, response_text)
                
                if json_matches:
                    # Try each JSON match
                    for json_str in json_matches:
                        try:
                            # Clean up common JSON formatting issues
                            json_str = json_str.replace("'", '"')
                            json_str = json_str.replace("True", "true").replace("False", "false")
                            
                            response_json = json.loads(json_str)
                            analysis = DiagnosisAnalysis(**response_json)
                            return analysis.model_dump()
                        except:
                            continue
                
                # If we're here, we couldn't extract valid JSON
                if attempt < max_retries - 1:
                    print(f"Retry {attempt+1}/{max_retries} for OpenAI - Invalid JSON response")
                    time.sleep(2)  # Add delay between retries
                    continue
                else:
                    # Final attempt, use manual extraction
                    print("Attempting manual extraction of values from OpenAI response")
                    
                    # Check for accuracy indication
                    is_accurate = "true" in response_text.lower() and not "false" in response_text.lower()
                    
                    # Check for confidence level
                    confidence = "Medium"  # Default
                    if "high" in response_text.lower():
                        confidence = "High"
                    elif "low" in response_text.lower():
                        confidence = "Low"
                    
                    # Extract some text for rationale
                    rationale = "Extracted from unstructured response. " + response_text[:200] + "..."
                    
                    return {
                        "is_accurate": is_accurate,
                        "confidence": confidence,
                        "rationale": rationale
                    }

        except Exception as e:
            print(f"Unexpected error in OpenAI LLM analysis (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Add delay between retries
            else:
                return {
                    "is_accurate": True,
                    "confidence": "Low",
                    "rationale": f"Unable to analyze due to unexpected error: {str(e)}"
                }

def extract_section(text, section_name):
    """Extract a specific section from the clinical notes"""
    pattern = f"{section_name}:(.*?)(?:\n\n|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# Input and output file paths
input_file = "./gemini_2_flash_nas_combined_ayu_inference_merged_latest.csv"
output_file = "./filtered_cases_with_llm_analysis_two_llms.csv"

try:
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Loaded CSV with shape: {df.shape}")
    
    # Convert Top 1 ddx hit and Top 5 ddx hit to numeric if they're strings
    df['Top 1 ddx hit'] = pd.to_numeric(df['Top 1 ddx hit'], errors='coerce')
    df['Top 5 ddx hit'] = pd.to_numeric(df['Top 5 ddx hit'], errors='coerce')
    
    # Filter rows where either Top 1 ddx hit or Top 5 ddx hit is 0
    filtered_df = df[(df['Top 1 ddx hit'] == 0) & (df['Top 5 ddx hit'] == 0)]
    
    if filtered_df.empty:
        print("No rows found where either Top 1 ddx hit or Top 5 ddx hit is 0.")
        sys.exit(0)
    
    # Print unique diagnoses in the filtered rows
    unique_diagnoses = filtered_df['Diagnosis'].unique()
    print(f"Found {len(filtered_df)} rows where either Top 1 ddx hit or Top 5 ddx hit is 0.")
    print(f"These rows contain {len(unique_diagnoses)} unique diagnoses:")
    for diagnosis in unique_diagnoses:
        print(f"- {diagnosis}")
    
    # Create new columns for the analysis
    filtered_df['GT_Diagnosis_Accuracy'] = None
    filtered_df['GT_Diagnosis_Confidence'] = None
    filtered_df['LLM_Judge_Rationale'] = None
    filtered_df['OpenAI_Diagnosis_Accuracy'] = None
    filtered_df['OpenAI_Diagnosis_Confidence'] = None
    filtered_df['OpenAI_Judge_Rationale'] = None
    
    # Process only the first 5 rows for testing
    # test_df = filtered_df.head(5)
    print(f"Processing first {len(filtered_df)} rows for testing...")
    
    # Analyze each case with tqdm progress bar
    for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Analyzing cases"):
        clinical_notes = row['Clinical Notes']
        diagnosis = row['Diagnosis']
        
        print(f"Analyzing case {index} with diagnosis: {diagnosis}")
        
        # Analyze the ground truth diagnosis using Gemini
        gemini_analysis = llm_judge_diagnosis(clinical_notes, diagnosis)
        
        # Analyze the ground truth diagnosis using OpenAI
        openai_analysis = openai_judge_diagnosis(clinical_notes, diagnosis)
        
        # Update the dataframe with Gemini results
        filtered_df.at[index, 'GT_Diagnosis_Accuracy'] = "Yes" if gemini_analysis['is_accurate'] else "No"
        filtered_df.at[index, 'GT_Diagnosis_Confidence'] = gemini_analysis['confidence']
        filtered_df.at[index, 'LLM_Judge_Rationale'] = gemini_analysis['rationale']
        
        # Update the dataframe with OpenAI results
        filtered_df.at[index, 'OpenAI_Diagnosis_Accuracy'] = "Yes" if openai_analysis['is_accurate'] else "No"
        filtered_df.at[index, 'OpenAI_Diagnosis_Confidence'] = openai_analysis['confidence']
        filtered_df.at[index, 'OpenAI_Judge_Rationale'] = openai_analysis['rationale']
    
    # Save the filtered dataframe to a new CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Filtered {len(filtered_df)} rows where either Top 1 ddx hit or Top 5 ddx hit is 0.")
    print(f"Results saved to {output_file}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)