import dspy
import time
import json

import dspy
from utils.metric_utils import load_gemini_lm_prod, load_open_ai_lm, load_gemini_lm, load_gemini2_lm, load_gemini2_5_lm
from dotenv import load_dotenv
from modules.TTxModule import TTxModule
from modules.TTxv2Module import TTxv2Module
from modules.TTxv3Module import TTxv3Module
from google import genai
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prompt_config import prompt_config
from ttx_client import process_medications

load_dotenv(
    "ops/.env"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

async def transform_ttx_output(llm_output: dict) -> dict:
    """Transform the TTx LLM output into a structured format using an LLM.
    
    Args:
        llm_output (dict): The raw output from the TTx model
        
    Returns:
        dict: A structured response with the following format:
        {
            "success": bool,
            "medications": [
                {
                    "name": "medication in route-form-strength format",
                    "dosage": "dosage information",
                    "frequency": "frequency of administration",
                    "duration": "duration of treatment",
                    "instructions": "instructions if any for this medication",
                    "rationale": "rationale for the medication",
                    "confidence": "confidence level of the medication"
                },
                ...
            ],
            "medical_advice": [{
                    "1": "Rest and stay hydrated. Drink plenty of warm fluids.",
                    "2": "Monitor temperature and symptoms.",
                    "3": "If symptoms worsen or new symptoms develop, seek further medical advice.",
                    ...
                }
            ],
            "tests_to_be_done": [{
                "test_name": "test_name",
                "test_reason": "test_reason"
                },
                ...
            ],
            "follow_up": [{
                "follow_up_required": "true/false",
                "follow_up_date": "date format example like this 2025-06-15 (format : 'YYYY-MM-DD') from the current date",
                "reason_for_follow_up": "short rationale about the follow up",
                "follow_up_time": "time of the follow up"
                },
                ...
            ],
            "referral": [{
                "referral_to": "referral_to",
                "referral_facility": "referral_facility",
                "priority_of_referral": "priority_of_referral",
                "remark": "remark"
                },
                ...
            ],
            "error": "error message if success is false"
        }
    """
    current_date_str = time.strftime('%Y-%m-%d')
    transform_prompt = f"""Given the following treatment recommendation output from a medical LLM, transform it into a structured format.
    The input is:
    {str(llm_output)}

    Transform this into a response with the following structure:
    {{
        "success": true/false,
        "medications": [
            {{
                "name": "medication in route-form-strength format",
                "dosage": "dosage information",
                "frequency": "frequency of administration",
                "duration": "duration of treatment",
                "instructions": "instructions if any for this medication",
                "rationale": "rationale for the medication",
                "confidence": "confidence level of the medication"
            }},
            ...
        ],
        "medical_advice": [
            {{
                "1": "Rest and stay hydrated. Drink plenty of warm fluids.",
                "2": "Monitor temperature and symptoms.",
                "3": "If symptoms worsen or new symptoms develop, seek further medical advice.",
                ...
            }}
        ],
        "tests_to_be_done": [
            {{
                "test_name": "test_name",
                "test_reason": "test_reason"
            }},
            ...
        ],
            "follow_up": [{{
                "follow_up_required": "true/false",
                "follow_up_date": "Calculate by adding the follow-up period from input (e.g., '3 days', '1 week') to the current date ({current_date_str}). Format: 'YYYY-MM-DD'.",
                "follow_up_time": "time of the follow up",
                "reason_for_follow_up": "short rationale about the follow up"
                }},
            ...
        ],
        "referral": [
            {{  
                "referral_to": "referral_to",
                "referral_facility": "referral_facility",
                "priority_of_referral": "priority_of_referral",
                "remark": "remark"
            }},
            ...
        "error": "error message if success is false"
    }}

    Guidelines for transformation:
    1. Extract all medication recommendations and structure them into the medications array
    2. For each medication:
       - name: Format as "MedicationName Strength Route Form" (e.g., "Paracetamol 500 mg Oral Tablet")
       - dosage: The recommended dosage (e.g., "1 tablet", "10 ml")
       - frequency: Use standard terms (e.g., "Once daily", "Twice daily", "Thrice daily")
       - duration: Include units (e.g., "3 Days", "1 Week")
       - instructions: Include timing and food instructions (e.g., "After food", "At bedtime")
    3. For medical_advice:
       - **CRITICAL: The `medical_advice` field MUST be an array containing a single object with numbered keys.**
       - Split the input string using newlines (\\n) or numbered points (1., 2., etc.)
       - Create numbered keys starting from "1", "2", "3", etc. for each advice point
       - Remove any leading numbers (1., 2., etc.) from the advice text before assigning to the numbered key
       - If the input is empty or not provided, return an empty array [] for `medical_advice`.
    4. For the `follow_up` array in the output JSON:
       - The input to this transformation task (shown as `llm_output` in the prompt context above, often found within a `data` object like `data.follow_up`) may contain a `follow_up` field. This field is typically a string formatted with key-value pairs separated by newlines (e.g., `follow_up_required: true\\nnext_followup_duration: 3\\nnext_followup_units: days\\nnext_followup_reason: ...`).
       - You MUST parse this input `follow_up` string to extract the following values:
         - `follow_up_required` (which will be a string like "true" or "false")
         - `next_followup_duration` (a number, e.g., 3)
         - `next_followup_units` (a string indicating units, e.g., "days", "weeks", "months")
         - `next_followup_reason` (a descriptive string)
       - To calculate the `follow_up_date` for the output JSON:
         - Use the extracted `next_followup_duration` and `next_followup_units`.
         - Add this duration to the `current_date` (which for this task is {current_date_str}).
         - Format the resulting date as 'YYYY-MM-DD'.
       - Example of date calculation: If `current_date` is {current_date_str}, and the input `follow_up` string contains `next_followup_duration: 3` and `next_followup_units: days`, then the `follow_up_date` in your output JSON should be the date 3 days from {current_date_str}.
       - The `reason_for_follow_up` field in the output JSON should be populated with the value extracted from `next_followup_reason`.
       - The `follow_up_required` field in the output JSON (which should be a boolean true/false) should be set based on the parsed `follow_up_required` string (e.g., "true" becomes true).
       - If the input `follow_up` string itself contains an absolute date instead of duration/units, use that absolute date and ensure it's formatted as 'YYYY-MM-DD'. If essential parts for calculation (duration, units) are missing and no absolute date is found, you may need to omit the `follow_up_date` or handle it as an error if appropriate.
       - for the follow_up_time, use one of the times btw 10 am to 6 pm with with this format: "HH:MM" or "HH:MM AM/PM" it should be in the same timezone as the current date.
    5. Set success to false and include an error message if:
       - No valid medication recommendations are found
       - The input format is invalid
       - The treatment is marked as "NA" or indicates insufficient information
    5. Ensure all medication names are properly formatted:
       - Use proper capitalization
       - Include strength in mg/ml
       - Specify route (Oral, Topical, etc.)
       - Include form (Tablet, Capsule, Solution, etc.)
    6. For combination medications:
       - List all active ingredients in the name
       - Include total strength of each component
    7. If tests_to_be_done is None, return an empty array [] for tests_to_be_done.
    8. If follow_up is None, return an empty array [] for follow_up.
    9. If referral is None, return an empty array [] for referral.
    
    Note:
    - If the input indicates that no treatment is possible or there's insufficient information,
      set success to false and include an appropriate error message.
    - **Reiteration for `medical_advice`**: It must be an array containing a single object with numbered keys. For example: `[{{"1": "Advice point 1.", "2": "Advice point 2.", "3": "Advice point 3."}}]` and NOT `["Advice point 1.", "Advice point 2."]`.
    - If `medical_advice` source is empty, return an empty array `[]`.
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=transform_prompt,
        )
        print("Transformation response: -----")
        print(response)
        print("-----")
        
        # Extract JSON from the markdown-formatted response
        response_text = response.text
        if "```json" in response_text:
            # Find the content between ```json and ```
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        else:
            # If no markdown formatting, try to find JSON between any backticks
            json_str = response_text.split("```")[1].strip() if "```" in response_text else response_text.strip()
            
        # Parse the extracted JSON
        transformed = json.loads(json_str)
        
        # Ensure medical_advice and adverse_effects are always arrays
        if "medical_advice" not in transformed or not isinstance(transformed["medical_advice"], list):
            transformed["medical_advice"] = []
        # if "adverse_effects" not in transformed or not isinstance(transformed["adverse_effects"], list):
        #     transformed["adverse_effects"] = []
            
        # Handle empty or null medical_advice
        if transformed["medical_advice"] is None or transformed["medical_advice"] == "":
            transformed["medical_advice"] = []
        
        # Ensure medical_advice maintains the correct format with numbered keys
        # The format should be an array containing a single object with numbered keys
            
        return transformed
    except Exception as e:
        print(f"Error transforming TTx output: {e}")
        # Fallback to a basic error response if transformation fails
        return {
            "success": False,
            "medications": [],
            "medical_advice": [],
            "tests_to_be_done": [],
            "follow_up": [],
            "referral": [],
            "error": "Error processing treatment recommendations. Please try again."
        }

load_gemini2_lm()

app = FastAPI(
    title="Treatment Recommendation Server",
    description="A simple API serving a DSPy Chain of Thought program for TTx",
    version="1.0.0"
)

class BaseTTxRequest(BaseModel):
    case: str
    diagnosis: str
    model_name: str


@app.post("/ttx/v1")
async def ttx_v1(request_body: BaseTTxRequest):
    cot = None
    if request_body.model_name == "gemini-2.0-flash":
        cot = TTxv2Module()
        cot.load("outputs/" + "24_04_2025_12_11_ttx_v2_gemini_cot_nas_v2_combined_llm_judge.json")
    elif request_body.model_name == "gemini-2.5-flash-preview-04-17":
        cot = TTxv3Module()
        # cot.load("outputs/" + "06_05_2025_12_33_ttx_v3_gemini2_5_cot_nas_v2_combined_medications.json")
        cot.load("outputs/" + "27_05_2025_13_26_ttx_v3_gemini2_5_cot_nas_v2_combined_medications_rationale_referral_tests_followup.json")
    else:
        raise HTTPException(status_code=400, detail="Invalid model name for TTx v2")

    dspy_program = dspy.asyncify(cot)

    try:
        result = await dspy_program(case=request_body.case, diagnosis=request_body.diagnosis)
        print(result)
        
        if hasattr(result, 'output') and hasattr(result.output, 'treatment') and result.output.treatment == "NA":
            print("no treatment possible")
            return {
                "status": "success",
                "data": "The Input provided does not have enough clinical details for AI based treatment recommendation."
            }

        response_json = {
            "data": result.toDict()
        }
        print("--------------------------------")
        print(response_json)
        print("--------------------------------")
        
        # Transform the output using the new function
        transformed_result = await transform_ttx_output(response_json)
        print(transformed_result)
        print("--------------------------------")

        return transformed_result
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")


@app.get("/health-status")
async def health_status():
    return {
        "status": "AVAILABLE",
        "description": "Service status for TTx server"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 