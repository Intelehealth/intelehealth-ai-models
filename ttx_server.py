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
                    "instructions": "instructions if any for this medication"
                },
                ...
            ],
            "medical_advice": [
                "Rest and stay hydrated. Drink plenty of warm fluids.",
                "Monitor temperature and symptoms. If symptoms worsen or new symptoms develop, seek further medical advice."
            ],
            "error": "error message if success is false"
        }
    """
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
                "instructions": "instructions if any for this medication"
            }},
            ...
        ],
        "medical_advice": [
            "Rest and stay hydrated. Drink plenty of warm fluids.",
            "Monitor temperature and symptoms.",
            "If symptoms worsen or new symptoms develop, seek further medical advice."
        ],
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
       - **CRITICAL: The `medical_advice` field MUST be a simple array of strings. Do NOT use an array of objects with numbered keys.**
       - Split the input string using newlines (\\n) or numbered points (1., 2., etc.)
       - Remove the number and period from the start of each point
       - Example of CORRECT `medical_advice` transformation:
         Input: "1. Rest and stay hydrated. Drink plenty of warm fluids.\\n2. Monitor temperature and symptoms."
         Output: [
             "Rest and stay hydrated. Drink plenty of warm fluids.",
             "Monitor temperature and symptoms."
         ]
       - If the input is empty or not provided, return an empty array [] for `medical_advice`.
       - Each point in the `medical_advice` array should be a complete sentence without the leading number.
    4. Set success to false and include an error message if:
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

    Note:
    - If the input indicates that no treatment is possible or there's insufficient information,
      set success to false and include an appropriate error message.
    - **Reiteration for `medical_advice`**: It must be a flat list of strings. For example: `["Advice point 1.", "Advice point 2."]` and NOT `[{{"1": "Advice point 1."}}, {{"2": "Advice point 2."}}]`.
    - Remove any leading numbers (1., 2., etc.) from each advice string.
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
        
        # Ensure each item in medical_advice is a string, not an object
        if transformed["medical_advice"] and isinstance(transformed["medical_advice"], list):
            for i, item in enumerate(transformed["medical_advice"]):
                if isinstance(item, dict):
                    # Extract the value from the numbered key
                    keys = list(item.keys())
                    if keys:
                        transformed["medical_advice"][i] = item[keys[0]]
            
        return transformed
    except Exception as e:
        print(f"Error transforming TTx output: {e}")
        # Fallback to a basic error response if transformation fails
        return {
            "success": False,
            "medications": [],
            "medical_advice": [],
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
        cot.load("outputs/" + "06_05_2025_12_33_ttx_v3_gemini2_5_cot_nas_v2_combined_medications.json")
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