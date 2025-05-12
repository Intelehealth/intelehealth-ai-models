import dspy
import time
import json
from utils.metric_utils import load_gemini_lm_prod, load_open_ai_lm, load_gemini_lm, load_gemini2_lm, load_gemini2_5_lm
from dotenv import load_dotenv
from modules.DDxModule import DDxModule
from modules.DDxMulModule import DDxMulModule
from modules.TelemedicineDDxModule import TelemedicineDDxModule
from google import genai
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prompt_config import prompt_config


load_dotenv(
    "ops/.env"
)

load_gemini2_lm()

# Configure OpenAI API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Differential Diagnosis Server",
    description="A simple API serving a DSPy Chain of Thought program for DDx",
    version="1.0.0"
)

class BaseDDxRequest(BaseModel):
    case: str
    model_name: str

class DDxRequestV1(BaseDDxRequest):
    pass

class DDxRequestV2(BaseDDxRequest):
    pass


async def transform_diagnosis_output(llm_output: dict) -> dict:
    """Transform the LLM output into the desired format using another LLM call."""
    transform_prompt = f"""Given the following diagnosis output from a medical LLM, transform it into a structured format.
    The input is:
    {str(llm_output)}

    Transform this into a response with the following structure:
    {{
        "result": [
            {{
                "diagnosis": "diagnosis name",
                "rationale": [
                    {{
                        "category": "Clinical Relevance and Features",
                        "content": "detailed explanation of clinical features"
                    }},
                    {{
                        "category": "Lack of Fit Reasoning",
                        "content": "explanation of why this diagnosis might not fit"
                    }},
                    {{
                        "category": "Relevance to Rural India",
                        "content": "explanation of rural context if applicable"
                    }},
                    {{
                        "category": "Clinical Relevance",
                        "content": "general medical explanation"
                    }}
                ],
                "likelihood": "likelihood score"
            }},
            ...
        ],
        "conclusion": "conclusion text"
    }}

    For each diagnosis in the input:
    - diagnosis: The name of the diagnosis
    - rationales: A list of rationale objects, each containing:
        * category: The category of the rationale. Common categories are:
            - "Clinical Relevance and Features": Details about how symptoms match the diagnosis
            - "Lack of Fit Reasoning": Why certain symptoms or factors make the diagnosis less likely
            - "Relevance to Rural India": Context specific to rural healthcare settings
            - "Clinical Relevance": General medical explanation of the condition
        * content: The detailed explanation for that category
    - likelihood: The likelihood score (High, Moderate-High, Moderate, Low-Moderate, Low)

    The conclusion should be a concise summary of the most likely diagnoses and key considerations.
    
    Note: Break down any multi-line rationales into separate rationale objects by category. Include "Lack of Fit Reasoning" when the diagnosis is less likely or when there are factors that make it less probable.
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=transform_prompt,
        )
        print("response: -----")
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
        return transformed
    except Exception as e:
        print(f"Error transforming diagnosis output: {e}")
        # Fallback to a basic transformation if parsing fails
        return {
            "result": [],
            "conclusion": "Error processing diagnosis output. Please try again."
        }

@app.post("/predict/v1")
async def ddx_v1(request_body: DDxRequestV1):
    cot = None
    prompt = ""
    if request_body.model_name == "gemini-2.0-flash":
        cot = DDxModule()
        cot.load("outputs/" + "10_02_2025_ddx_gemini2_only_num_trials_20_ayu_data_top_k5_single_diagnosis.json")
        prompt = prompt_config[1]
    else:
        raise HTTPException(status_code=400, detail="Invalid model name for v1")

    dspy_program = dspy.asyncify(cot)

    try:
        print("prompt selected: ", prompt)
        result = await dspy_program(case=request_body.case, question=prompt)
        print(result)
        if hasattr(result, 'output') and hasattr(result.output, 'diagnosis') and result.output.diagnosis == "NA":
            print("no diagnosis possible")
            return {
                "status": "success",
                "data": "The Input provided does not have enough clinical details for AI based assessment."
            }

        # Transform the diagnosis output
        transformed_output = await transform_diagnosis_output(result.toDict())
        return {
            "status": "success",
            "data": transformed_output
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")


@app.post("/predict/v2")
async def ddx_v2(request_body: DDxRequestV2):
    cot = None
    prompt = ""
    if request_body.model_name == "gemini-2.0-flash-001":
        cot = TelemedicineDDxModule()
        cot.load("outputs/" + "19_03_2025_21_31_ddx_gemini_cot_ayu_cleaned_data_llm_judge.json")
        prompt = prompt_config[2]
    else:
        raise HTTPException(status_code=400, detail="Invalid model name for v2")

    dspy_program = dspy.asyncify(cot)

    try:
        print("prompt selected: ", prompt)
        result = await dspy_program(case=request_body.case, question=prompt)
        print(result)
        if hasattr(result, 'output') and hasattr(result.output, 'diagnosis') and result.output.diagnosis == "NA":
            print("no diagnosis possible")
            return {
                "status": "success",
                "data": "The Input provided does not have enough clinical details for AI based assessment."
            }

        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")


@app.get("/health-status")
async def health_status():
    return {
        "status": "AVAILABLE",
        "description": "Service status for DDX server"
    }