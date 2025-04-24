import dspy
import time

import dspy
from utils.metric_utils import load_gemini_lm_prod, load_open_ai_lm, load_gemini_lm, load_gemini2_lm, load_gemini2_5_lm
from dotenv import load_dotenv
from modules.TTxModule import TTxModule
from modules.TTxv2Module import TTxv2Module

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prompt_config import prompt_config


load_dotenv(
    "ops/.env"
)

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
        "description": "Service status for TTx server"
    } 