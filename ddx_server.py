import dspy
import time

import dspy
from utils.metric_utils import load_gemini_lm_prod, load_open_ai_lm, load_gemini_lm, load_gemini2_lm
from dotenv import load_dotenv
from modules.DDxModule import DDxModule
from modules.DDxMulModule import DDxMulModule

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prompt_config import prompt_config


load_dotenv(
    "ops/.env"
)

load_gemini2_lm()

app = FastAPI(
    title="Differential Diagnosis Server",
    description="A simple API serving a DSPy Chain of Thought program for DDx",
    version="1.0.0"
)

class DDxInfo(BaseModel):
    case: str
    model_name: str
    prompt_version: int

cot = DDxModule()

@app.post("/predict")
async def ddx(request_body: DDxInfo):
    prompt = ""
    if request_body.model_name == "gemini-2.0-flash":
        if request_body.prompt_version == 1:
            cot.load("outputs/" + "10_02_2025_ddx_gemini2_only_num_trials_20_ayu_data_top_k5_single_diagnosis.json")
            prompt = prompt_config[1]
        else:
            return {
                "status": "error",
                "message": "Invalid prompt version"
            }
    else:
        return {
            "status": "error",
            "message": "Invalid model name"
        }

    dspy_program = dspy.asyncify(cot)

    try:
        result = await dspy_program(case=request_body.case, question=prompt)
        print(result)
        if result.output.diagnosis == "NA":
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
        raise HTTPException(status_code=500, detail=str(e))
        return {
            "status": "error",
            "message": "Internal Server Error. Please try again later."
        }
    

@app.get("/health-status")
async def health_status():
    return {
        "status": "AVAILABLE",
        "description": "Service status for DDX server"
    }