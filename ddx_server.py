import dspy
import time

import dspy
from utils.metric_utils import load_gemini_lm_prod, load_open_ai_lm, load_gemini_lm
from dotenv import load_dotenv
from modules.DDxModule import DDxModule

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


load_dotenv(
    "ops/.env"
)

load_gemini_lm_prod()

app = FastAPI(
    title="Differential Diagnosis Server",
    description="A simple API serving a DSPy Chain of Thought program for DDx",
    version="1.0.0"
)

class DDxInfo(BaseModel):
    case: str
    question: str

cot = DDxModule()
cot.load("outputs/" + "15_01_2025_ddx_gemini2_only_num_trials_20_patient_data_top_k5_NA_diag_questions_nas_trial1.json")

dspy_program = dspy.asyncify(cot)

@app.post("/predict")
async def ddx(request_body: DDxInfo):
    try:
        result = await dspy_program(case=request_body.case,question=request_body.question)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health-status")
async def health_status():
    return {
        "status": "AVAILABLE",
        "description": "Service status for DDX server"
    }