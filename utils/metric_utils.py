import openai
import os
import json
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import time
import dspy

load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
  api_key = OPENAI_API_KEY,
  organization=OPENAI_ORG_ID,
  project=OPENAI_PROJECT_ID,
)


class DdxResponse(BaseModel):
    score: int
    rationale: str


class GDdxResponse(BaseModel):
    score: float
    rationale: str

def metric_fun(gold, pred, trace=None):
    print(gold.diagnosis)
    print(pred.diagnosis)

    gold_d = gold.diagnosis.lower().strip(":")
    pred_d = pred.diagnosis.lower()

    if gold_d == pred_d:
        return 1.0
    elif gold_d in pred_d or pred_d in gold_d:
        return 1.0
    else:
        return 0.0

import sys

def openai_llm_judge(gold, pred, trace=None):
    
    print("############## evaluating open ai llm judge ###############")
    print(gold.diagnosis)
    pred_diagnosis = pred.output
    print(pred_diagnosis)


    print("\n")
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."},   
            {"role": "user", "content": f"Expected output: " + gold.diagnosis},
            {"role": "user", "content": f"Predicted output: " + str(pred_diagnosis) },
            {"role": "user", "content": """Evaluate the semantic similarity between the predicted and expected outputs. Consider the following: 
             1. Is the expected diagnosis present in the top 5 diagnosises predicted?
             2. Is the core meaning preserved even if the wording differs from medical terminologies and synonyms for the matching expected and predicted diagnosis?
             3. Are there any significant omissions or additions in the predicted output?
             
             Provide output as valid JSON with field `score` as '1' for similar and '0' for not similar and field `rationale` having the reasoning string for this score."""}
        ],
        response_format = DdxResponse
    )
    # print(response)
    # Extract the content from the first choice
    # content = response.choices[0].message.content
    content = response.choices[0].message.parsed

    print("Response from llm:")
    print("LLM Judge score: ", content.score)
    score = content.score
    rationale = content.rationale
    print("Rationale: ", content.rationale)
    

    # time.sleep(2)

    return score


def load_gemini_lm():
    gemini = dspy.Google("models/gemini-1.5-pro", api_key=GEMINI_API_KEY, temperature=1.0)
    dspy.settings.configure(lm=gemini, max_tokens=10000)

def load_open_ai_lm():
    lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=1.0)
    dspy.configure(lm=lm)