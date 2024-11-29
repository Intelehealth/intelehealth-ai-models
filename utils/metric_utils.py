import openai
import os
import json
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")


client = OpenAI(
  api_key = OPENAI_API_KEY,
  organization=OPENAI_ORG_ID,
  project=OPENAI_PROJECT_ID,
)


class DdxResponse(BaseModel):
    score: int
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

def openai_llm_judge(gold, pred, trace=None):
    
    print("############## evaluating open ai llm judge ###############")
    print(gold.diagnosis)
    pred_diagnosis = pred.output.diagnosis
    print(pred_diagnosis)

    print("\n")
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."},   
            {"role": "user", "content": f"Expected output: " + gold.diagnosis},
            {"role": "user", "content": f"Predicted output: " + pred_diagnosis },
            {"role": "user", "content": """Evaluate the semantic similarity between the predicted and expected outputs. Consider the following: 
             1. Are the diagnosis similar?
             2. Is the core meaning preserved even if the wording differs from medical terminologies and synomyms for the diagnosis?
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
    print("Rationale: ", content.rationale)
    rationale = content.rationale

    

    return score
