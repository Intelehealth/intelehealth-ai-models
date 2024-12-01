import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge
import os
import random
from dotenv import load_dotenv
import pandas as pd

from modules.DDxModule import DDxModule
load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=1.0)

dspy.configure(lm=lm)


cot = DDxModule()
cot.load("outputs/" + "ddx_open_ai_gpt-01_cot_trial_cleaned_data_llm_judge_metric.json")


df = pd.read_csv("data/DDx_database-190-cases-data-cleaned.csv")

print(df.columns)

new_df = df[["case_id", "diagnosis", "patient_case_prompt", "history"]]

print(new_df)

training_examples = []

i = 0
for index,row in new_df.iterrows():
    # if i == 5:
    #     break
    # i = i + 1
    case_id = row["case_id"]
    gt_diagnosis = row["diagnosis"]
    patient_case_prompt = row["history"]
    question = row["patient_case_prompt"]

    output = cot(case=patient_case_prompt, question=question)

    resp = {
        "case_id": case_id,
        "gt_diagnosis": gt_diagnosis,
        "LLM_diagnosis": output.output.diagnosis,
        "LLM_rationale" : output.output.rationale
    }

    training_examples.append(resp)


pd.DataFrame(training_examples).to_csv("gpt4o_miprov2_inference_cot.csv", index=False)