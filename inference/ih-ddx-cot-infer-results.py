import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge, load_gemini_lm
import os
import random
from dotenv import load_dotenv
import pandas as pd

from modules.DDxModule import DDxModule
load_dotenv(
    "ops/.env"
)
import sys

load_gemini_lm()

cot = DDxModule()
cot.load("outputs/" + "ddx_open_ai_gemini_pro_medpalm_added_cot_traces_cleaned_data_llm_judge_metric.json")

df = pd.read_csv("data/DDx_database-190-cases-data-cleaned.csv")
print(df.columns)

new_df = df[["case_id", "speciality", "diagnosis", "patient_case_prompt", "history"]]

print(new_df)


def receive_new_data(new_df):
    for index,row in new_df[:].iterrows():
        print("############################################")
        case_id = row["case_id"]
        gt_diagnosis = row["diagnosis"]
        patient_case = row["history"]
        question = row["patient_case_prompt"]
        print("ROW -----> ", case_id, index)
        resp = {
                "case_id": case_id,
                "speciality": row["speciality"],
                "patient_case": patient_case,
                "gt_diagnosis": gt_diagnosis,
                "LLM_diagnosis": "",
                "LLM_rationale" : ""
        }
        try:
            output = cot(case=patient_case, question=question)
            print("diagnosis: ", output.output.diagnosis)
            print("rationale: ", output.output.rationale)
            resp["LLM_diagnosis"] =  output.output.diagnosis
            resp["LLM_rationale"] = output.output.rationale
        except:
            print("exception happened")
            resp["LLM_diagnosis"] =  ""
            resp["LLM_rationale"] = ""
        yield resp

# Write data as soon as a new row is available
for new_data in receive_new_data(new_df):
    # Append the new row to the DataFrame
    new_row_df = pd.DataFrame([new_data])
    # Concatenate the new row DataFrame with the existing DataFrame
    fdf = pd.concat([df, new_row_df], ignore_index=True)
    # Optionally, write the DataFrame to a CSV file after each row
    new_row_df.to_csv('gemini_final_output.csv', mode='a', header=False, index=False)

    # Print the current state of the DataFrame
    print(fdf)
# pd.DataFrame(training_examples).to_csv("gemini_medpalm_small_5_added_cot_inference_cot.csv", index=False)
# pd.DataFrame(non_training_examples).to_csv("gemini_medpalm_small_5_added_nt_cot_inference_cot.csv", index=False)