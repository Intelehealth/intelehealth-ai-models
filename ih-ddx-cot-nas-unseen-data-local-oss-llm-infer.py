import dspy
import time
from utils.metric_utils import load_ollama_url, openai_llm_judge, load_gemini_lm, load_open_ai_lm, load_hyperbolic_llama_3_1, load_open_ai_o1_mini_lm
import os
import random
from dotenv import load_dotenv
import pandas as pd

from modules.DDxLocalModule import DDxLocalModule
load_dotenv(
    "ops/.env"
)
import sys

#load_gemini_lm()
#load_open_ai_lm()
load_ollama_url()
#load_hyperbolic_llama_3_1()

#cot = DDxModule()
cot = DDxLocalModule()
cot.load("outputs/" + "24_01_2025_ddx_open_ai_ollama_meditron_70_bsr_patient_cleaned_data_llm_judge.json")

df = pd.read_csv("./data/DDx_database-unseen-data-NAS-without-corrections.csv")
print(df.columns)

new_df = df[df.columns]

print(new_df)

question = """You are a doctor/physician assigned with task of diffential diagnosis on a patient from rural India.
            Given their case with the history of presenting illness, symptoms, their physical exams, and demographics, give me the top 5 differential diagnosis for this patient with likelihood. 

            Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient.
"""


def receive_new_data(new_df):
    start = time.time()
    for index,row in new_df[:].iterrows():
        if index < 187:
            continue
        else:
            print("############################################")
            case_id = row["OpenMRS_id"]
            gt_diagnosis = row["Diagnosis"]
            gt_diagnosis_corr = row["Corrections_in_Diagnosis"]
            patient_case = row["Clinical_notes"]
            print("ROW -----> ", case_id, index)
            resp = {
                    "case_id": case_id,
                    "location_village": row["Location_village"],
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "gt_diagnosis_corr": gt_diagnosis_corr,
                    "LLM_diagnosis": "",
                    "LLM_rationale" : "",
                    "conclusion": "",
                    # "further_questions": ""
            }
            try:
                output = cot(case=patient_case, question=question)
                print("diagnosis: ", output.output.diagnosis)
                print("rationale: ", output.output.diagnosis)
                resp["LLM_diagnosis"] =  output.output.diagnosis
                resp["LLM_rationale"] = output.output.reasoning
                # resp["conclusion"] = output.output.conclusion
                # resp["further_questions"] = output.output.further_questions
            except Exception as e:
                print(e)
                print("exception happened")
                resp["LLM_diagnosis"] =  ""
                resp["LLM_rationale"] = ""
                resp["conclusion"] = ""
                resp["further_questions"] = ""
            end = time.time() - start
            print("time taken: ", end)
            yield resp

# Write data as soon as a new row is available
for new_data in receive_new_data(new_df):
    # Append the new row to the DataFrame
    new_row_df = pd.DataFrame([new_data])
    # Concatenate the new row DataFrame with the existing DataFrame
    fdf = pd.concat([df, new_row_df], ignore_index=True)
    # Optionally, write the DataFrame to a CSV file after each row
    new_row_df.to_csv('24_01_2025_meditron_mini_patient_data_inference_output_nas_unseen_cot_topk_5_1.csv', mode='a', header=False, index=False)

    # Print the current state of the DataFrame
    print(fdf)
