import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge, load_gemini_lm, load_open_ai_lm
import os
import random
from dotenv import load_dotenv
import pandas as pd

from modules.DDxModule import DDxModule
load_dotenv(
    "ops/.env"
)
import sys

#load_gemini_lm()
load_open_ai_lm()

cot = DDxModule()
cot.load("outputs/" + "03_01_2025_ddx_open_ai_only_num_trials_20_top_k5_nas.json")

df = pd.read_csv("./data/DDx_database-unseen-data-NAS-without-corrections.csv")
print(df.columns)

new_df = df[df.columns]

print(new_df)


question = "You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis. Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."


def receive_new_data(new_df):
    for index,row in new_df[:].iterrows():
        print("############################################")
        case_id = row["OpenMRS_id"]
        gt_diagnosis = row["Diagnosis"]
        patient_case = row["Clinical_notes"]
        print("ROW -----> ", case_id, index)
        resp = {
                "case_id": case_id,
                "location_village": row["Location_village"],
                "patient_case": patient_case,
                "gt_diagnosis": gt_diagnosis,
                "LLM_diagnosis": "",
                "LLM_rationale" : "",
                "conclusion": "",
        }
        try:
            output = cot(case=patient_case, question=question)
            print("diagnosis: ", output.output.diagnosis)
            print("rationale: ", output.output.rationale)
            resp["LLM_diagnosis"] =  output.output.diagnosis
            resp["LLM_rationale"] = output.output.rationale
            resp["conclusion"] = output.output.conclusion
        except:
            print("exception happened")
            resp["LLM_diagnosis"] =  ""
            resp["LLM_rationale"] = ""
            resp["conclusion"] = ""
        yield resp

# Write data as soon as a new row is available
for new_data in receive_new_data(new_df):
    # Append the new row to the DataFrame
    new_row_df = pd.DataFrame([new_data])
    # Concatenate the new row DataFrame with the existing DataFrame
    fdf = pd.concat([df, new_row_df], ignore_index=True)
    # Optionally, write the DataFrame to a CSV file after each row
    new_row_df.to_csv('04_01_2025_openai_only_output_nas_unseen_cot_topk_5_more_rationale_conclusion.csv', mode='a', header=False, index=False)

    # Print the current state of the DataFrame
    print(fdf)
# pd.DataFrame(training_examples).to_csv("gemini_medpalm_small_5_added_cot_inference_cot.csv", index=False)
# pd.DataFrame(non_training_examples).to_csv("gemini_medpalm_small_5_added_nt_cot_inference_cot.csv", index=False)