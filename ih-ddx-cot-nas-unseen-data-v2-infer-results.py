import dspy
from utils.metric_utils import load_gemini2_lm, metric_fun, openai_llm_judge, load_gemini_lm, load_open_ai_lm, load_hyperbolic_llama_3_1, load_open_ai_o1_mini_lm
from dotenv import load_dotenv
import pandas as pd
import argparse
import os

from modules.DDxModule import DDxModule
from modules.TelemedicineDDxModule import TelemedicineDDxModule

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run differential diagnosis inference on NAS unseen data')
parser.add_argument('--input_csv', type=str, required=True,
                    help='Input CSV file containing the patient data')
parser.add_argument('--output_csv', type=str, required=True,
                    help='Output CSV file to save the results')
parser.add_argument('--model', type=str, choices=['gemini2', 'openai', 'llama'], default='gemini2',
                    help='LLM model to use')
parser.add_argument('--trained_file', type=str, required=True,
                    help='Trained model file to load')
parser.add_argument('--module', type=str, choices=['ddx', 'telemedicine'], default='ddx',
                    help='Module type to use (ddx or telemedicine)')
args = parser.parse_args()

load_dotenv(
    "ops/.env"
)

# Configure model based on command line argument
if args.model == 'gemini2':
    load_gemini2_lm()
elif args.model == 'openai':
    load_open_ai_lm()
elif args.model == 'gemini':
    load_gemini_lm()
elif args.model == 'llama':
    load_hyperbolic_llama_3_1()

cot = DDxModule()
# cot = DDxQnModule()
cot.load("outputs/" + args.trained_file)

df = pd.read_csv(args.input_csv)
print(df.columns)

new_df = df[df.columns]

print(new_df)
print(df.shape)

question = ""
# Initialize appropriate module based on argument
if args.module == 'telemedicine':
    cot = TelemedicineDDxModule()
    question = """
    You are a doctor conducting a telemedicine consultation with a patient in rural India.
    Based on patient history, symptoms, physical exam findings, and demographics:
    1. Provide top 5 differential diagnoses ranked by likelihood
    2. For each: include likelihood score (high/moderate/low) and brief rationale
    3. For high/moderate: mention key features, infections, and rural India relevance
    4. For low: briefly explain why it doesn't fit
    Keep all responses concise and to the point.
    """
elif args.module =='ddx':
    cot = DDxModule()
    question = "You are a doctor with the following patient from rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? For each diagnosis, include the likelihood score and the brief rationale for that diagnosis. For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural india. For a low likelihood diagnosis, include lack of fit reasoning under the rationale for that diagnosis. Please rank the differential diagnoses based on the likelihood and provide a brief explanation for each diagnosis. Please don't include  CASE and Question in the rationale.Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."
else:
    question_1 = "You are a doctor with the following patient from rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? For each diagnosis, include the likelihood score and a brief rationale focusing on the most relevant clinical features and relevance to a rural Indian context. Please rank the differential diagnoses based on the likelihood and provide a brief explanation for each diagnosis. DO NOT include case and question in the output anywhere. Keep rationale and conclusion very brief."

def receive_new_data(new_df):
    for index,row in new_df[:].iterrows():
        # if index < 133:
        #     continue
        # else:
            print("############################################")
            case_id = row["visit_id"]
            gt_diagnosis = row["Diagnosis"]
            patient_case = row["Clinical Notes"]
            print("ROW -----> ", case_id, index)
            # Initialize response dictionary with fields matching module signatures
            if args.module == 'telemedicine':
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                    "LLM_rationale": "",
                    "further_questions": "",
                    "follow_up_recommendations": ""
                }
            else:
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                    "LLM_rationale": ""
                }
            
            try:
                output = cot(case=patient_case, question=question)
                print("diagnosis: ", output.output.diagnosis)
                print("rationale: ", output.output.rationale)
                
                # Set basic fields that exist in both modules
                resp["LLM_diagnosis"] = output.output.diagnosis 
                resp["LLM_rationale"] = output.output.rationale

                # Set additional fields for telemedicine module
                if args.module == 'telemedicine':
                    resp["further_questions"] = output.output.further_questions
                    resp["follow_up_recommendations"] = output.output.follow_up_recommendations

            except Exception as e:
                # Set module-specific empty fields based on module type
                if args.module == 'telemedicine':
                    resp["further_questions"] = ""
                    resp["follow_up_recommendations"] = ""
                print(e)
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
    # Write the DataFrame to a CSV file after each row
    new_row_df.to_csv(args.output_csv, mode='a', header=False, index=False)

    # Print the current state of the DataFrame
    print(fdf)
