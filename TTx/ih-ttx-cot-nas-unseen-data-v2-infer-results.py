import dspy
from utils.metric_utils import load_gemini2_lm, metric_fun, openai_llm_judge, load_gemini_lm, load_open_ai_lm, load_hyperbolic_llama_3_1, load_open_ai_o1_mini_lm
from dotenv import load_dotenv
import pandas as pd
import argparse
import os

from modules.TTxv2Module import TTxv2Module

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run treatment recommendation inference on NAS unseen data')
parser.add_argument('--input_csv', type=str, default='/Users/bsb/work/intelehealth-ai-models/data/NAS_UnseenData_02042025_v3_medications.csv',
                    help='Input CSV file containing the patient data')
parser.add_argument('--output_csv', type=str, required=True,
                    help='Output CSV file to save the results')
parser.add_argument('--model', type=str, choices=['gemini2', 'openai', 'llama'], default='gemini2',
                    help='LLM model to use')
parser.add_argument('--trained_file', type=str, default='24_04_2025_12_11_ttx_v2_gemini_cot_nas_v2_combined_llm_judge.json',
                    help='Trained model file to load')
parser.add_argument('--module', type=str, choices=['ttx_v2'], default='ttx_v2',
                    help='Module type to use (currently only ttx_v2 supported)')
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

cot = TTxv2Module()
df = pd.read_csv(args.input_csv)
print(df.columns)

new_df = df[df.columns]

print(new_df)
print(df.shape)

question = "What is the relevant medication, the strength, route form, the dosage, frequency, number of days to take the medication and the reason for the medication for the patient given the diagnosis and patient case?"

cot.load("outputs/" + args.trained_file)

def receive_new_data(new_df):
    for index, row in new_df[:].iterrows():
        print("############################################")
        case_id = row["Visit_id"]
        gt_diagnosis = row["Diagnosis"]
        patient_case = row["Clinical_notes"]
        print("ROW -----> ", case_id, index)
        
        # Get all columns from the original row
        resp = row.to_dict()
        
        # Add LLM output fields
        resp["LLM_medication_recommendations"] = ""
        resp["LLM_medical_advice"] = ""
        
        try:
            output = cot(case=patient_case, diagnosis=gt_diagnosis)
            print("medication recommendations: ", output.output.medication_recommendations)
            
            resp["LLM_medication_recommendations"] = output.output.medication_recommendations
            resp["LLM_medical_advice"] = output.output.medical_advice

        except Exception as e:
            print(e)
            print("exception happened")
            resp["LLM_medication_recommendations"] = ""
            resp["LLM_medical_advice"] = ""
            
        yield resp

# Write data as soon as a new row is available
file_exists = os.path.isfile(args.output_csv)

for new_data in receive_new_data(new_df):
    # Append the new row to the DataFrame for display
    new_row_df = pd.DataFrame([new_data])
    fdf = pd.concat([df, new_row_df], ignore_index=True)
    
    # Write the row to CSV immediately - with header only if file doesn't exist yet
    new_row_df.to_csv(args.output_csv, mode='a', header=not file_exists, index=False)
    file_exists = True  # Set to True after first write
    
    # Print the current state of the DataFrame
    print(fdf) 