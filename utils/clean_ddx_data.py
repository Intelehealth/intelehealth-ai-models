import dspy
import pandas as pd

df = pd.read_csv("../data/DDx_database-190-cases-data.csv")

print(df.columns)

new_df = df[["Case No.", "Specialty", "Case ID", "Final Diagnosis", "Patient Case Prompt"]]

print(new_df)

training_examples = []
import sys
for index,row in new_df.iterrows():
    print("-------------------")
    print(index)
    case_id = row["Case ID"]
    diagnosis = row["Final Diagnosis"]
    patient_case_prompt = row["Patient Case Prompt"]
    history = ""
    prompt = ""
    import re
    
    delimiter = """You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis. Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."""

# Split the text
    parts = patient_case_prompt.split(delimiter, maxsplit=1)
    instruction = delimiter.strip()
    case_details = parts[1].strip() if len(parts) > 1 else ""

    if len(case_details) < 2:
        parts = patient_case_prompt.split("HISTORY", maxsplit=1)
        instruction = parts[0].strip()  # Everything before "HISTORY"
        case_details = "HISTORY \n" + parts[1].strip() if len(parts) > 1 else ""  # Everything after "HISTORY"
    
    if len(case_details) < 2:
        parts = patient_case_prompt.split("Hx", maxsplit=1)
        instruction = parts[0].strip()  # Everything before "HISTORY"
        case_details = "Hx \n" + parts[1].strip() if len(parts) > 1 else ""  # Everything after "HISTORY"

    if len(case_details) < 2:
        parts = patient_case_prompt.split(" Salient features", maxsplit=1)
        instruction = parts[0].strip()  # Everything before "Salient Features"
        case_details = " Salient features \n" + parts[1].strip() if len(parts) > 1 else ""  # Everything after "Salient Features"

    training_examples.append({
        "Case No.": row["Case No."],
        "speciality": row["Specialty"],
        "case_id": case_id,
        "diagnosis": diagnosis,
        "original": row["Patient Case Prompt"],
        "patient_case_prompt": instruction,
        "history": case_details
    })
    
    

training_df = pd.DataFrame(training_examples)
print(training_df)
training_df.to_csv("../data/DDx_database-190-cases-data-cleaned.csv", index=False)