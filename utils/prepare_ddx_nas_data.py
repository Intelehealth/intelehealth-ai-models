import dspy
import pandas as pd

df = pd.read_csv("./data/DDx_database-unseen-data-NAS-without-corrections.csv")

print(df.columns)

new_df = df[df.columns]

print(new_df)

qn = "You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis. Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."

training_examples = []

for index,row in new_df.iterrows():
    case_id = row["OpenMRS_id"]
    diagnosis = row["Diagnosis"]
    patient_case_prompt = row["Clinical_notes"]
    
    if str(diagnosis) == "nan":
        continue
    else:
        example = dspy.Example(
            case_id = str(case_id),
            case = str(patient_case_prompt),
            question = qn,
            diagnosis = str(diagnosis)
        ).with_inputs("case", "question")

        training_examples.append(example)

print(len(training_examples))

print(training_examples[1])


def ret_training_examples():
    return training_examples