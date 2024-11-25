import dspy
import pandas as pd

df = pd.read_csv("data/ddx_cases.csv")

print(df.columns)

new_df = df[["Case ID", "Diagnosis", "Patient Case Prompt"]]

print(new_df)

training_examples = []

for index,row in new_df.iterrows():
    case_id = row["Case ID"]
    diagnosis = row["Diagnosis"]
    patient_case_prompt = row["Patient Case Prompt"]

    example = dspy.Example(
        case_id = case_id,
        case_info = patient_case_prompt,
        diagnosis = diagnosis
    ).with_inputs("Patient Case Prompt")

    training_examples.append(example)

print(len(training_examples))

print(training_examples[1])