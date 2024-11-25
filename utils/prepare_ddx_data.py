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
        question = "For the shared patient case output a top diagnosis string. " + patient_case_prompt,
        answer = diagnosis
    ).with_inputs("question")

    training_examples.append(example)

print(len(training_examples))

print(training_examples[1])


def ret_training_examples():
    return training_examples