import dspy
import pandas as pd

# df = pd.read_csv("data/seperated_patient_prompts.csv")

# print(df.columns)

# new_df = df[["case_id", "diagnosis", "patient_case_prompt", "history"]]

df = pd.read_csv("data/DDx_database-190-cases-data-cleaned.csv")

print(df.columns)

new_df = df[["case_id", "diagnosis", "patient_case_prompt", "history"]]

print(new_df)

training_examples = []

for index,row in new_df.iterrows():
    case_id = row["case_id"]
    diagnosis = row["diagnosis"]
    patient_case_prompt = row["history"]
    question = row["patient_case_prompt"]

    example = dspy.Example(
        case_id = case_id,
        case = str(patient_case_prompt),
        question = question,
        diagnosis = diagnosis
    ).with_inputs("case", "question")

    training_examples.append(example)

print(len(training_examples))

print(training_examples[1])


def ret_training_examples():
    return training_examples