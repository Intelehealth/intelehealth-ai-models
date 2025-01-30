import dspy
import pandas as pd

########## PATIENT DATA ################################3
df = pd.read_csv("data/seperated_patient_prompts.csv")
print(df.columns)
new_df = df[["case_id", "diagnosis", "original", "patient_case_prompt", "history"]]


############# AYU ####################
# df = pd.read_csv("./data/DDx_database-190-cases-data-cleaned-ayu.csv")
# print(df.columns)
# new_df = df[["case_id", "diagnosis", "ayu_patient_prompt", "history"]]

# print(new_df)

training_examples = []

for index,row in new_df.iterrows():
    case_id = row["case_id"]
    diagnosis = row["diagnosis"]
    patient_case_prompt = row["history"]
    question = row["patient_case_prompt"] # replace with ayu patient prompt when relevant

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


def ret_stratified_examples(trainset):
    # Convert dspy.Example objects to dictionary format
    train_data = []
    for example in trainset:
        train_data.append({
            'case_id': example.case_id,
            'case': example.case,
            'question': example.question,
            'diagnosis': example.diagnosis
        })

    # Convert to DataFrame
    df = pd.DataFrame(train_data)

    # Get value counts of diagnoses
    diagnosis_counts = df['diagnosis'].value_counts()
    print(f"Original diagnosis distribution:\n{diagnosis_counts}\n")

    # Calculate how many examples we want per diagnosis (for 50 total examples)
    n_diagnoses = len(diagnosis_counts)
    samples_per_diagnosis = 1

    # Sample equally from each diagnosis
    balanced_samples = []
    for diagnosis in df['diagnosis'].unique():
        diagnosis_data = df[df['diagnosis'] == diagnosis]
        # Take minimum of available samples or desired samples per diagnosis
        n_samples = min(len(diagnosis_data), samples_per_diagnosis)
        sampled = diagnosis_data.sample(n=n_samples, random_state=42)
        balanced_samples.append(sampled)

    # Combine all balanced samples
    balanced_df = pd.concat(balanced_samples).sample(frac=1, random_state=42)  # Shuffle the final dataset
    print(balanced_df.size)
    # Convert back to dspy.Example format
    balanced_trainset = []
    for _, row in balanced_df.iterrows():
        example = dspy.Example(
            case_id=row['case_id'],
            case=row['case'],
            question=row['question'],
            diagnosis=row['diagnosis']
        ).with_inputs("case", "question")
        balanced_trainset.append(example)

    print(f"\nFinal diagnosis distribution:")
    final_dist = balanced_df['diagnosis'].value_counts()
    print(final_dist)
    print(f"\nTotal examples: {len(balanced_trainset)}")
    return balanced_trainset