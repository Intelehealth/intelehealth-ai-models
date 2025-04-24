import dspy
import pandas as pd

# Load the dataset
# df = pd.read_csv("./data/DDx_database-190-cases-data-cleaned-ayu.csv")
df = pd.read_csv("./data/DDx_database-unseen-data-NAS-without-corrections.csv")
# Handle potential missing values
df.fillna('', inplace=True)


# Create dspy examples
ttx_examples = []

for index, row in df.iterrows():
    # Use correct column names from the new CSV
    case_id = row["OpenMRS_id"] 
    diagnosis = row["Diagnosis"]
    case_notes = row["Clinical_notes"] # Map Clinical_notes to case
    medications = row["Medications"]
    medical_advice = row["Medical_advice"]

    # Create an example with 'case' and 'diagnosis' as inputs
    # And 'medications', 'medical_advice' as outputs (implicitly)
    example = dspy.Example(
        case_id=str(case_id), # Ensure case_id is string if needed
        case=str(case_notes),
        diagnosis=str(diagnosis),
        medications_gt=str(medications),
        medical_advice=str(medical_advice)
    ).with_inputs("case", "diagnosis") # Define inputs

    ttx_examples.append(example)

print(f"Created {len(ttx_examples)} examples.")

# Optional: Print the first example to verify
if ttx_examples:
    print("\nFirst example:")
    print(ttx_examples[0])

def ret_ttx_examples():
    """Returns the list of prepared dspy examples."""
    return ttx_examples

# Example usage:
if __name__ == "__main__":
    examples = ret_ttx_examples()
    if examples:
        print("\nFirst example from function call:")
        print(examples[0])
        print(f"\nRetrieved {len(examples)} examples via function call.")
    else:
        print("\nNo examples were created or retrieved.") 