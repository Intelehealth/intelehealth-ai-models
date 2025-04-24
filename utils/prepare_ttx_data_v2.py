import dspy
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("./data/NAS_v2_UnseenData_Combined.csv")
# Handle potential missing values
df.fillna('', inplace=True)

# Create dspy examples
ttx_examples = []

for index, row in df.iterrows():
    # Extract data from relevant columns with explicit empty value handling
    visit_id = str(row["Visit_id"]) if pd.notna(row["Visit_id"]) and row["Visit_id"] != "" else ""
    diagnosis = str(row["Diagnosis"]) if pd.notna(row["Diagnosis"]) and row["Diagnosis"] != "" else ""
    case_notes = str(row["Clinical_notes"]) if pd.notna(row["Clinical_notes"]) and row["Clinical_notes"] != "" else ""
    
    # Additional columns for v2 with explicit empty value handling
    medicines = str(row["Medicines"]) if pd.notna(row["Medicines"]) and row["Medicines"] != "" else ""
    strength = str(row["Strength"]) if pd.notna(row["Strength"]) and row["Strength"] != "" else ""
    dosage = str(row["Dosage"]) if pd.notna(row["Dosage"]) and row["Dosage"] != "" else ""
    medical_test = str(row["Medical_test"]) if pd.notna(row["Medical_test"]) and row["Medical_test"] != "" else ""
    medical_advice = str(row["Medical_advice"]) if pd.notna(row["Medical_advice"]) and row["Medical_advice"] != "" else ""
    referral_advice = str(row["Referral_advice"]) if pd.notna(row["Referral_advice"]) and row["Referral_advice"] != "" else ""
    medications = str(row["Medications"]) if pd.notna(row["Medications"]) and row["Medications"] != "" else ""

    # Create an example with 'case' and 'diagnosis' as inputs
    # And the additional fields as outputs
    example = dspy.Example(
        case_id=visit_id,
        case=case_notes,
        diagnosis=diagnosis,
        medications_gt=medications,
        medicines_gt=medicines,
        strength_gt=strength,
        dosage_gt=dosage,
        medical_test_gt=medical_test,
        medical_advice_gt=medical_advice,
        referral_advice_gt=referral_advice
    ).with_inputs("case", "diagnosis", "medicines", "strength", "dosage", "medical_test", "medical_advice")  # Define inputs

    ttx_examples.append(example)

print(f"Created {len(ttx_examples)} examples.")

# Optional: Print the first example to verify
if ttx_examples:
    print("\nFirst example:")
    print(ttx_examples[0])

def ret_ttx_v2_examples():
    """Returns the list of prepared dspy examples."""
    return ttx_examples

# Example usage:
if __name__ == "__main__":
    examples = ret_ttx_v2_examples()
    if examples:
        print("\nFirst example from function call:")
        print(examples[0])
        print(f"\nRetrieved {len(examples)} examples via function call.")
    else:
        print("\nNo examples were created or retrieved.") 