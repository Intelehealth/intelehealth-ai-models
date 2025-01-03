import pandas as pd
import numpy as np

df = pd.read_csv("data/Unseen_data_NAS.csv")

print(df.columns)

new_df = df[df.columns]

print(new_df)

training_examples = []
import sys
for index,row in new_df.iterrows():
    print("-------------------")
    print(index)
    print(row)
    OpenMRS_id = row["OpenMRS_id"]
    Gender = row["Gender"]
    Age = row["Age"]
    Location_village = row["Location_village"]
    Clinical_notes = row["Clinical_notes"]
    Diagnosis = row["Diagnosis"]
    Corrections_in_Diagnosis = row["Corrections in Diagnosis"]
    Medications = row["Medications"]
    Medical_test = row["Medical_test"]
    Medical_advice = row["Medical_advice"]
    Additional_comments = row["Additional_comments"]

    # diag = ""

    # if Corrections_in_Diagnosis != "" or Corrections_in_Diagnosis !="NaN" or np.isnan(Corrections_in_Diagnosis) != True:
    #     diag = Corrections_in_Diagnosis
    # else:
    #     diag = Diagnosis

    training_examples.append({
        "OpenMRS_id": row["OpenMRS_id"],
        "Gender": row["Gender"],
        "Age": row["Age"],
        "Location_village": row["Location_village"],
        "Clinical_notes": row["Clinical_notes"],
        "Diagnosis": Diagnosis,
        "Corrections_in_Diagnosis": row["Corrections in Diagnosis"],
        "Medications" :row["Medications"],
        "Medical_test": row["Medical_test"],
        "Medical_advice": row["Medical_advice"],
        "Additional_comments" : row["Additional_comments"]
    })
    
    

training_df = pd.DataFrame(training_examples)
print(training_df)
training_df.to_csv("data/DDx_database-unseen-data-NAS-without-corrections.csv", index=False)
