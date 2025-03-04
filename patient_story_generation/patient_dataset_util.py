import pandas as pd
import os

def merge_patient_data():
    # Define file paths
    patient_file = os.path.join('../data', 'Patient_Listing_Report_2025-02-17_12-44-21.csv')
    visit_file = os.path.join('../data', 'Visit_Listing_with_CN_2025-02-17_14-06-03.csv')
    
    # Read the CSV files
    patient_df = pd.read_csv(patient_file)
    visit_df = pd.read_csv(visit_file)
    
    # Merge the dataframes on 'patient_id'
    merged_df = pd.merge(patient_df, visit_df, on='Patient_id', how='outer')
    
    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(os.path.join('../data', 'Unified_Patient_Data.csv'), index=False)

# Call the function to execute the merge
merge_patient_data()
