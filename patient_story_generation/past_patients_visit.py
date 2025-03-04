import pandas as pd

csv_file_path = '../data/Unified_Patient_Data.csv'  # Replace with the actual path to your CSV file

def get_patients_with_visits(csv_path, min_visits=1, max_visits=5, num_patients=455):
    """
    Finds patients who have a specified number of visits within a range.

    Args:
        csv_path (str): The path to the CSV file.
        min_visits (int): The minimum number of visits a patient must have.
        max_visits (int): The maximum number of visits a patient must have.
        num_patients (int): The number of patients to return.

    Returns:
        list: A list of patient IDs who have the specified number of visits.  Returns an empty list if no patients match criteria, or if the CSV is unreadable.
    """
    try:
        df = pd.read_csv(csv_path)
        # print(df.head())
        print(df.columns)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_path} is empty.")
        return []
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV at {csv_path}.  Check file format.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return []

    # Check if 'Patient_id' column exists
    if 'Patient_id' not in df.columns:
        print("Error: 'Patient_id' column not found in the CSV file.")
        return []

    patient_visit_counts = df['Patient_id'].value_counts()

    # Filter for patients with the desired number of visits
    eligible_patients = patient_visit_counts[
        (patient_visit_counts >= min_visits) & (patient_visit_counts <= max_visits)
    ].index.tolist()

    # Return the first num_patients
    return eligible_patients[:num_patients]

def get_patient_data(patient_id):
    global csv_file_path
    """
    Retrieves all data rows for a specific patient from the CSV file.

    Args:
        csv_path (str): The path to the CSV file.
        patient_id: The ID of the patient to retrieve data for.

    Returns:
        dict: A dictionary where keys are visit numbers (1, 2, 3...) and values are
              dictionaries containing all data for that visit. Returns empty dict if
              patient not found or if there's an error reading the CSV.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}

    if 'Patient_id' not in df.columns:
        print("Error: 'Patient_id' column not found in the CSV file.")
        return {}

    # Filter rows for the specific patient
    patient_data = df[df['Patient_id'] == patient_id]
    
    if patient_data.empty:
        print(f"No data found for patient ID: {patient_id}")
        return {}

    # Convert to dictionary with visit number as key
    result = {}
    for index, (_, row) in enumerate(patient_data.iterrows(), 1):
        result[index] = row.to_dict()

    return result

# # Example usage:
# patients = get_patients_with_visits(csv_file_path)

# if patients:
#     print(f"Found patients with {3}-{4} visits:")
#     for patient_id in patients:
#         print(f"\nPatient ID: {patient_id}")
#         patient_data = get_patient_data(patient_id)

#         if len(patient_data) >= 2:
#             print(f"Number of visits: {len(patient_data)}")
#             print(patient_data[1].keys())
        
# else:
#     print("No patients found matching the criteria, or an error occurred.")