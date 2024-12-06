import pandas as pd

class PatientData:
    def __init__(self, csv_file_1, csv_file_2):
        # Read the CSV files once during initialization
        self.data_1 = pd.read_csv(csv_file_1)
        self.data_2 = pd.read_csv(csv_file_2)
        # Merge the dataframes on 'patient_id' and store it
        self.merged_data = pd.merge(self.data_1, self.data_2, on='Patient_id')

    def get_merged_data(self):
        # Return the merged data
        return self.merged_data

    def count_duplicate_patient_ids(self):
        # Count duplicates in each CSV
        duplicates_1 = self.data_1['Patient_id'].duplicated().sum()
        duplicates_2 = self.data_2['Patient_id'].duplicated().sum()
        return duplicates_1, duplicates_2

    def read_common_data_to_dict(self):
        # Create a common dictionary with Patient_id as the key
        common_dict = {}
        
        # Populate the dictionary with data from the first CSV
        for _, row in self.data_1.iterrows():
            common_dict[row['Patient_id']] = row.to_dict()
        
        # Update the dictionary with data from the second CSV
        for _, row in self.data_2.iterrows():
            patient_id = row['Patient_id']
            if patient_id in common_dict:
                common_dict[patient_id].update(row.to_dict())
            else:
                common_dict[patient_id] = row.to_dict()
        
        return common_dict


import time
start_time = time.time()
pdata = PatientData("../data/Patient_Story_RawData1.csv", "../data/Patient_Story_RawData2.csv")
end_time = time.time()

print("time taken to merge: ", end_time-start_time)
print(pdata.get_merged_data().shape)


duplicates = pdata.count_duplicate_patient_ids()
print("Duplicates in CSV 1:", duplicates[0])
print("Duplicates in CSV 2:", duplicates[1])

common_data_dict = pdata.read_common_data_to_dict()
print("Common Data Dictionary:", common_data_dict[952268785])