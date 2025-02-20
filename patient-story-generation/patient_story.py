from past_patients_visit import get_patient_data


from google import genai
import os, sys
import pandas as pd


from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)
# Create a client

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
 
# Define the model you are going to use
model_id =  "gemini-2.0-flash" # or "gemini-2.0-flash-lite-preview-02-05"  , "gemini-2.0-pro-exp-02-05"
template = "This is ABC 👩🏽, a 27-year-old from Nasik, Maharashtra🇮🇳, living at 444, Kathe Lane. She was born on August 6, 1997🎂, and can be reached at 9876543210. Muskan first visited us a few months ago with complaints of recurring headaches and fatigue, which we traced to stress and mild anemia. On her second visit, she reported dizziness and nausea, so we adjusted her treatment with a nutritional plan and supplements."

# Read the CSV file
csv_file_path = '../data/Unified_Patient_Data.csv'
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: CSV file at {csv_file_path} is empty.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while reading the CSV: {e}")
    sys.exit(1)

# Get unique patient IDs
patient_ids = df['Patient_id'].unique()

# Initialize a list to store patient stories
patient_stories = []

# Process each patient
for patient_id in patient_ids:
    patient_data = get_patient_data(patient_id)

    if patient_data:  # Only process if we got valid data
        prompt = ("""
        Using the patient_data provided, generate a story for this patient similar to the given template below.
        {template}

        Patient Data: {patient_data}.

        Also, ensure follow up date is mentioned for the patient in the story. Also, if there are contact numbers and details, mention them in the story. Ensure all demographic details of the patient and nature of work if any are mentioned in the story.

        Please do not include any other information outside this patient data provided and do not hallucinate.
        """).format(
            template=template,
            patient_data=patient_data
        )

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            print(f"\nStory for Patient ID {patient_id}:")
            print(response.text)
            print("-" * 80)  # Separator between stories
            # Save the response to the list
            patient_stories.append({"Patient ID": patient_id, "Story": response.text})
        except Exception as e:
            print(f"Error generating story for patient {patient_id}: {e}")
            continue

# Save all patient stories to a CSV file
output_csv_path = '../data/patient_stories_trial3.csv'
pd.DataFrame(patient_stories).to_csv(output_csv_path, index=False)
print(f"Patient stories saved to {output_csv_path}")

# patient_id = 952277936
# patient_data = get_patient_data(patient_id)
# print(patient_data)