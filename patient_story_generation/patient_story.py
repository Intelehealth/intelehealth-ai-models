from past_patients_visit import get_patient_data


from google import genai
import os, sys
import pandas as pd
from tqdm import tqdm


from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)
# Create a client

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
 
# Define the model you are going to use
model_id =  "gemini-2.0-flash" # or "gemini-2.0-flash-lite-preview-02-05"  , "gemini-2.0-pro-exp-02-05"
template = "This is ABC 👩🏽, a 27-year-old from Nasik, Maharashtra🇮🇳. Muskan first visited us a few months ago with complaints of recurring headaches and fatigue, which we traced to stress and mild anemia. On her second visit, she reported dizziness and nausea, so we adjusted her treatment with a nutritional plan and supplements."

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
for patient_id in tqdm(patient_ids, desc="Processing patients"):
    patient_data = get_patient_data(patient_id)

    if patient_data:  # Only process if we got valid data
        prompt = ("""
        Using the patient_data provided, generate a story for this patient in an empathetic tone and manner highlighting the patient's converstions with the healthworker and the challenges they faced for the doctor in a userful and helpful manner.

        Start the story similar to this template:
        {template}

        Patient Data: {patient_data}.

        Patient story should include the following:
        - Current consultation reason.
        - Past visit details.
        - Risk details from all the previous visits.
        - 2,3 points of human touch: name, age, occupation, family members

        Also, ensure follow up date is mentioned for the patient in the story.
        If any medications were prescribed in a visit, do mention them in the story.

        Please do not include any other information outside this patient data provided and do not hallucinate.
        Do not inlcude sentences in any other language other than English. 
        Do not include lines at the start of the story like - Here is a story for this patient, generated from the provided data in an empathetic tone.
        """).format(
            template=template,
            patient_data=patient_data
        )

        try:
            response = client.models.generate_content(
                # model="gemini-2.0-flash",
                model="gemini-2.5-flash-preview-04-17",
                contents=prompt,
            )
            story_text = response.text
            line_count = len(story_text.splitlines())
            word_count = len(story_text.split())

            print(f"\nStory for Patient ID {patient_id}:")
            print(story_text)
            print(f"Line Count: {line_count}")
            print(f"Word Count: {word_count}")
            print("-" * 80)  # Separator between stories
            # Save the response to the list
            patient_stories.append({
                "Patient ID": patient_id,
                "Story": story_text,
                "Line Count": line_count,
                "Word Count": word_count,
                "Prompt": prompt,
                "Patient Data": patient_data
            })
        except Exception as e:
            print(f"Error generating story for patient {patient_id}: {e}")
            continue

# Save all patient stories to a CSV file
output_csv_path = '../data/patient_story_generation_evals_13.csv'
pd.DataFrame(patient_stories).to_csv(output_csv_path, index=False)
print(f"Patient stories saved to {output_csv_path}")

# patient_id = 952277936
# patient_data = get_patient_data(patient_id)
# print(patient_data)