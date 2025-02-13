from past_patients_visit import get_patient_data


from google import genai
import os, sys

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

patient_id = 9522243921
patient_data = get_patient_data(patient_id)
print(patient_data[1]["Age"])
print(patient_data[1]["Gender"])
print(patient_data[1]["Chief_complaint"])
print(patient_data[2]["Chief_complaint"])


prompt = ("""
Using the patient_data provided, generate a story for this patient similar to the given template below.
{template}

Patient Data: {patient_data}.
          
Also, ensure follow up date is mentioned for the patient in the story.

Please do not include any other information outside this patient data provided and do not hallucinate.
""").format(
    template=template,
    patient_data=patient_data
)




response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)

print(response.text)