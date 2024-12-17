from google import genai


import os
from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)


MEDLM_PROJECT_ID = os.getenv("MEDLM_PROJECT_ID")

client = genai.Client(
    vertexai=True, project=MEDLM_PROJECT_ID, location='us-central1'
)


response = client.models.generate_content(
    model='medlm-medium', contents='What is ring worm disease?'
)
print(response.text)