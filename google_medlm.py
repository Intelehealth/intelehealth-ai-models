import requests
import os
from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)


MEDLM_API_BASE = os.getenv("MEDLM_API_BASE")
url = MEDLM_API_BASE


dspy_question_prompt = "You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics.\nWhat would be the top 5 differential diagnosis for this patient?\nFor each diagnosis, include the SNOMED CT code, likelihood score and the rationale for that diagnosis.\nFor high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceding infection, and relevance to rural India.\nFor a low likelihood diagnosis, include lack of fit reasoning under the rationale for that diagnosis.\nPlease rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis.\nPlease remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient. The final output should have sections like this ###Diagnostics ###Conclusion ###\n\n"
patient_history = "HISTORY CC/ID: 52-year-old man with a painful, swollen, hot right knee. Mr… was awakened from sleep last night by exquisite pain in his right knee, which had become swollen and warm. He had felt well during the day preceding the onset of pain and had attended a crab-fest that afternoon. He denies trauma to the joint, penetrating injuries, injections, or extramarital sexual contact. He lives in the city and does not enjoy hiking or camping, although he likes to fish. He cannot recall any tick exposures. He reports subjective fevers and sweats, but hasn't taken his temperature; the ROS is otherwise negative. Although no other joints currently hurt, he recalls an intensely painful left big toe several years ago that got better with 'aspirin.' Past Medical History: Hypertension; hernia repair. Meds: HCTZ, 25 mg. All: NKDA. Family History/Social History: Parents both deceased; father had 'arthritis.' Married, with two adult children. PHYSICAL EXAMINATION VS: Temp 39C. BP 150/90. HR 100, RR 16. 02 sat RA. Gen: large man, lying on gurney, nontoxic but uncomfortable. HEENT: unremarkable. Neck: supple, no thyromegaly or adenopathy. Lungs: clear. Q': RRR, normal S1, S2, 216 HSM at apex. Abdomen: soft, NT/ND, +BS. Ext: no track marks; right knee swollen, warm, tender to touch; no swelling or lymphangitis; 2+ peripheral pulses. Skin: no rashes, no necrotic-appearing lesions. Genitourinary system: no urethral discharge. Neuro: nonfocal. LABORATORY WBC 13; Hct, Plt Count normal; Cr 1.2; serum urate normal; INR 1.0."

payload = {
	  "instances": [
		      {
                  "content": dspy_question_prompt + "\n\n" + patient_history

				}
					],
					"parameters": {
                            "temperature": 0.0,
                            "maxOutputTokens": 7500,
                            "topK": 5,
                            "topP": 0.95
				    }
}

import subprocess

def get_access_token():
    try:
        # Run the gcloud command and capture the output
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True, 
                                check=True)
        
        # Get the access token from the output
        access_token = result.stdout.strip()
        return access_token

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        return None

# Usage
token = get_access_token()
if token:
    print("Access Token:", token)
else:
    print("Failed to retrieve access token.")

token_val = get_access_token()

headers = {
  'Authorization': 'Bearer ' + token_val,
  'Content-Type': 'application/json; charset=utf-8'
}

response = requests.request("POST", url, headers=headers, json=payload)

print(response.text)
