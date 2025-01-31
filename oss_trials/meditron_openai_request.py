from openai import OpenAI
import json

# Initialize OpenAI client that points to the local LM Studio server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",

)

user_prompt = "what are the top 5 differential diagnosis for indian doctor in rural setting for the case given by user. Order by likelhiood of each diagnosis from top to bottom."
user_multiple_prompt = """Here is their case with the history of presenting illness, symptoms, their physical exams, and demographics.
        What would be the top primary diagnosis for this patient.
        Also include other 4 differential diagnosis for the patient.
        Please rank the differential diagnoses based on the likelihood from top to bottom.
        If the data provided for the patient is not is not sufficient to make diagnosis, mark both primary diagnosis as "NA" and ask further questions to be asked to make diagnosis more clear.
        If there was a primary diagnosis found, mark further questions as "NA"
        Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."""
# Define the conversation with the LLM
messages = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": user_prompt + """CASE: Gender: Female Age: 31 years Chief_complaint: ► **Leg, Knee or Hip Pain** : • Site - Right leg, Hip, Thigh, Knee, Site of knee pain - Front, Back, Lateral/medial. Swelling - No, Calf, Left leg, Hip, Thigh, Knee, Site of knee pain - Front, Back, Lateral/medial. Swelling - No, Calf, Hip. • Duration - 2 महिने. • Pain characteristics - Sharp shooting, हातपाय दुखतात . • Onset - Gradual. • Progress - Wax and wane. • पाठीला वेदना जातात . • Aggravating factors - Symptom aggravated by motion, Associated with motion - All planes of motion, Worsened by rest (relieved by activity). • H/o specific illness - • Patient reports - None • Trauma/surgery history - No recent h/o trauma/surgery. • Injection drug use - No h/o injection / drug use. • Prior treatment sought - None. ► **Associated symptoms** : • Patient reports - Stiffness - Timing - Morning - lasting <= 60 minutes. • Patient denies Physical_examination: **General exams:** • Eyes: Jaundice-no jaundice seen, [picture taken]. • Eyes: Pallor-normal pallor, [picture taken]. • Arm-Pinch skin* - pinch test normal. • Nail abnormality-nails normal, [picture taken]. • Nail anemia-Nails are not pale, [picture taken]. • Ankle-no pedal oedema, [picture taken]. **Joint:** • tenderness seen, [picture taken]. • no deformity around joint. • full range of movement is seen. • joint is not swollen. • pain during movement. • no redness around joint. **Back:** • tenderness observed. Patient_medical_history: • Pregnancy status - Not pregnant. • Allergies - No known allergies. • Alcohol use - No. • Smoking history - Patient denied/has no h/o smoking. • Drug history - No recent medication. Family_history: - Vitals:- Sbp: 110.0 Dbp: 85.0 Pulse: 81.0 Temperature: 36.28 'C Weight: 42.15 Kg Height: 147.0 cm RR: 23.0 SPO2: 98.0 HB: Null Sugar_random: Null Blood_group: Null Sugar_pp: Null Sugar_after_meal: Null"""

            }
        ]

# Define the expected response structure
response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "diagnosis_with_rationale",
        "schema": {
            "type": "object",
            "properties": {
                "diagnoses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "disease": {"type": "string"},
                            "rationale": {"type": "string"}
                        },
                         "required": ["diagnosis_id", "disease", "rationale"]
                    },
                    "minItems": 1
                }
            },
            "required": ["diagnoses"]
        }
    }
}

multiple_diagnosis_schema =  {
    "type": "json_schema",
    "json_schema": {
        "name": "mulitple_diagnosis_with_rationale",
        "schema": {
            "type": "object",
            "properties": {
                "primary_diagnosis": {
                    "type": "string",
                        "properties": {
                            "disease": {"type": "string"},
                            "rationale": {"type": "string"}
                        },
                    "minItems": 1
                },
                "diagnoses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "disease": {"type": "string"},
                            "rationale": {"type": "string"}
                        },
                         "required": ["disease", "rationale"]
                    },
                    "maxItems": 3
                }
            },
            "required": ["diagnoses", "primary_diagnosis"]
        }
    }
}

# Get response from AI
response = client.chat.completions.create(
    model="TheBloke/meditron-70B-GGUF",
    messages=messages,
    response_format=response_schema,
    max_tokens=-1,
    temperature=0.2 # Add temperature here

    )

# Parse and display the results
results = json.loads(response.choices[0].message.content)
print(json.dumps(results, indent=2))
