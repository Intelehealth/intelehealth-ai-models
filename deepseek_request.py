from openai import OpenAI
import json
import re  # Make sure to import the re module
import os

from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)

DEEPSEEK_R1_API_KEY_OPENROUTER = os.getenv("DEEPSEEK_R1_API_KEY_OPENROUTER")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=DEEPSEEK_R1_API_KEY_OPENROUTER,
)


sys_prompt = """System message:

Your input fields are:
1. `case` (str): case with the history of presenting illness, physical exams, and demographics of a patient
2. `question` (str): the patient prompt question

Your output fields are:
1. `reasoning` (str): all the reasoning, thinking you did are filled here.
2. `diagnosis` (str): Top five differential diagnosis for the patient ranked by likelhood if the provided data is sufficient. Else, mark as "NA".
3. `rationale` (str): detailed chain of thought reasoning for each of these top 5 diagnosis predicted. don't include the case and question in this one.
4. `conclusion` (str): Final conclusion on top likely diagnsois considering all rationales
5. `further_questions` (str): Further questions to ask for more clear diagnosis based on provided data. Do not hallucinate on questions.

In adhering to this structure, your objective is:
        You are a doctor with the following patient rural India.
        Here is their case with the history of presenting illness, their physical exams, and demographics.
        What would be the top 5 differential diagnosis for this patient?
        For each diagnosis, include the likelihood score and the rationale for that diagnosis
        For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural india.
        For a low likelihood diagnosis, include lack of fit reasoning under the rationale for that diagnosis.
        Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis.
        Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient.

    Final outout should be in JSON format:
        
    EXAMPLE JSON OUTPUT should follow the following schema:
        
        {
            "type": "json_schema",
            "json_schema": {
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
                                "required": ["disease", "rationale"]
                            },
                            "maxItems": 5
                        },
                        "reasoning" : {
                            "type": "string"
                        },
                        "conclusion" : {
                            "type": "string"
                        },
                        "further_questions" : {
                            "type": "string"
                        }
                    },
                    "required": ["diagnoses", "conclusion", "reasoning", "further_questions"]
                }
            }
        }
"""

case_prompt= """
HISTORY
                    31 year old woman who was previously in good health presents to the clinic with a 4 day history of severe nausea and vomiting. She has been unable to keep down liquids secondary to profound nausea. She also reports general fatigue and weakness over the past week and a 2 day history of swelling in her lower legs, face and arms. She denies any fever or chills, but does report feeling feverish approximately 2 weeks ago in the setting of a severe sore throat with both symptoms resolving after a few days.

                                              She has no cough but does describe mild shortness of breath with exertion over the past week. No abdominal pain, diarrhea or constipation. She has no dysuria() but does note that urine has been small in volume and brownish in colour. She reports no new medications, no changes in diet, no one sick at home, she does work in an elementary school and often gets sick.

                                                                        Past Medical History : depression Allergy: No known drug allergies

                                                                                                  Social History: No smoking, alcohol intake, no drugs.
                                                                                                                            Temp 36.2C BP 155/95 HR 85 RR 16 O2 sat 94% RA


                                                                                                                                                      PHYSICAL EXAMINATION


                                                                                                                                                                                Ill appearing young woman with swollen face. Abdomen is soft, +BS, mildly and diffusely tender to palpation, non distended
                                                                                                                                                                                                          Extremities: Pitting edema in hands and legs, no rashes, no joint abnormalities"""

# #arthalgia
# cp1 = """
# "Gender: Female
 
#   Age: 67 years
 
#   Chief_complaint: ► **Headache** : 
#  • Duration - 5 Days. 
#  • Site - Localized - कानाच्या दोन्ही बाजूंना . 
#  • Severity - Moderate. 
#  • Onset - Acute onset (Patient can recall exact time when it started). 
#  • Character of headache - Throbbing, Stabbing. 
#  • Radiation - पूर्ण शरीरात वेदना जातात . 
#  • Timing - No particular time. 
#  • Exacerbating factors - bending, Exposure to cold. 
#  • Prior treatment sought - None. 
#  ► **Leg, Knee or Hip Pain** : 
#  • Site - Right leg, Hip, Knee, Site of knee pain - Front, Back,
#  Lateral/medial. Swelling - Yes, Calf, Left leg, Hip, Knee, Site of knee pain -
#  Front, Back, Lateral/medial. Swelling - Yes, Calf, Hip. 
#  • Duration - 5 Days. 
#  • Pain characteristics - Sharp shooting. 
#  • Onset - Gradual. 
#  • Progress - Static (Not changed). 
#  • गुढग्यांच्या खाली ज्यास्त वेदना होता
 
#   Physical_examination: **General exams:** 
#  • Eyes: Jaundice-no jaundice seen, [picture taken]. 
#  • Eyes: Pallor-normal pallor, [picture taken]. 
#  • Arm-Pinch skin* - appears slow on pinch test. 
#  • Nail abnormality-nails normal, [picture taken]. 
#  • Nail anemia-Nails are not pale, [picture taken]. 
#  • Ankle-no pedal oedema, [picture taken]. 
#  **Joint:** 
#  • non-tender. 
#  • no deformity around joint. 
#  • full range of movement is seen. 
#  • joint is swollen, [picture taken]. 
#  • pain during movement. 
#  • no redness around joint. 
#  **Back:** 
#  • tenderness observed. 
#  **Head:** 
#  • No injury.
 
#   Patient_medical_history: • Pregnancy status - Not pregnant. 
#  • Allergies - No known allergies. 
#  • Alcohol use - No. 
#  • Smoking history - Patient denied/has no h/o smoking. 
#  • Drug history - No recent medication. 
 
#   Family_history: -
 
#   Vitals:- 
 
#  Sbp: 110.0
 
#   Dbp: 81.0
 
#   Pulse: 85.0
 
#   Temperature: 36.56 'C
 
#   Weight: 42.6 Kg
 
#   Height: 150.0 cm
 
#   RR: 26.0
 
#   SPO2: 100.0
 
#   HB: Null
 
#   Sugar_random: Null
 
#   Blood_group: Null
 
#   Sugar_pp: Null
 
#   Sugar_after_meal: Null"
# """

# cp2 = """
# "Gender: Female
 
#   Age: 30 years
 
#   Chief_complaint: ► **Abdominal Pain** : 
#  • Site - Lower (C) - Hypogastric/Suprapubic. 
#  • Pain radiates to - Middle (R) - Right Lumbar. 
#  • 3 Days. 
#  • Onset - Gradual. 
#  • Timing - Not linked to any particular time of day. 
#  • Character of the pain - Dull, aching. 
#  • Severity - Mild, 1-3. 
#  • Exacerbating Factors - None. 
#  • Relieving Factors - औषधाचे नाव सांगू शकत नाही. 
#  • Menstrual history - Is menstruating - 16, 07/Jun/2024. 
#  • Prior treatment sought - None. 
#  • Additional information - मासिक पाळी दरम्यान ओटीपोटाला वेदना होतात . 
#  ► **Associated symptoms** : 
#  • Patient denies - 
#  Nausea, Vomiting, Anorexia, Diarrhea, Constipation, Fever, Abdominal
#  distention/Bloating, Belching/Burping, Passing gas, Color change in stool
#  [describe], Blood in stool, change in frequency of urination, Color change in
#  urine, Hiccups, Restlessness,
 
#   Physical_examination: **General exams:** 
#  • Eyes: Jaundice-no jaundice seen, [picture taken]. 
#  • Eyes: Pallor-normal pallor, [picture taken]. 
#  • Arm-Pinch skin* - pinch test normal. 
#  • Nail abnormality-nails normal, [picture taken]. 
#  • Nail anemia-Nails are not pale, [picture taken]. 
#  • Ankle-no pedal oedema, [picture taken]. 
#  **Abdomen:** 
#  • no distension. 
#  • no scarring. 
#  • no tenderness. 
#  • Lumps-no lumps.
 
#   Patient_medical_history: • Pregnancy status - Not pregnant. 
#  • Allergies - No known allergies. 
#  • Alcohol use - No. 
#  • Smoking history - Patient denied/has no h/o smoking. 
#  • Drug history - No recent medication. 
 
#   Family_history: -
 
#   Vitals:- 
 
#  Sbp: 122.0
 
#   Dbp: 92.0
 
#   Pulse: 85.0
 
#   Temperature: 36.28 'C
 
#   Weight: 67.15 Kg
 
#   Height: 151.0 cm
 
#   RR: 21.0
 
#   SPO2: 97.0
 
#   HB: Null
 
#   Sugar_random: Null
 
#   Blood_group: Null
 
#   Sugar_pp: Null
 
#   Sugar_after_meal: Null"
# """
import pandas as pd
import time

# Read the CSV file
df = pd.read_csv("./data/NAS_UnseenData_Combined.csv")

# Create empty DataFrame to store results
results_df = pd.DataFrame(columns=[
    "visit_id", 
    "patient_case",
    "gt_diagnosis",
    "reasoning",
    "diagnosis",
    "conclusion",
    "further_questions"
])


single_diagnosis_diff_schema =  {
    "type": "json_schema",
    "json_schema": {
        "name": "single_diagnosis_diff_schema",
        "schema": {
            "type": "object",
            "properties": {
                # "primary_diagnosis": {
                #     "type": "string",
                #         "properties": {
                #             "disease": {"type": "string"},
                #             "rationale": {"type": "string"}
                #         },
                #     "minItems": 1
                # },
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
                    "maxItems": 5
                },
                "reasoning" : {
                    "type": "string"
                },
                "conclusion" : {
                    "type": "string"
                },
                "further_questions" : {
                    "type": "string"
                }
            },
            "required": ["diagnoses", "conclusion", "reasoning", "further_questions"]
        }
    }
}

# Process each row
for index, row in df.iterrows():
    # if index < 34:
    #     continue
    print(f"Processing row {index}")
    
    # Construct case text
    case = f"""
    Clinical History: {row['Clinical_notes']}
    """
    
    # Store response data
    resp = {
        "visit_id": row["Visit_id"],
        "patient_case": case,
        "gt_diagnosis": row["Diagnosis"],
        "reasoning": "",
        "diagnosis": "",
        "rationale": "",
        "conclusion": "",
        "further_questions": ""
    }

    messages = [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": case

            }
        ]
    
    try:
        # Make API request
        print(messages)

        response = client.chat.completions.create(
            #   model="deepseek/deepseek-chat:free",
            model="deepseek/deepseek-r1:free",
            messages=messages,
            max_tokens=18000,  # Increase max tokens for longer responses
            temperature=1.0, 
            # response_format=single_diagnosis_diff_schema
            response_format= {
                "type": "json_object"
            }
        )

        # Print the raw response for debugging
        print("########################################################")
        print("Raw response:", response)

        # Extract JSON if wrapped in ```json
        # Check if content is None
        if response.choices[0].message.content is None or response.choices[0].message.content == "":
            print("Warning: Received null content in response")
            resp["reasoning"] = ""
            resp["diagnosis"] = "" 
            resp["rationale"] = ""
            resp["conclusion"] = ""
            resp["further_questions"] = ""
            continue
        content = response.choices[0].message.content
        json_match = re.search(r'```json(.*?)```', content, re.DOTALL)  # Use regex to find JSON block

        if json_match:
            json_content = json_match.group(1).strip()  # Extract the JSON content
            output = json.loads(json_content)  # Parse the JSON content
        else:
            output = json.loads(content)  # Fallback if no JSON block found

        print("Parsed output:", output)
        print("########################################################")
        # sys.exit(0)
        # Parse individual fields from output
        resp["reasoning"] = output.get("reasoning", "")
        resp["diagnosis"] = output.get("diagnosis", "")
        resp["rationale"] = output.get("rationale", "") 
        resp["conclusion"] = output.get("conclusion", "")
        resp["further_questions"] = output.get("further_questions", "")
        #{'diagnoses': [{'disease': 'Acute Bronchitis (Bacterial/Viral)', 'rationale': 'High likelihood. Clinical features include acute productive cough (yellow/green sputum), wheezing, and cold weather exacerbation. Smoking history (if valid) increases risk. Common in rural India due to biomass fuel exposure and limited healthcare access.'}, {'disease': 'Gastroesophageal Reflux Disease (GERD)', 'rationale': 'Moderate likelihood. Nocturnal burning throat, hoarseness, and postnasal drip align with GERD-induced cough. Lack of typical GERD symptoms like heartburn reduces certainty but does not exclude it.'}, {'disease': 'Chronic Sinusitis with Postnasal Drip', 'rationale': 'Moderate likelihood. Persistent cough, postnasal drip, and bilateral ear pain suggest eustachian tube dysfunction secondary to sinusitis. Common in untreated upper respiratory infections in rural settings.'}, {'disease': 'Fungal Otitis Externa', 'rationale': 'Moderate likelihood. Bilateral ear itching without discharge/swimming history fits fungal etiology. Prevalent in rural India due to humid climates and home-remedy practices (e.g., oil instillation).'}, {'disease': 'Allergic Rhinitis', 'rationale': 'Low likelihood. Postnasal drip and ear itching could indicate allergies, but lack of seasonal or allergen-specific history reduces probability. Requires confirmation via allergen exposure history.'}], 'reasoning': 'The patient's cough (wet, yellow/green sputum) and wheezing suggest airway inflammation. Conflicting smoking history complicates COPD assessment. Ear itching without discharge favors fungal otitis. No fever or hypoxia argues against pneumonia. GERD and sinusitis are plausible due to nocturnal symptoms and postnasal drip.', 'conclusion': 'Acute Bronchitis is the most likely diagnosis, exacerbated by environmental/cold triggers. GERD and fungal otitis externa require concurrent management.', 'further_questions': '1. Clarify smoking/biomass fuel exposure. 2. Ask about heartburn/regurgitation (GERD). 3. Inquire about ear-cleaning methods. 4. Assess allergy triggers. 5. Confirm sputum progression/fever history.'}
        # Store results

        
    except Exception as e:
        print(f"Error processing row {index}: {str(e)}")
    
    # Append to results DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([resp])], ignore_index=True)
    
    # Save after each row
    results_df.to_csv('deepseek_nas_unseen_v2_deepseek_r1_results_1_onwards_1.csv', mode='a', header=False, index=False)
    
    # Small delay between requests
    time.sleep(1)

print("Processing complete")


