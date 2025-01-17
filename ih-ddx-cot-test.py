import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge, load_gemini_lm
import os
import random
from dotenv import load_dotenv
import pandas as pd

from modules.DDxQnsModule import DDxQnModule
load_dotenv(
    "ops/.env"
)
import sys

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# open ai
# lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=1.0)
# dspy.configure(lm=lm, top_k=5)

# gemini 1.5 pro
gemini = dspy.Google("models/gemini-1.5-pro", api_key=GEMINI_API_KEY,)
dspy.settings.configure(lm=gemini, max_tokens=10000, top_k=5)

cot = DDxQnModule()
#cot.load("outputs/" + "07_01_2025_ddx_open_ai_gpt-4o_cot_trial_patient_cleaned_data_llm_judge.json")
cot.load("outputs/" + "Ddx_open_ai_gemini_pro_medpalm_num_trials_10_top_k5.json")

# Test cases
patient_case = """
"Male, 29 years
►Skin disorder:
• Type of the skin lesion - Subcutaneous nodules.
• Site - forehead, behind left ear and near left eye. .
• No. of lesions - Multiple lesions.
• Duration - 7 Days.
• Progression - Transient.
• Prior treatment sought - He tried moisturizer and ice on the lesions. .
►Associated symptoms:
• Patient denies -
Fever

c. = Associated with, s. = Not associated with, h/o = History of
 Family History

 Past Medical History

• He had hit on his forehead one week prior. .

 Vitals


Temp:        98.80 F        Height:        0 cm        Weight:        kg        BMI:                SP02:        99 %        BP:        130/70        HR:        84        RR:        20

 On Examination

General exams:
• Eyes: Jaundice-Don't know.
• Eyes: Pallor-Don't know.
• Arm-Pinch skin* - Don't know.
• Nail abnormality-Don't know.
• Nail anemia-Don't know.
• Ankle oedema-Don't know.
Any Location:
• Skin Rash:-no rash.
Physical Examination

No Physical Exam Images available! 
Additional Documents
"""

# NAS patient case without matching diagnosis; GT is Arthtralgia:Primary & Provisional
patient_case_without_matching_diagnosis = """
"Gender: Female
 
  Age: 28 years
 
  Chief_complaint: ► **Leg, Knee or Hip Pain** : 
 • Site - Left leg - Buttock. 
 • Duration - 3 Days. 
 • Pain characteristics - Dull aching. 
 • Onset - Sudden (minutes to hours). 
 • Progress - Wax and wane. 
 • वेदना पूर्ण शरीरात जातात . 
 • Aggravating factors - Symptom aggravated by motion - Associated with motion
 - Climbing stairs. 
 • H/o specific illness - 
 • Patient reports - 
 None 
 • Trauma/surgery history - No recent h/o trauma/surgery. 
 • Injection drug use - No h/o injection / drug use. 
 • Prior treatment sought - None. 
 ► **Associated symptoms** : 
 • Patient reports - 
 Night pain 
 • Patient denies - 
 Fever, Difficulty breathing, Swelling in the painful area, Redness in the
 painful area, Affected leg feels shorter, Stiffness, Unable to walk, Unable to
 bear weight, Locking in a specific position, Groin pain 
 
  Physical_examination: **General exams:** 
 • Eyes: Jaundice-no jaundice seen, [picture taken]. 
 • Eyes: Pallor-normal pallor, [picture taken]. 
 • Arm-Pinch skin* - pinch test normal. 
 • Nail abnormality-nails normal, [picture taken]. 
 • Nail anemia-Nails are not pale, [picture taken]. 
 • Ankle-no pedal oedema. 
 **Joint:** 
 • tenderness seen. 
 • no deformity around joint. 
 • full range of movement is seen. 
 • joint is not swollen. 
 • no pain during movement. 
 • no redness around joint. 
 **Back:** 
 • tenderness observed.
 
  Patient_medical_history: • Pregnancy status - Not pregnant. 
 • Allergies - No known allergies. 
 • Alcohol use - No. 
 • Smoking history - Patient denied/has no h/o smoking. 
 • Drug history - No recent medication. 
 
  Family_history: -
 
  Vitals:- 
 
 Sbp: 125.0
 
  Dbp: 92.0
 
  Pulse: 88.0
 
  Temperature: 36.94 'C
 
  Weight: 39.65 Kg
 
  Height: 145.0 cm
 
  RR: 21.0
 
  SPO2: 99.0
 
  HB: Null
 
  Sugar_random: Null
 
  Blood_group: Null
 
  Sugar_pp: Null
 
  Sugar_after_meal: Null"
  """

patient_case_no_data="""
im a 45 year old female.

I have pain in stomatch

"""

patient_case_vague = """
"Gender: Female
 
  Age: 50 years
 
  Chief_complaint: ► **Fever** : 
 • Duration - 3 Days. 
 • Nature of fever - Irregular (comes & goes). 
 • Timing - Evening, Night. 
 • Severity - Low. 
 • H/o specific illness - 
 • Patient reports - 
 None 
 • H/o of specific epidemic in community - None. 
 • Prior treatment sought - None. 
 ► **Headache** : 
 • Duration - 3 Days. 
 • Site - Diffuse. 
 • Severity - Mild. 
 • Onset - Acute onset (Patient can recall exact time when it started). 
 • Character of headache - Throbbing. 
 • Radiation - pain does not radiate. 
 • Timing - No particular time. 
 • Exacerbating factors - bending, Standing up from lying position, Exposure to
 cold. 
 • Prior treatment sought - None. 
 ► **Leg, Knee or Hip Pain** : 
 • Site - Right leg, Hip, Buttock, Thigh, Knee, Site of knee pain - Front,
 Back, Lateral/medial, Calf, Left leg, Hip, Buttock, Thigh, Knee, Site of knee
 pain - Front, Back, Lateral/medial, Calf, Hip. 
 • Duration - 3 Days
 
  Physical_examination: **General exams:** 
 • Eyes: Jaundice-no jaundice seen, [picture taken]. 
 • Eyes: Pallor-normal pallor, [picture taken]. 
 • Arm-Pinch skin* - pinch test normal. 
 • Nail abnormality-nails normal, [picture taken]. 
 • Nail anemia-Nails are not pale, [picture taken]. 
 • Ankle-no pedal oedema. 
 **Any Location:** 
 • Ulcer:-no ulcer. 
 • Skin Rash:-no rash. 
 **Mouth:** 
 • back of throat normal. 
 **Joint:** 
 • tenderness seen. 
 • no deformity around joint. 
 • full range of movement is seen. 
 • joint is not swollen. 
 • no pain during movement. 
 • no redness around joint. 
 **Abdomen:** 
 • no tenderness. 
 **Back:** 
 • tenderness observed. 
 **Head:** 
 • No injury.
 
  Patient_medical_history: • Pregnancy status - Not pregnant. 
 • Allergies - No known allergies. 
 • Alcohol use - No. 
 • Smoking history - Patient denied/has no h/o smoking. 
 • Drug history - No recent medication. 
 
  Family_history: -
 
  Vitals:- 
 
 Sbp: 142.0
 
  Dbp: 80.0
 
  Pulse: 86.0
 
  Temperature: 36.22 'C
 
  Weight: 46.5 Kg
 
  Height: 154.0 cm
 
  RR: 22.0
 
  SPO2: 99.0
 
  HB: Null
 
  Sugar_random: Null
 
  Blood_group: Null
 
  Sugar_pp: Null
 
  Sugar_after_meal: Null"

"""

patient_case_vague_1 = """
"Gender: Male
 
  Age: 71 years
 
  Chief_complaint: ► **Cough** : 
 • Duration - Acute (0-2 weeks). 
 • Timing - खोकला, अशकपणा,. 
 • Aggravating factors - Cold weather. 
 • Type of cough - Wet - Colour of sputum - Clear. 
 • Recent h/o medication - None. 
 • Smoking - No h/o of smoking. 
 • Occupational history - शेती काम . 
 • Prior treatment sought - No. 
 ► **Fatigue and General weakness** : 
 • Duration - 4 Days. 
 • Timing - सकाळी, संध्याकाळी . 
 • Eating habits - जेवण,2 वेळा करता पण अनियमित. 
 • Stressful condition - No. 
 • Prior treatment sought - None. 
 ► **Skin disorder** : 
 • Type of the skin lesion - Eczematous skin. 
 • Site - Feet. 
 • No. of lesions - Single lesions. 
 • Duration - 16 Days. 
 • Progression - Transient. 
 • Exposure to irritants/offending agents - No. 
 • Prior treatment sought - 
 • Additional information - आशी
 
  Physical_examination: **General exams:** 
 • Eyes: Jaundice-no jaundice seen, [picture taken]. 
 • Eyes: Pallor-normal pallor, [picture taken]. 
 • Arm-Pinch skin* - appears slow on pinch test. 
 • Nail abnormality-nails normal, [picture taken]. 
 • Nail anemia-Nails are not pale, [picture taken]. 
 • Ankle-no pedal oedema, [picture taken]. 
 **Any Location:** 
 • Skin Bruise:-bruises seen, 1. , [picture taken]. 
 • Skin Rash:-no rash. 
 **Joint:** 
 • no deformity around joint. 
 • joint is not swollen. 
 • pain during movement.
 
  Patient_medical_history: • Allergies - जेवण जास्त झाल्यावर पंचन होत नाही . 
 • Alcohol use - Yes - No. of drinks consumed in one go - 1-2. 
 • Smoking history - Patient denied/has no h/o smoking. 
 • Drug history - No recent medication. 
 
  Family_history: •Do you have a family history of any of the following? : Asthma, Mother.. 
 
  Vitals:- 
 
 Sbp: 120.0
 
  Dbp: 85.0
 
  Pulse: 80.0
 
  Temperature: 36.39 'C
 
  Weight: 50.65 Kg
 
  Height: 165.0 cm
 
  RR: 21.0
 
  SPO2: 99.0
 
  HB: Null
 
  Sugar_random: Null
 
  Blood_group: Null
 
  Sugar_pp: Null
 
  Sugar_after_meal: Null"
"""

case_5 = """
"Gender: Female
 
  Age: 59 years
 
  Chief_complaint: ► **Headache** : 
 • Duration - 2 महिने. 
 • Site - Diffuse. 
 • Severity - Mild. 
 • Onset - Acute onset (Patient can recall exact time when it started). 
 • Character of headache - Throbbing. 
 • Radiation - pain does not radiate. 
 • Timing - No particular time. 
 • Exacerbating factors - bending, Standing up from lying position, Exposure to
 cold. 
 • Prior treatment sought - None. 
 ► **Leg, Knee or Hip Pain** : 
 • Site - Right leg, Hip, Buttock, Thigh, Knee, Site of knee pain - Front,
 Back, Lateral/medial, Calf, Left leg, Hip, Buttock, Thigh, Knee, Site of knee
 pain - Front, Back, Lateral/medial, Calf, Hip. 
 • Duration - 1 महिने. 
 • Pain characteristics - Dull aching, Tingling numbness. 
 • Onset - Sudden (minutes to hours). 
 • Progress - Wax and wane. 
 • वेदना पूर्ण शरीरात जातात . 
 • Aggravating factors - Symptom aggravated by motion, Associated with motio
 
  Physical_examination: **General exams:** 
 • Eyes: Jaundice-no jaundice seen, [picture taken]. 
 • Eyes: Pallor-normal pallor, [picture taken]. 
 • Arm-Pinch skin* - pinch test normal. 
 • Nail abnormality-nails normal, [picture taken]. 
 • Nail anemia-Nails are not pale, [picture taken]. 
 • Ankle-no pedal oedema. 
 **Joint:** 
 • tenderness seen. 
 • no deformity around joint. 
 • full range of movement is seen. 
 • joint is not swollen. 
 • no pain during movement. 
 • no redness around joint. 
 **Back:** 
 • tenderness observed. 
 **Head:** 
 • No injury.
 
  Patient_medical_history: • Pregnancy status - Not pregnant. 
 • Allergies - No known allergies. 
 • Alcohol use - No. 
 • Smoking history - Patient denied/has no h/o smoking. 
 • Medical History - High Blood Pressure - 25/May/2024 - Current medication -
 बी, पी ची गोळी चालू आहे.. 
 • Drug history - No recent medication. 
 
  Family_history: -
 
  Vitals:- 
 
 Sbp: 150.0
 
  Dbp: 101.0
 
  Pulse: 85.0
 
  Temperature: 36.83 'C
 
  Weight: 45.35 Kg
 
  Height: 150.0 cm
 
  RR: 21.0
 
  SPO2: 98.0
 
  HB: Null
 
  Sugar_random: Null
 
  Blood_group: Null
 
  Sugar_pp: Null
 
  Sugar_after_meal: Null"
  """

case_6 = """
"Gender: Male
 
  Age: 61 years
 
  Chief_complaint: ► **Headache** : 
 • Duration - 3 Days. 
 • Site - Diffuse. 
 • Severity - Mild. 
 • Onset - Acute onset (Patient can recall exact time when it started). 
 • Character of headache - Throbbing. 
 • Radiation - 
 • Timing - No particular time. 
 • Exacerbating factors - bending, Standing up from lying position, Exposure to
 cold. 
 • Prior treatment sought - None. 
 ► **Leg, Knee or Hip Pain** : 
 • Site - Right leg, Hip, Buttock, Thigh, Knee, Site of knee pain - Front,
 Back, Lateral/medial, Calf, Left leg, Hip, Buttock, Thigh, Knee, Site of knee
 pain - Front, Back, Lateral/medial, Calf, Hip. 
 • Duration - 3 Days. 
 • Pain characteristics - Dull aching. 
 • Onset - Sudden (minutes to hours). 
 • Progress - Wax and wane. 
 • वेदना पूर्ण जातात . 
 • Aggravating factors - Symptom aggravated by motion, Associated with motion -
 All planes of motion, Climbing stairs, Worsened by rest (relieved by activity
 
  Physical_examination: **General exams:** 
 • Eyes: Jaundice-no jaundice seen. 
 • Eyes: Pallor-normal pallor. 
 • Arm-Pinch skin* - pinch test normal. 
 • Nail abnormality-nails normal, [picture taken]. 
 • Nail anemia-Nails are not pale, [picture taken]. 
 • Ankle-no pedal oedema. 
 **Joint:** 
 • tenderness seen. 
 • no deformity around joint. 
 • full range of movement is seen. 
 • joint is not swollen. 
 • no pain during movement. 
 • no redness around joint. 
 **Back:** 
 • tenderness observed. 
 **Head:** 
 • No injury.
 
  Patient_medical_history: • Allergies - No known allergies. 
 • Alcohol use - No. 
 • Smoking history - Patient denied/has no h/o smoking. 
 • Drug history - No recent medication. 
 
  Family_history: -
 
  Vitals:- 
 
 Sbp: 105.0
 
  Dbp: 80.0
 
  Pulse: 86.0
 
  Temperature: 36.78 'C
 
  Weight: 43.25 Kg
 
  Height: 152.0 cm
 
  RR: 21.0
 
  SPO2: 99.0
 
  HB: Null
 
  Sugar_random: Null
 
  Blood_group: Null
 
  Sugar_pp: Null
 
  Sugar_after_meal: Null"
"""

case_7 = """
"Gender: Female
 
  Age: 19 years
 
  Chief_complaint: ► **Cold, Sneezing** : 
 • 3 Days. 
 • Precipitating factors - Wind. 
 • Prior treatment sought - None. 
 ► **Skin disorder** : 
 • Type of the skin lesion - Eczematous skin. 
 • Site - पायावर हातावर पुरळ . 
 • No. of lesions - Multiple lesions. 
 • Duration - 1 महिने. 
 • Progression - Transient. 
 • Exposure to irritants/offending agents - No. 
 • Prior treatment sought - None. 
 ► **Associated symptoms** : 
 • Patient reports - 
 Skin discharge - Blood. 
 • Patient denies - 
 Fever, Cough, Nasal congestion/Stuffy nose, Runny nose, Headache, Body pain,
 Chills, Painful skin lesion, Skin bruises, Nose bleed, Gum bleeding, Abdominal
 pain, Itchy skin, Joint pain, Dandruff, Sensitive to the sun 
 
  Physical_examination: **General exams:** 
 • Eyes: Jaundice-no jaundice seen, [picture taken]. 
 • Eyes: Pallor-normal pallor, [picture taken]. 
 • Arm-Pinch skin* - pinch test normal. 
 • Nail abnormality-nails normal, [picture taken]. 
 • Nail anemia-Nails are not pale, [picture taken]. 
 • Ankle-no pedal oedema. 
 **Any Location:** 
 • Skin Rash:-rash seen, skin appears to be dead, dry or peeling off (eschar
 seen). , [picture taken].
 
  Patient_medical_history: • Pregnancy status - Not pregnant. 
 • Allergies - No known allergies. 
 • Alcohol use - No. 
 • Smoking history - Patient denied/has no h/o smoking. 
 • Drug history - No recent medication. 
 
  Family_history: -
 
  Vitals:- 
 
 Sbp: 100.0
 
  Dbp: 73.0
 
  Pulse: 85.0
 
  Temperature: 36.83 'C
 
  Weight: 34.9 Kg
 
  Height: 149.0 cm
 
  RR: 21.0
 
  SPO2: 98.0
 
  HB: Null
 
  Sugar_random: Null
 
  Blood_group: Null
 
  Sugar_pp: Null
 
  Sugar_after_meal: Null"
"""

question = """You are a doctor with the following patient rural India.
        Here is their case with the history of presenting illness, their physical exams, and demographics.
        If there is no sufficient data to diagnose a patient from a case history of the patient provided, please mark the diagnosis as NA and output list of questions for the patient to make it easier to the doctor to make further progress. Do not include questions that are not pertinent to the provided case history.
        Otherwise, if there is sufficient data provide the below:
        What would be the top 5 differential diagnosis for this patient?
        For each diagnosis, include the likelihood score and the rationale for that diagnosis
        For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural india.
        For a low likelihood diagnosis, include lack of fit reasoning under the rationale for that diagnosis.
        Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis.
        If the case history data provided was sufficient to make diagnosis of a patient, leave further questions as empty string.
        Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."""

question_1 = """

"""
output = cot(case=patient_case_no_data , question=question)


print("diagnosis: ", output.output.diagnosis)
print("rationale: ", output.output.rationale)
print("conclusion: ", output.output.conclusion)
print("questions: ", output.output.further_questions)


print("##########################")
print(lm.inspect_history())
