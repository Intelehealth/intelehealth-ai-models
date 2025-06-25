import requests
import json

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


patient_case_1 = """
"Gender: Male
 
  Age: 21 years
 
  Chief_complaint: ► **Skin disorder** : 
 • Type of the skin lesion - Skin rash. 
 • Site - Face. 
 • No. of lesions - Multiple lesions. 
 • Duration - 1 महिने. 
 • Progression - Transient. 
 • Exposure to irritants/offending agents - No. 
 • Prior treatment sought - None. 
 ► **Associated symptoms** : 
 • Patient reports - 
 Skin discharge - Clear. 
 • Patient denies - 
 Painful skin lesion, Skin bruises, Nose bleed, Gum bleeding, Abdominal pain,
 Fever, Itchy skin, Runny nose, Joint pain, Dandruff, Sensitive to the sun 
 
  Physical_examination: **General exams:** 
 • Eyes: Jaundice-no jaundice seen, [picture taken]. 
 • Eyes: Pallor-normal pallor, [picture taken]. 
 • Arm-Pinch skin* - pinch test normal. 
 • Nail abnormality-nails normal, [picture taken]. 
 • Nail anemia-Nails are not pale, [picture taken]. 
 • Ankle-no pedal oedema. 
 **Any Location:** 
 • Skin Rash:-rash seen, 5. surface is smooth. rash not present on palms and
 soles. no eschar. , [picture taken].
 
  Patient_medical_history: • Allergies - No known allergies. 
 • Alcohol use - No. 
 • Smoking history - Patient denied/has no h/o smoking. 
 • Drug history - No recent medication. 
 
  Family_history: -
 
  Vitals:- 
 
 Sbp: 102.0
 
  Dbp: 83.0
 
  Pulse: 86.0
 
  Temperature: 37.0 'C
 
  Weight: 46.0 Kg
 
  Height: 156.0 cm
 
  RR: 21.0
 
  SPO2: 99.0
 
  HB: Null
 
  Sugar_random: Null
 
  Blood_group: Null
 
  Sugar_pp: Null
 
  Sugar_after_meal: Null"
"""

test_case_1 = """Gender: Male

  Age: 25 years
  """
patient_case_2 = """
Gender: Male\n\n\u00a0 Age: 21 years\n\n\u00a0 Chief_complaint: ► **Skin disorder** :\n\u00a0• Type of the skin lesion - Skin rash.\n\u00a0• Site - Face.\n\u00a0• No. of lesions - Multiple lesions.\n\u00a0• Duration - 1 महिने.\n\u00a0• Progression - Transient.\n\u00a0• Exposure to irritants/offending agents - No.\n\u00a0• Prior treatment sought - None.\n\u00a0► **Associated symptoms** :\n\u00a0• Patient reports -\n\u00a0Skin discharge - Clear.\n\u00a0• Patient denies -\n\u00a0Painful skin lesion, Skin bruises, Nose bleed, Gum bleeding, Abdominal pain,\n\u00a0Fever, Itchy skin, Runny nose, Joint pain, Dandruff, Sensitive to the sun\n\n\u00a0 Physical_examination: **General exams:**\n\u00a0• Eyes: Jaundice-no jaundice seen, [picture taken].\n\u00a0• Eyes: Pallor-normal pallor, [picture taken].\n\u00a0• Arm-Pinch skin* - pinch test normal.\n\u00a0• Nail abnormality-nails normal, [picture taken].\n\u00a0• Nail anemia-Nails are not pale, [picture taken].\n\u00a0• Ankle-no pedal oedema.\n\u00a0**Any Location:**\n\u00a0• Skin Rash:-rash seen, 5. surface is smooth. rash not present on palms and\n\u00a0soles. no eschar. , [picture taken].\n\n\u00a0 Patient_medical_history: • Allergies - No known allergies.\n\u00a0• Alcohol use - No.\n\u00a0• Smoking history - Patient denied/has no h/o smoking.\n\u00a0• Drug history - No recent medication.\n\n\u00a0 Family_history: -\n\n\u00a0 Vitals:-\n\n\u00a0Sbp: 102.0\n\n\u00a0 Dbp: 83.0\n\n\u00a0 Pulse: 86.0\n\n\u00a0 Temperature: 37.0 '\''C\n\n\u00a0 Weight: 46.0 Kg\n\n\u00a0 Height: 156.0 cm\n\n\u00a0 RR: 21.0\n\n\u00a0 SPO2: 99.0\n\n\u00a0 HB: Null\n\n\u00a0 Sugar_random: Null\n\n\u00a0 Blood_group: Null\n\n\u00a0 Sugar_pp: Null\n\n\u00a0 Sugar_after_meal: Null
"""

patient_case_3 = """
"Gender: Female

 Age: 46 years

 Chief_complaint: ► **Leg, Knee or Hip Pain** :  
• Site - Right leg - Calf. Left leg - Calf.  
• Duration - 2 महिने.  
• Pain characteristics - Tingling numbness.  
• Onset - Gradual.  
• Progress - Static (Not changed).  
• हाताला आणि पायाला मुंग्या येतात .  
• Aggravating factors - None.  
• H/o specific illness -  
• Patient reports -  
पोटामध्ये आतड्याला सूज  
• Trauma/surgery history - No recent h/o trauma/surgery.  
• Injection drug use - No h/o injection / drug use.  
• Prior treatment sought - None.  
► **Oedema** :  
• Duration - 1 महिने.  
• Site - Localized - पोताच्या वरती डावीकडे .  
• Onset - Gradually increased - 1 महिने.  
• Swelling symptoms - Swelling is painful. Surface of swelling - Rough.  
• Prior treatment sought - खाजगी दावाखण्या

 Physical_examination: **General exams:**  
• Eyes: Jaundice-no jaundice seen, [picture taken].  
• Eyes: Pallor-normal pallor, [picture taken].  
• Arm-Pinch skin* - pinch test normal.  
• Nail abnormality-nails normal, [picture taken].  
• Nail anemia-Nails are not pale, [picture taken].  
• Ankle-no pedal oedema, [picture taken].  
**Joint:**  
• non-tender.  
• no deformity around joint.  
• full range of movement is seen.  
• joint is not swollen.  
• no pain during movement.  
• no redness around joint.  
**Neck:**  
• Thyroid swelling-no swelling in front of neck.  
**Face:**  
• face appears normal.  
**Back:**  
• tenderness observed.

 Patient_medical_history: • Pregnancy status - Not pregnant.  
• Allergies - No known allergies.  
• Alcohol use - No.  
• Smoking history - Patient denied/has no h/o smoking.  
• Medical History - None.  
• Drug history - No recent medication.  

 Family_history: •Do you have a family history of any of the following? : None.  

 Vitals:- 

Sbp: 139.0

 Dbp: 81.0

 Pulse: 88.0

 Temperature: 36.28 'C

 Weight: 58.0 Kg

 Height: 151.0 cm

 BMI: 25.44

 RR: 22.0

 SPO2: 98.0

 HB: Null

 Sugar_random: Null

 Blood_group: Null

 Sugar_pp: Null

 Sugar_after_meal: Null

"""
patient_case_4 = """
Patient Information:

Gender: Male

Age: 21 years

Chief Complaint:

Skin disorder:

Type of skin lesion: Skin rash.

Site: Face.

Number of lesions: Multiple lesions.

Duration: 1 month.

Progression: Transient.

Exposure to irritants/offending agents: No.

Prior treatment sought: None.

Associated symptoms:

Patient reports:

Skin discharge - Clear.

Patient denies:

Painful skin lesion

Skin bruises

Nose bleed

Gum bleeding

Abdominal pain

Fever

Itchy skin

Runny nose

Joint pain

Dandruff

Sensitive to the sun

Physical Examination:

General exams:

Eyes: No jaundice seen [picture taken].

Eyes: Normal pallor [picture taken].

Arm-Pinch skin: Pinch test normal.

Nail abnormality: Nails normal [picture taken].

Nail anemia: Nails are not pale [picture taken].

Ankle: No pedal edema.

Any Location:

Skin Rash: Rash seen, surface is smooth. Rash not present on palms and soles. No eschar [picture taken].

Patient Medical History:

Allergies: No known allergies.

Alcohol use: No.

Smoking history: Patient denied/has no history of smoking.

Drug history: No recent medication.

Family History: -

Vitals:

Systolic Blood Pressure (SBP): 102.0

Diastolic Blood Pressure (DBP): 83.0

Pulse: 86.0

Temperature: 37.0 °C

Weight: 46.0 Kg

Height: 156.0 cm

Respiratory Rate (RR): 21.0

Oxygen Saturation (SPO2): 99.0

Hemoglobin (HB): Null

Random Blood Sugar: Null

Blood Group: Null

Postprandial Blood Sugar (Sugar_pp): Null

Blood Sugar After Meal: Null
"""

patient_case_5 = """
"Gender: Female

 Age: 57 years

 Chief_complaint: ► **Abdominal Pain** :  
• Site - Middle (C) - Umbilical.  
• Pain does not radiate.  
• 3 Days.  
• Onset - Sudden.  
• Timing - Morning, Night.  
• Character of the pain - Colicky / Intermittent (comes & goes).  
• Severity - Moderate, 4-6.  
• Exacerbating Factors - None.  
• Relieving Factors - None.  
• Menstrual history - Menopause  
• Prior treatment sought - None.  
► **Cold, Sneezing** :  
• 3 Days.  
• Precipitating factors - Cold weather.  
• Prior treatment sought - None.  
► **Cough** :  
• Timing - Day, Night.  
• Aggravating factors - Cold weather.  
• Type of cough - Wet - Colour of sputum - Clear.  
• Recent h/o medication - None.  
• Smoking - No h/o of smoking.  
• Occupational history - शेतातील काम .  
• Prior treatment sought - No.  
► **Associated symptoms** :  
• Patient reports -  
Fever, Abdominal distention/Bloating, Belching/Burping, Breathless

 Physical_examination: **General exams:**  
• Eyes: Jaundice-no jaundice seen, [picture taken].  
• Eyes: Pallor-normal pallor, [picture taken].  
• Arm-Pinch skin* - pinch test normal.  
• Nail abnormality-nails normal, [picture taken].  
• Nail anemia-Nails are not pale.  
• Ankle-no pedal oedema, [picture taken].  
**Abdomen:**  
• no distension.  
• no scarring.  
• no tenderness.  
• Lumps-no lumps.

 Patient_medical_history: • Pregnancy status - Not pregnant.  
• Allergies - No known allergies.  
• Alcohol use - No.  
• Smoking history - Patient denied/has no h/o smoking.  
• Medical History - None.  
• Drug history - No recent medication.  

 Family_history: •Do you have a family history of any of the following? : None.  

 Vitals:- 

Sbp: 130.0

 Dbp: 84.0

 Pulse: 97.0

 Temperature: 36.56 'C

 Weight: 43.0 Kg

 Height: 152.0 cm

 BMI: 18.61

 RR: 21.0

 SPO2: 99.0

 HB: Null

 Sugar_random: Null

 Blood_group: Null

 Sugar_pp: Null

 Sugar_after_meal: Null
"""
# EARLIER /predict endpoint
# response = requests.post(
#     "http://127.0.0.1:8000/predict",
#     json={"model_name": "gemini-2.0-flash", "case": patient_case_3, "prompt_version": 1 }
# )

# if response.status_code == 200:
#     formatted_json = json.dumps(response.json(), indent=4)
#     print(formatted_json)
# else:
#     print(f"Error: {response.status_code}")
#     print(response.text)

# Test case for predict/v1 endpoint

# Test case for predict/v1 endpoint


patient_case_6 = """Gender: M\nAge: 23\n\nChief_complaint: **Cold, Sneezing**: \n•  2 Hours.\n• Precipitating factors - Cold weather.\n• Prior treatment sought - None.\n• Additional information - Test.\n** Associated symptoms**:  \n• Patient reports -\n Body pain,  Chills \n• Patient denies -\n Cough,  Fever,  Headache,  Itchy throat,  Nasal congestion/Stuffy nose,  Runny nose\n\nPhysical_examination: **General exams: **\n• Eyes: Jaundice-no jaundice seen. \n• Eyes: Pallor-normal pallor. \n• Arm-Pinch skin* - pinch test normal. \n• Nail abnormality-nails normal. \n• Nail anemia-Nails are normal. \n• Ankle-no pedal oedema.\n\nFamily_history: Do you have a family history of any of the following? : • Diabetes, Mother..\n\nMedical_history: • Allergies - No known allergies.\n• Alcohol use - No.\n• Smoking history - Patient denied/has no h/o smoking.\n• Medical History - Diabetes - Current medication - Can't tell the name of a medicine.\n• Drug history - No recent medication.\n\n\n\n"""

patient_case_7 = """
"""

print("\n\nTesting /predict/v1 endpoint:")
response_v1 = requests.post(
    "http://127.0.0.1:8000/predict/v1",
    json={"model_name": "gemini-2.0-flash", "case": patient_case_6, "prompt_version": "1", "tracker": "test_tracker_123"}
)

if response_v1.status_code == 200:
    formatted_json = json.dumps(response_v1.json(), indent=4)
    print(formatted_json)
else:
    print(f"Error: {response_v1.status_code}")
    print(response_v1.text)

# Test case for predict/v2 endpoint
