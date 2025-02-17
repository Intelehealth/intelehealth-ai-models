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


response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"model_name": "gemini-2.0-flash", "case": patient_case_1, "prompt_version": 1 }
)

if response.status_code == 200:
    formatted_json = json.dumps(response.json(), indent=4)
    print(formatted_json)
else:
    print(f"Error: {response.status_code}")
    print(response.text)

