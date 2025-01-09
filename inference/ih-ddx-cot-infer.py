import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge, load_gemini_lm
import os
import random
from dotenv import load_dotenv
import pandas as pd

from modules.DDxModule import DDxModule
load_dotenv(
    "ops/.env"
)
import sys

load_gemini_lm()

cot = DDxModule()
cot.load("outputs/" + "18_12_2024_ddx_open_ai_gemini_pro_num_trials_20_top_k5.json")



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

question = "You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis. Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."


output = cot(case=patient_case, question=question)


print("diagnosis: ", output.output.diagnosis)
print("rationale: ", output.output.rationale)
print("conclusion: ", output.output.conclusion)
