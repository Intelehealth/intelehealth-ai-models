import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge
import os
import random
from dotenv import load_dotenv
from modules.DDxModule import DDxModule
load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=1.0)

dspy.configure(lm=lm)


cot = DDxModule()
cot.load("outputs/" + "ddx_open_ai_gpt-01_cot_trial_cleaned_data_llm_judge_metric.json")

case = """
HISTORY 
CC/ID: 52-year-old man with a painful. swollen, hot right knee. Mr…  was awakened from sleep last night by exquisite pain in his right knee, which had become  swollen and warm. He had felt well during the day preceding the onset of pain. and had attended a crab-fest that afternoon. He denies trauma to the joint. penetrating injuries, injections, or extramarital sexual contact. He lives in the city and does not enjoy hiking or camping, although he likes to fish. 


He cannot recall any tick exposures. He reports subjective fevers and sweats, but hasn't taken his temperature; the ROS is otherwise negative. Although no other joints currently hurt, he recalls an intensely painful left big toe several years ago that got better with ""aspirin. "" 


Past Medical History: Hypertension; ; hernia repair. Meds:. HCTZ. 25 mg  All: NKDA 


Family History/Social History: Parents both deceased; father had ""arthritis."" Married, with two adult children. 


PHYSICAL EXAMINATION


VS: Temp 39C. BP 150/90. HR 100, RR 16. 02 sat RA 


Gen: large man, lying on gurney, nontoxic but uncomfortable. HEENT: unremarkable. Neck: supple, no thyromegaly or adenopathy. Lungs: clear. Q': RRR. normal S,Sz, 216 HSM at apex. 
Abdomen: soft. NT/ND, +BS. Ext: no track marks; right knee swollen, warm, tender to touch; no swelling or lymphangitis; 2+ peripheral pulses. Skin: no rashes. no necrotic-appearing lesions. Genitourinary system: no urethral discharge. Neuro: nonfocal. 


LABORATORY


WBC 13; Hct. Plt Count normal; Cr 1.2; serum urate normal; INR 1.0.
"""

case_2 = """
"You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis. Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient.

Mr …, 65 years old, retired police officer, normotensive, nondiabetic, nonsmoker presented with multiple blisters over different parts of the body for 2 months. The blisters at first appeared over forearms and gradually involved axilla, groin and front and back of trunk. Most of the
blisters are large and tense, few are small. Some blisters have ruptured leaving denuded areas that have healed spontaneously. He also noticed erythematous patches over the front and back of the trunk, some of which healed spontaneously. There is no involvement of the mouth or genitalia. The patient also complains of mild generalized itching for the same duration. He had similar attack 3 years back, but the lesions were sparse and recovered completely. He did not complain of difficulty in swallowing. There is no history of fever, joint pain, muscle pain or intake of any drugs prior to the appearance of these blisters. His bowel and bladder habits are normal.

General examination
 
The patient looks very ill and apathetic.  Mildly anemic. No jaundice, cyanosis, edema, dehydration, clubbing, koilonychias, leukonychia. No lymphadenopathy and no thyromegaly

 Pulse—98/min
 Blood pressure—110/70 mm of Hg
 Temperature—98 °F
 Respiratory rate—20/min.

Integumentary system
1. Skin:
 Multiple large tense bulla are present over flexor aspect of forearms, groin, axilla, front and
back of the trunk, some are on normal skin and some are on erythematous skin. Intact blisters
contain clear fluid. Some bullae are ruptured. There are multiple denuded areas that do not increase by confluence. Most of the ruptured bullae are in healing stage and some are healed with mild hyperpigmentation.
Dermatology
 Erythematous patches are mostly observed over the trunk. Most of the lesions show central 
 Nikolsky and Asboe-Hansen signs are negative.

2. Hair: normal.

3. Nail: normal.

4. Mucous membrane:
 Conjunctival—normal.
 Oral—normal.
 Genital—normal.Examination of other systems reveals no abnormalities."
"""

question = """
You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis. Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient.
"""
output = cot(case=case_2, question=question)

print("############ INFERENCE: #############")
print("Diagnosis :" , output.output.diagnosis)

print("\n")
print("Rationale :", output.output.rationale)
