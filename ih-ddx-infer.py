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
cot.load("outputs/" + "ddx_open_ai_gpt-01_cot_trial1_llm_judge_metric.json")

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

question = """
You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis. Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient.
"""
output = cot(case=case, question=question)

print("############ INFERENCE: #############")
print("Diagnosis :" , output.output.diagnosis)

print("\n")
print("Rationale :", output.output.rationale)
