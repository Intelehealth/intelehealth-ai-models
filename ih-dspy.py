import dspy
from utils.metric_utils import load_gemini_lm, load_ollama_deepseek_r1_32b_llm_url_non_chat, load_gemini_lm_prod, load_ollama_meditron_70b_url
from dotenv import load_dotenv


load_dotenv(
    "ops/.env"
)

# Get the LLM instance
load_ollama_deepseek_r1_32b_llm_url_non_chat()
#load_ollama_meditron_70b_url()
#load_gemini_lm_prod()
# Configure DSPy to use the LLM
# dspy.configure(lm=lm)


# from signatures.DDxQnSignature import DDxQnFields
from signatures.DDxNestedSignature import DDxNestedSignature

class DDxQnModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(DDxNestedSignature)

    def forward(self, case, question):
        prediction = self.generate_answer(case=case, question=question)
        print(prediction)
        return dspy.Prediction(output=prediction)


import dspy
import os

# OR if you are using a local model with LlamaCpp
# from dspy.llm import LlamaCpp
# llm = LlamaCpp(model_path="./models/llama-2-7b-chat.Q5_K_M.gguf") #Replace with your model path
# dspy.settings.configure(lm=llm)

class CaseDiagnosis(dspy.Signature):
    """Given a medical case and a question, diagnose the condition."""
    case = dspy.InputField(desc="A detailed description of the patient's symptoms, medical history, and examination findings.")
    question = dspy.InputField(desc="A specific question about the case, often asking for a diagnosis or differential.")
    # diagnosis = dspy.OutputField(desc="A well-reasoned diagnosis based on the provided case information, explaining the reasoning process.")
    diagnosis = dspy.OutputField(desc="Top five differential diagnosis for the patient ranked by likelhood")
    rationale = dspy.OutputField(desc="detailed chain of thought reasoning for each of these top 5 diagnosis predicted. don't include the case and question in this one.")
    conclusion = dspy.OutputField(desc="Final conclusion on top likely diagnsois considering all rationales")
    further_questions = dspy.OutputField(desc="Further questions for the patient if diagnosis was marked as 'INSUFFICIENT DATA'")


class DiagnoseCase(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(CaseDiagnosis)
    
    def forward(self, case, question):
        prediction = self.predictor(case=case, question=question)
        return prediction

# Create the module
diagnose_module = DDxQnModule()


question_prompt = "You are a doctor with the following patient rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis. Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."



# Run with the default LM configured with `dspy.configure` above.
case_description= """
HISTORY
                    31 year old woman who was previously in good health presents to the clinic with a 4 day history of severe nausea and vomiting. She has been unable to keep down liquids secondary to profound nausea. She also reports general fatigue and weakness over the past week and a 2 day history of swelling in her lower legs, face and arms. She denies any fever or chills, but does report feeling feverish approximately 2 weeks ago in the setting of a severe sore throat with both symptoms resolving after a few days.

                                              She has no cough but does describe mild shortness of breath with exertion over the past week. No abdominal pain, diarrhea or constipation. She has no dysuria() but does note that urine has been small in volume and brownish in colour. She reports no new medications, no changes in diet, no one sick at home, she does work in an elementary school and often gets sick.

                                                                        Past Medical History : depression Allergy: No known drug allergies

                                                                                                  Social History: No smoking, alcohol intake, no drugs.
                                                                                                                            Temp 36.2C BP 155/95 HR 85 RR 16 O2 sat 94% RA


                                                                                                                                                      PHYSICAL EXAMINATION


                                                                                                                                                                                Ill appearing young woman with swollen face. Abdomen is soft, +BS, mildly and diffusely tender to palpation, non distended
                                                                                                                                                                                                          Extremities: Pitting edema in hands and legs, no rashes, no joint abnormalities"""


# Run it
diagnosis = diagnose_module(case=case_description, question=question_prompt)
print("--- Generated Diagnosis ---")
print(diagnosis)
