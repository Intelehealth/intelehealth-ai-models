import dspy

class DDxLocalFields(dspy.Signature):
    """
        You are a doctor/physician assigned with task of diffential diagnosis on a patient from rural India.
        Given their case with the history of presenting illness, symptoms, their physical exams, and demographics, give me the top 5 differential diagnosis for this patient with likelihood. 
    """
    case = dspy.InputField(desc="case with the history of presenting illness, symptoms, physical exams, and demographics of a patient")
    question = dspy.InputField(desc="the patient prompt question")
    diagnosis = dspy.OutputField(desc="Top five differential diagnosis for the patient ranked by likelihood with each diagnosis separated by a comma. Do not include any brackets or other punctuation.")
    # rationale = dspy.OutputField(desc="detailed chain of thought reasoning for each of these top 5 diagnosis predicted. don't include the case and question in this one.")
    # conclusion = dspy.OutputField(desc="Final conclusion on top likely diagnsois considering all rationales")
