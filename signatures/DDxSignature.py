import dspy

class DDxFields(dspy.Signature):
    """
        You are a doctor with the following patient rural India.
        Here is their case with the history of presenting illness, their physical exams, and demographics.
        What would be the top 5 differential diagnosis for this patient?
        For each diagnosis, include the likelihood score and the rationale for that diagnosis
        For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural india.
        For a low likelihood diagnosis, include lack of fit reasoning under the rationale for tha diagnosis.
        Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis.
        Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient.
    """
    case = dspy.InputField(desc="case with the history of presenting illness, physical exams, and demographics of a patient")
    question = dspy.InputField(desc="the patient prompt question")
    diagnosis = dspy.OutputField(desc="Top five differential diagnosis for the patient ranked by likelhood")
    rationale = dspy.OutputField(desc="detailed chain of thought reasoning for each of these top 5 diagnosis predicted. don't include the case and question in this one.")
    conclusion = dspy.OutputField(desc="Final conclusion on top likely diagnsois considering all rationales")
