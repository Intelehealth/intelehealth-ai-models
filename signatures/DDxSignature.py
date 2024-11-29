import dspy


class DDxFields(dspy.Signature):
    """You are a doctor with the following patient rural India.
    Here is their case with the history of presenting illness, their physical exams, and demographics.
    What would be the top differential diagnosis for this patient?
    Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."""
    case = dspy.InputField(desc="case with the history of presenting illness, physical exams, and demographics of a patient")
    question = dspy.InputField(desc="the patient prompt question")
    diagnosis = dspy.OutputField(desc="Topmost diagnosis for the patient")
    rationale = dspy.OutputField(desc="a detailed explanation for this diagnosis")
