import dspy
from typing import Optional


class DiagnosisItem(dspy.Signature):
    """
        Represents a single differential diagnosis item.
    """
    diagnosis_name = dspy.OutputField(desc="Name of the differential diagnosis")
    likelihood = dspy.OutputField(desc="Likelihood score for this diagnosis (e.g., High, Moderate, Low)")
    rationale = dspy.OutputField(desc="Detailed rationale for this diagnosis, including clinical relevance, features, rural India context, etc.")

class DDxOutput(dspy.Signature):
    """
        Represents the complete output of the differential diagnosis task.
    """
    diagnoses: Optional[list[DiagnosisItem]]= dspy.OutputField(desc="List of top 5 differential diagnoses, each with likelihood and rationale")
    conclusion = dspy.OutputField(desc="Final conclusion on top likely diagnoses considering all rationales")

class DDxNestedSignature(dspy.Signature):
    """
        You are a doctor with the following patient rural India.
        Here is their case with the history of presenting illness, their physical exams, and demographics.
        What would be the top differential diagnosis UPTO 5 for this patient?

        For each diagnosis, include the likelihood score and the rationale for that diagnosis
        For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural india.
        For a low likelihood diagnosis, include lack of fit reasoning under the rationale for tha diagnosis.
        Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis.
        Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient
    """
    case = dspy.InputField(desc="case with the history of presenting illness, physical exams, and demographics of a patient")
    question = dspy.InputField(desc="the patient prompt question")
    ddx_results: list[DDxOutput]= dspy.OutputField(desc="list of differential diagnoses")