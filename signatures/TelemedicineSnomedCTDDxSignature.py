import dspy

class TelemedicineSnomedCTDDxFields(dspy.Signature):
    """
        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Based on patient history, symptoms, physical exam findings, and demographics:
        1. Provide the top 5 differential diagnoses, with highest confidence ranked in order of likelihood, picked from the snomed ct database.
        2. Ensure diagnoses are relevant to a telemedicine context in India.
        3. For each diagnosis: include a brief rationale.

        Keep all responses concise and to the point.
    """
    case = dspy.InputField(desc="Patient history, symptoms, physical exam findings, and demographics")
    question = dspy.InputField(desc="Patient prompt question")
    diagnosis = dspy.OutputField(desc="Top 5 differential diagnoses from ICD-11, with highest confidence, excluding specified chapters.")
    rationale = dspy.OutputField(desc="Brief reasoning for each diagnosis, without repeating case details")
    further_questions = dspy.OutputField(desc="2-3 critical follow-up questions if needed, only relevant to the case")
    conclusion = dspy.OutputField(desc="Final conclusion in a crisp format of limited to 2-3 lines without losing relevance")
    follow_up_recommendations = dspy.OutputField(desc="Concise care recommendations including nearby facilities if needed") 