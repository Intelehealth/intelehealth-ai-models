import dspy

class TelemedicineTenDDxFields(dspy.Signature):
    """
        Your role is to act as a doctor / healthcare worker conducting a telemedicine consultation with a patient in rural India.
        Based on patient history, symptoms, physical exam findings, and demographics:
        1. Provide top 10 differential diagnoses ranked by likelihood
        2. For each: include likelihood score (high/moderate/low) and brief rationale
    
        Keep all responses concise and to the point.
    """
    case = dspy.InputField(desc="Patient history, symptoms, physical exam findings, vitals and demographics")
    question = dspy.InputField(desc="Patient prompt question")
    diagnosis = dspy.OutputField(desc="Top 10 differential diagnoses with likelihood (high/moderate/low)")
    rationale = dspy.OutputField(desc="Brief reasoning for each diagnosis, without repeating case details")
    further_questions = dspy.OutputField(desc="2-3 critical follow-up questions if needed, only relevant to the case")
    follow_up_recommendations = dspy.OutputField(desc="Concise care recommendations including nearby facilities if needed") 