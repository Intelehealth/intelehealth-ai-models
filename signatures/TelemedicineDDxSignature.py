import dspy

class TelemedicineDDxFields(dspy.Signature):
    """
        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Based on patient history, symptoms, physical exam findings, and demographics:
        1. Provide top 5 differential diagnoses ranked by likelihood
        2. For each: include likelihood score (high/moderate/low) and brief rationale
        3. For high/moderate: mention key features, infections, and rural India relevance
        4. For low: briefly explain why it doesn't fit
        Keep all responses concise and to the point.
    """
    case = dspy.InputField(desc="Patient history, symptoms, physical exam findings, and demographics")
    question = dspy.InputField(desc="Patient prompt question")
    diagnosis = dspy.OutputField(desc="Top 5 differential diagnoses with likelihood (high/moderate/low)")
    rationale = dspy.OutputField(desc="Brief reasoning for each diagnosis, without repeating case details")
    further_questions = dspy.OutputField(desc="2-3 critical follow-up questions if needed, only relevant to the case")
    follow_up_recommendations = dspy.OutputField(desc="Concise care recommendations including nearby facilities if needed") 