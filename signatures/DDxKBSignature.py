import dspy

class DDxKBSignature(dspy.Signature):
    """
        Based on given patient history, symptoms, physical exam findings, and demographics please provide top 5 differential diagnoses ranked by likelihood. Refer to ICD 11 and SNOMED CT terminologies to provide the diagnosis.
        
        For each diagnosis: include likelihood score (high/moderate/low) and brief rationale
        For high/moderate: mention key features, infections, and rural India relevance
        For low: briefly explain why it doesn't fit

        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Keep all responses concise and to the point.
    """
    case = dspy.InputField(desc="Patient history, symptoms, physical exam findings, and demographics")
    question = dspy.InputField(desc="Patient prompt question")
    diagnosis = dspy.OutputField(desc="Top 5 differential diagnoses with the likelihood (high/moderate/low)")
    further_questions = dspy.OutputField(desc="2-3 critical follow-up questions if needed, only relevant to the case")
    follow_up_recommendations = dspy.OutputField(desc="Concise care recommendations including nearby facilities if needed") 