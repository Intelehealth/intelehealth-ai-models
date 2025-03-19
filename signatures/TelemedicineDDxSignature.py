import dspy

class TelemedicineDDxFields(dspy.Signature):
    """
        You are a doctor conducting a telemedicine consultation with a patient in rural India.
        You are provided with the history of presenting illness, their physical exams, and demographics for each patient.
        What would be the top 5 differential diagnosis for this patient?
        For each diagnosis, include the likelihood score and the rationale for that diagnosis
        For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural india.
        For a low likelihood diagnosis, include lack of fit reasoning under the rationale for that diagnosis.
        Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis.
        Please remember this is a telemedicine consultation for a patient in rural India and use this as a consideration for the diseases likely for the patient and dont output a json
    """
    case = dspy.InputField(desc="case with the history of presenting illness, physical exams, and demographics of a patient")
    question = dspy.InputField(desc="the patient prompt question")
    diagnosis = dspy.OutputField(desc="Top five differential diagnosis for the patient ranked by likelhood")
    rationale = dspy.OutputField(desc="detailed chain of thought reasoning for each of these top 5 diagnosis predicted. Please don't include the case and question in the rationale.")
    conclusion = dspy.OutputField(desc="Final conclusion on top likely diagnosis considering all rationales")
    further_questions = dspy.OutputField(desc="further questions to ask the patient ONLY if the data was not sufficient. Please only ask questions that are relevant to the patient's case. Don't include any questions outside scope of provided patient data.")
    telemedicine_limitations = dspy.OutputField(desc="limitations of making this diagnosis via telemedicine and what in-person examinations might be beneficial")
    follow_up_recommendations = dspy.OutputField(desc="recommendations for follow-up care including nearest healthcare facilities if needed") 