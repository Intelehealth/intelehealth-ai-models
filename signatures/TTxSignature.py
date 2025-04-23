import dspy

class TTxFields(dspy.Signature):
    """
        Based on given patient history, symptoms, physical exam findings, and the diagnosis, provide the medication for the patient.
        
        For each medication: include the name of the medication, the dosage, number of days to take the medication, and the reason for the medication.

        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Keep all responses concise and to the point.
    """
    case = dspy.InputField(desc="Patient history, symptoms, physical exam findings, and demographics")
    question = dspy.InputField(desc="Patient prompt question")
    diagnosis = dspy.InputField(desc="Diagnosis of the patient as done by the doctor")
    medications = dspy.OutputField(desc="Top 5 relevant medications with the likelihood (high/moderate/low) with brief rationale for each of the medications.")
    further_questions = dspy.OutputField(desc="2-3 critical follow-up questions if needed, only relevant to the case")
    follow_up_recommendations = dspy.OutputField(desc="Concise care recommendations including nearby facilities if needed.") 