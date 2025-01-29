import dspy

class DDxMulFields(dspy.Signature):
    """
        You are a doctor with the following patient rural India.
        Here is their case with the history of presenting illness, symptoms, their physical exams, and demographics.
        What would be the top primary diagnosis for this patient?
        What would be the additional primary diagnosis for this patient ?
        Also include other differential diagnosis other than the primary diagnosis found above upto 3.
        For each diagnosis, include the likelihood score and the rationale for that diagnosis
        For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural india.
        For a low likelihood diagnosis, include lack of fit reasoning under the rationale for that diagnosis.
        Please rank the differential diagnoses based on the likelihood.
        If the data provided for the patient is not is not sufficient to make diagnosis, mark both primary diagnosis as "NA" and ask further questions to be asked to make diagnosis more clear.
        If there was a primary diagnosis found, mark further questions as "NA"
        Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient.
    """
    case = dspy.InputField(desc="case with the history of presenting illness, physical exams, and demographics of a patient")
    question = dspy.InputField(desc="the patient prompt question")
    primary_diagnosis_1 = dspy.OutputField(desc="Top primary diagnosis for the patient.")
    primary_diagnosis_2 = dspy.OutputField(desc="Another top primary diagnosis for the patient. If not available, put NA.")
    diagnosis = dspy.OutputField(desc="the other diagnosis not captured in above primary diagnosis ranked in order of likelihood upto three.")
    rationale = dspy.OutputField(desc="brief chain of thought reasoning for each of these diagnosis predicted. don't include the case and question in this one.")
    further_questions = dspy.OutputField(desc="further questions to ask the patient ONLY if the data was not sufficient. don't include any questions outside scope of provided patient data.")
