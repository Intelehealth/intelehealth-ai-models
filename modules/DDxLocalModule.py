import dspy

from signatures.DDxLocalQnSignature import DDxLocalFields

class DDxLocalModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(DDxLocalFields)

    def forward(self, case, question):
        print("Question in forward:")
        print(question)
        question = """
            You are a doctor/physician assigned with task of diffential diagnosis on a patient from rural India.
            Given their case with the history of presenting illness, symptoms, their physical exams, and demographics, give me the top 5 differential diagnosis for this patient with likelihood. 

            Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient.
        """
        prediction = self.generate_answer(case=case, question=question)
        print(prediction)
        print("diagnosis prediction: ", prediction.diagnosis)
        print("diagnosis prediction type: ", type(prediction.diagnosis))
        prediction.diagnosis = prediction.diagnosis.lstrip("[").rstrip("]")
        
        return dspy.Prediction(output=prediction)