import dspy

from signatures.TelemedicineSnomedCTDDxSignature import TelemedicineSnomedCTDDxFields

class TelemedicineSnomedCTDDxModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(TelemedicineSnomedCTDDxFields)

    def forward(self, case, question):
        print("Question in forward:")
        question = """
            Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
            Based on patient history, symptoms, physical exam findings, and demographics:
            1. Provide the top 5 differential diagnoses, with highest confidence ranked in order of likelihood, picked from the snomed ct database.
            2. Ensure diagnoses are relevant to a telemedicine context in India.
            3. For each diagnosis: include a brief rationale and the confidence of prediction - high, moderate, low etc.
            4. do not include any snomed ct codes in the response.

            Keep all responses concise and to the point.
        """
        print(question)

        prediction = self.generate_answer(case=case, question=question)
        print(prediction)
        
        return dspy.Prediction(output=prediction) 