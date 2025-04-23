import dspy

from signatures.TTxSignature import TTxFields

class TTxModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(TTxFields)

    def forward(self, case, diagnosis):
        print("Question in forward:")
        question = "What is the relevant medication, dosage, number of days to take the medication, and the reason for the medication for the patient given the diagnosis and patient case?"
        print(question)
        prediction = self.generate_answer(case=case, diagnosis=diagnosis, question=question)
        print(prediction)
        return dspy.Prediction(output=prediction) 