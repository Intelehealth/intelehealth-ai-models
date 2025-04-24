import dspy

from signatures.TTxv2Signature import TTxv2Fields

class TTxv2Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(TTxv2Fields)

    def forward(self, case, diagnosis):
        print("Question in forward:")
        question = "What is the relevant  medication, the strength, route form, the dosage, frequency, number of days to take the medication and the reason for the medication for the patient given the diagnosis and patient case?"
        print(question)
        prediction = self.generate_answer(case=case, diagnosis=diagnosis, question=question)
        print(prediction)
        return dspy.Prediction(output=prediction) 