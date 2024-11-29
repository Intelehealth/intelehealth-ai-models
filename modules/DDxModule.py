import dspy

from signatures.DDxSignature import DDxFields

class DDxModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(DDxFields)

    def forward(self, case, question):
        prediction = self.generate_answer(case=case, question=question)
        print(prediction)
        return dspy.Prediction(output=prediction)