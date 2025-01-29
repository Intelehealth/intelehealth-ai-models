import dspy

from signatures.DDxMulQnSignature import DDxMulFields

class DDxMulModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(DDxMulFields)

    def forward(self, case, question):
        prediction = self.generate_answer(case=case, question=question)
        print(prediction)
        return dspy.Prediction(output=prediction)