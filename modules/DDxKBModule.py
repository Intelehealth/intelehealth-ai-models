import dspy

from signatures.DDxKBSignature import DDxKBSignature

class DDxKBModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(DDxKBSignature)

    def forward(self, case, question):
        print("Question in forward:")
        print(question)
        prediction = self.generate_answer(case=case, question=question)
        print(prediction)
        
        return dspy.Prediction(output=prediction) 