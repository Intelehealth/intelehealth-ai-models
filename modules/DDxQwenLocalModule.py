import dspy
from signatures.DDxQwenLocalSignature import ClinicalNotesSignature


class DDxQwenLocalModule(dspy.Module):
    """A simple dspy.Module for Differential Diagnosis."""
    def __init__(self):
        super().__init__()
        # Define the predictor using the signature
        # This will use the globally configured dspy.LM
        self.predictor = dspy.Predict(ClinicalNotesSignature)

    def forward(self, clinical_notes):
        """Runs the prediction."""
        prediction = self.predictor(clinical_notes=clinical_notes)
        return dspy.Prediction(differential_diagnoses=prediction) 