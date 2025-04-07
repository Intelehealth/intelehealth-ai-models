import dspy
class ClinicalNotesSignature(dspy.Signature):
    """Given clinical notes, provide the top 5 differential diagnoses."""
    clinical_notes = dspy.InputField(desc="Patient history, symptoms, physical exam findings, and demographics.")
    differential_diagnoses = dspy.OutputField(desc='Top 5 differential diagnoses ranked by likelihood. Respond in the following format: \nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>') # Added formatting hint
