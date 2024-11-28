import dspy
from utils import prepare_ddx_data
import os

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini = dspy.Google("models/gemini-1.5-pro", api_key=GEMINI_API_KEY, temperature=0.5)
dspy.settings.configure(lm=gemini)


trainset = prepare_ddx_data.ret_training_examples()
print(trainset[:2])


def metric_fun(gold, pred, trace=None):
    print(gold.diagnosis)
    print(pred.diagnosis)

    gold_d = gold.diagnosis.lower()
    pred_d = pred.diagnosis.lower()

    if gold_d == pred_d:
        return 1.0
    elif gold_d in pred_d or pred_d in gold_d:
        return 1.0
    else:
        return 0.0

class DDxFields(dspy.Signature):
    """You are a doctor with the following patient rural India.
    Here is their case with the history of presenting illness, their physical exams, and demographics.
    What would be the top differential diagnosis for this patient?
    Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."""
    case = dspy.InputField(desc="case with the history of presenting illness, physical exams, and demographics of a patient")
    diagnosis = dspy.OutputField(desc="Topmost diagnosis for the patient")
    rationale = dspy.OutputField(desc="a detailed explanation for this diagnosis")


gemini = dspy.Google("models/gemini-1.5-pro", api_key=GEMINI_API_KEY, temperature=0.5)
dspy.settings.configure(lm=gemini)

cot = dspy.ChainOfThought(DDxFields)

tp = dspy.MIPROv2(metric=metric_fun, auto="light", num_threads=5)
optimizedcot = tp.compile(cot, trainset=trainset)

optimizedcot.save("outputs/" + "ddx_google_gemini_1.5_pro_cot_trial_5_lowercase_metric.json")
