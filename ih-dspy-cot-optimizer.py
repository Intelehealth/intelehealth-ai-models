import dspy
from utils import prepare_ddx_data
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

trainset = prepare_ddx_data.ret_training_examples()
print(trainset[:2])

class DDxFields(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="output the top diagnosis")

lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=0.5)

dspy.configure(lm=lm)

cot = dspy.ChainOfThought(DDxFields)

# metric has to be turned to refined manual function here
tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=5)
optimized_react = tp.compile(cot, trainset=trainset[:30])

optimized_react.save("outputs/" + "open_ai_gpt4o_cot_trial_1.json")
