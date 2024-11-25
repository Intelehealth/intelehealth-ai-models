import dspy
from utils import prepare_ddx_data
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

trainset = prepare_ddx_data.ret_training_examples()
print(trainset[:2])

lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=0.5)

dspy.configure(lm=lm)

react = dspy.ReAct("question -> answer", tools=[])

tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=5)
optimized_react = tp.compile(react, trainset=trainset[:20])

optimized_react.save("outputs/" + "open_ai_gpt4o_trial_1.json")
