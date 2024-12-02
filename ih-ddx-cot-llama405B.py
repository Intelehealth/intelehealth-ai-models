import os
import openai
import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge
import os
import random
from dotenv import load_dotenv

from modules.DDxModule import DDxModule

load_dotenv(
    "ops/.env"
)
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
HYPERBOLIC_API_BASE = os.getenv("HYPERBOLIC_API_BASE")

lm = dspy.LM('openai/meta-llama/Meta-Llama-3.1-405B-Instruct', api_key=HYPERBOLIC_API_KEY, api_base=HYPERBOLIC_API_BASE)
dspy.configure(lm=lm, max_tokens=10000)

trainset = prepare_ddx_data.ret_training_examples()
trainset = trainset[:50]
print(trainset[:2])
random.shuffle(trainset)


# metric has to be turned to refined manual function here
tp = dspy.MIPROv2(metric=openai_llm_judge, num_threads=5)
optimizedcot = tp.compile(DDxModule(), trainset=trainset, num_trials=5)

optimizedcot.save("outputs/" + "ddx_open_ai_llama405b_pro_cot_trial_cleaned_data_llm_judge_metric.json")
