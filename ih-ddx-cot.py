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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

trainset = prepare_ddx_data.ret_training_examples()
print(trainset[:2])
random.shuffle(trainset)




lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=1.0)

dspy.configure(lm=lm)


# metric has to be turned to refined manual function here
tp = dspy.MIPROv2(metric=openai_llm_judge, num_threads=10)
optimizedcot = tp.compile(DDxModule(), trainset=trainset)

optimizedcot.save("outputs/" + "ddx_open_ai_gpt-01_cot_trial1_llm_judge_metric.json")
