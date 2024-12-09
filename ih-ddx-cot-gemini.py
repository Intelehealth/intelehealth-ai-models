import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge, load_gemini_lm
import os
import random
from dotenv import load_dotenv

from modules.DDxModule import DDxModule

load_dotenv(
    "ops/.env"
)

load_gemini_lm()

trainset = prepare_ddx_data.ret_training_examples()
print(trainset[:2])
random.shuffle(trainset[:50])


# metric has to be turned to refined manual function here
tp = dspy.MIPROv2(metric=openai_llm_judge, num_threads=5)
optimizedcot = tp.compile(DDxModule(), trainset=trainset, num_trials=5)

optimizedcot.save("outputs/" + "ddx_open_ai_gemini_pro_medpalm_cot_cleaned_data_llm_judge_metric.json")
