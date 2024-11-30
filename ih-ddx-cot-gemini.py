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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini = dspy.Google("models/gemini-1.5-pro", api_key=GEMINI_API_KEY, temperature=1.0)
dspy.settings.configure(lm=gemini, max_tokens=10000)

trainset = prepare_ddx_data.ret_training_examples()
print(trainset[:2])
random.shuffle(trainset)


# metric has to be turned to refined manual function here
tp = dspy.MIPROv2(metric=openai_llm_judge, num_threads=5)
optimizedcot = tp.compile(DDxModule(), trainset=trainset, num_trials=5)

optimizedcot.save("outputs/" + "ddx_open_ai_gemini_pro_cot_trial_cleaned_data_llm_judge_metric.json")
