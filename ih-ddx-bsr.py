import dspy
from utils import prepare_ddx_data
from utils.metric_utils import metric_fun, openai_llm_judge, load_open_ai_lm, load_open_ai_o1_mini_lm, load_ollama_url
import os
import random
from dotenv import load_dotenv

from modules.DDxModule import DDxModule
from modules.DDxLocalModule import DDxLocalModule

load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

trainset = prepare_ddx_data.ret_training_examples()
print(trainset[:2])
random.shuffle(trainset)

#load_open_ai_lm()
#load_open_ai_o1_mini_lm()
load_ollama_url()

# tp = dspy.MIPROv2(metric=openai_llm_judge, num_threads=4, num_candidates=4, max_labeled_demos=4)
tp = dspy.BootstrapFewShotWithRandomSearch(metric=openai_llm_judge,
                                           max_bootstrapped_demos=2,
                                           max_labeled_demos=2,
                                           num_threads=1,
                                           num_candidate_programs=2)

optimizedcot = tp.compile(DDxLocalModule(), trainset=trainset[:20])

optimizedcot.save("outputs/" + "23_01_2025_ddx_open_ai_openbiollm_ollama70_bsr_patient_cleaned_data_llm_judge.json")
