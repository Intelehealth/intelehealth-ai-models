import dspy
from utils import prepare_ddx_data
from utils.metric_utils import openai_llm_judge, openai_llm_reasoning_judge, load_gemini2_lm, load_gemini2_5_lm, load_gemini_vertex_finetuned_lm, load_gemini_2_5_vertex_lm, load_gemini2_5_lm_1
import os
import random
from dotenv import load_dotenv

from modules.DDxModule import DDxModule
from modules.DDxLocalModule import DDxLocalModule
from modules.TelemedicineDDxModule import TelemedicineDDxModule
from modules.TelemedicineTenDDxModule import TelemedicineTenDDxModule
from modules.DDxKBModule import DDxKBModule

load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

trainset = prepare_ddx_data.ret_training_examples()
print(trainset[:2])
random.shuffle(trainset)


# #load_open_ai_lm()
# #load_open_ai_o1_mini_lm()
# load_ollama_openbio_70b_llm_url()

# tp = dspy.MIPROv2(metric=openai_llm_judge, num_threads=4, num_candidates=4, max_labeled_demos=4)
# tp = dspy.MIPROv2(metric=openai_llm_judge, num_threads=4)


import argparse
from datetime import datetime

# Set up argument parser
parser = argparse.ArgumentParser(description='Run differential diagnosis with specified parameters')
parser.add_argument('--llm', type=str, choices=['openai', 'gemini', 'gemini2', 'gemini_vertex_finetuned' , 'gemini_2_5_flash_vertex'], required=True,
                   help='LLM to use (openai or gemini)')
parser.add_argument('--num_trials', type=int, default=2,
                   help='Number of trials to run (default: 2)')
args = parser.parse_args()

if args.llm == 'gemini':
    load_gemini2_lm()
elif args.llm == 'gemini2':
    load_gemini2_5_lm()
elif args.llm == 'gemini2_5':
    load_gemini2_5_lm_1()
elif args.llm == 'gemini_vertex_finetuned':
    load_gemini_vertex_finetuned_lm()
elif args.llm == 'gemini_2_5_flash_vertex':
    load_gemini_2_5_vertex_lm()

# Get current datetime for filename
current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M")

# Create output filename based on parameters
output_filename = f"{current_datetime}_ddx_{args.llm}_cot_ayu_cleaned_data_llm_judge.json"

# Update number of trials based on argument
tp = dspy.MIPROv2(metric=openai_llm_reasoning_judge, num_threads=4, num_candidates=4, max_labeled_demos=4)
optimizedcot = tp.compile(TelemedicineDDxModule(), trainset=trainset, num_trials=args.num_trials)

optimizedcot.save("outputs/" + output_filename)
