import dspy
from utils import prepare_ttx_data # Changed from prepare_ddx_data
from utils import prepare_ttx_data_v2
from utils.metric_utils import openai_llm_ttx_judge, openai_llm_ttx_v2_judge, load_gemini2_lm, load_gemini2_5_lm, load_gemini2_5_lm_1
import os
import random
from dotenv import load_dotenv

from modules.TTxModule import TTxModule
from modules.TTxv2Module import TTxv2Module
from modules.TTxv3Module import TTxv3Module

load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#trainset = prepare_ttx_data.ret_ttx_examples() # Changed from prepare_ddx_data
trainset_v2 = prepare_ttx_data_v2.ret_ttx_v2_examples()
print(trainset_v2[:2])
random.shuffle(trainset_v2)


import argparse
from datetime import datetime

# Set up argument parser
parser = argparse.ArgumentParser(description='Run treatment recommendation (TTx) with specified parameters') # Updated description
parser.add_argument('--llm', type=str, choices=['openai', 'gemini', 'gemini2', 'gemini2_5'], required=True,
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
# Get current datetime for filename
current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M")

# Create output filename based on parameters
output_filename = f"{current_datetime}_ttx_v3_{args.llm}_cot_nas_v2_combined_medications.json" # Updated filename format

# Update number of trials based on argument
tp = dspy.MIPROv2(metric=openai_llm_ttx_v2_judge, num_threads=4, num_candidates=4, max_labeled_demos=4)
optimizedcot = tp.compile(TTxv3Module(), trainset=trainset_v2, num_trials=args.num_trials) # Use TTxModule

optimizedcot.save("outputs/" + output_filename) 