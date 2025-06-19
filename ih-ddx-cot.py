import dspy
from utils import prepare_ddx_data
from utils.metric_utils import openai_llm_reasoning_judge, load_gemini2_lm, load_gemini2_5_lm, load_gemini_vertex_finetuned_lm, \
      load_gemini_2_5_vertex_lm, load_gemini2_5_lm_1, load_open_ai_lm, load_open_ai_lm_4_1, load_lm_studio_medgemma_27b_text_it, \
        load_gemini_2_5_pro_lm, load_hyperbolic_llama_3_3_70b_instruct, load_aws_bedrock_lm
import os
import random
from dotenv import load_dotenv

from modules.DDxModule import DDxModule
from modules.DDxLocalModule import DDxLocalModule
from modules.TelemedicineDDxModule import TelemedicineDDxModule
from modules.TelemedicineTenDDxModule import TelemedicineTenDDxModule
from modules.DDxKBModule import DDxKBModule
from modules.TelemedicineDDxModuleRAG import TelemedicineDDxModuleRAG
from modules.TelemedicineICD11DDxModule import TelemedicineICD11DDxModule
from modules.TelemedicineSnomedCTDDxModule import TelemedicineSnomedCTDDxModule

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
parser.add_argument('--llm', type=str, choices=['openai', 'openai_4_1', 'gemini', 'gemini2', 'gemini_vertex_finetuned' , 'gemini_2_5_flash_vertex', 'medgemma_27b_text_it', 'gemini_2_5_pro', 'hyperbolic_llama_3_3_70b_instruct', 'aws_bedrock_llama_3_2_11b'], required=True,
                   help='LLM to use (openai or gemini)')
parser.add_argument('--num_trials', type=int, default=2,
                   help='Number of trials to run (default: 2)')
parser.add_argument('--module', type=str, choices=['TelemedicineDDxModule', 'DDxModule', 'DDxLocalModule', 'TelemedicineTenDDxModule', 'DDxKBModule', 'TelemedicineDDxModuleRAG', 'TelemedicineICD11DDxModule', 'TelemedicineSnomedCTDDxModule'], default='TelemedicineDDxModule',
                   help='Module to use for differential diagnosis (default: TelemedicineDDxModule)')
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
elif args.llm == 'openai':
    load_open_ai_lm()
elif args.llm == 'openai_4_1':
    load_open_ai_lm_4_1()
elif args.llm == 'medgemma_27b_text_it':
    load_lm_studio_medgemma_27b_text_it()
elif args.llm == 'gemini_2_5_pro':
    load_gemini_2_5_pro_lm()
elif args.llm == 'hyperbolic_llama_3_3_70b_instruct':
    load_hyperbolic_llama_3_3_70b_instruct()
elif args.llm == 'aws_bedrock_llama_3_2_11b':
    load_aws_bedrock_lm()

# Get current datetime for filename
current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M")

# Create output filename based on parameters
output_filename = f"{current_datetime}_ddx_{args.llm}_{args.module}_cot_ayu_cleaned_data_llm_judge.json"

# Update number of trials based on argument
tp = dspy.MIPROv2(metric=openai_llm_reasoning_judge, num_threads=4, num_candidates=4, max_labeled_demos=4)

# Instantiate the selected module
if args.module == 'DDxModule':
    module_to_compile = DDxModule()
elif args.module == 'DDxLocalModule':
    module_to_compile = DDxLocalModule()
elif args.module == 'TelemedicineTenDDxModule':
    module_to_compile = TelemedicineTenDDxModule()
elif args.module == 'DDxKBModule':
    module_to_compile = DDxKBModule()
elif args.module == 'TelemedicineDDxModuleRAG':
    module_to_compile = TelemedicineDDxModuleRAG()
elif args.module == 'TelemedicineICD11DDxModule':
    module_to_compile = TelemedicineICD11DDxModule()
elif args.module == 'TelemedicineSnomedCTDDxModule':
    module_to_compile = TelemedicineSnomedCTDDxModule()
else:  # Default to TelemedicineDDxModule
    module_to_compile = TelemedicineDDxModule()

optimizedcot = tp.compile(module_to_compile, trainset=trainset, num_trials=args.num_trials)

optimizedcot.save("outputs/" + output_filename)
