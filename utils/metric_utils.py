import openai
import os
import json
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import time
import dspy
import litellm

litellm.drop_params = True

load_dotenv(
    "ops/.env"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MEDLM_API_BASE = os.getenv("MEDLM_API_BASE")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
HYPERBOLIC_API_BASE = os.getenv("HYPERBOLIC_API_BASE")
MEDLM_PROJECT_JSON = os.getenv("MEDLM_PROJECT_JSON")
VERTEXAI_PROJECT = os.getenv("VERTEXAI_PROJECT")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

client = OpenAI(
  api_key = OPENAI_API_KEY,
  organization=OPENAI_ORG_ID,
  project=OPENAI_PROJECT_ID,
)


class TTxResponse(BaseModel):
    score: int
    rationale: str

class DDxResponse(BaseModel):
    score: int
    rationale: str

class GDdxResponse(BaseModel):
    score: float
    rationale: str

def metric_fun(gold, pred, trace=None):
    print(gold.diagnosis)
    print(pred.diagnosis)

    gold_d = gold.diagnosis.lower().strip(":")
    pred_d = pred.diagnosis.lower()

    if gold_d == pred_d:
        return 1.0
    elif gold_d in pred_d or pred_d in gold_d:
        return 1.0
    else:
        return 0.0

import sys


def gemini_llm_judge(gold, pred, trace=None):
    print("############## evaluating gemini llm judge ###############")
    print(gold.diagnosis)
    pred_diagnosis = pred.output
    print(pred_diagnosis)

    print("\n")
    messages = [
        {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for given case history of a patient for a doctor in rural India."},
        {"role": "user", "content": f"Expected output: {gold.diagnosis}\nPredicted output: {str(pred_diagnosis)}\n\nEvaluate the semantic similarity between the predicted and expected outputs. Consider the following:\n1. Is the expected diagnosis present in the top 5 diagnoses predicted?\n2. Is the core meaning preserved even if the wording differs from medical terminologies and synonyms for the matching expected and predicted diagnosis?\n3. Are there any significant omissions or additions in the predicted output?\n\nProvide output as valid JSON with field `score` as '1' for similar and '0' for not similar and field `rationale` having the reasoning string for this score."}
    ]

    gemini = dspy.Google("models/gemini-1.5-pro", api_key=GEMINI_API_KEY)
    dspy.settings.configure(lm=gemini, max_tokens=10000, temperature=0.1)
    response = gemini.complete(messages[-1]["content"])

    try:
        content = json.loads(response)
        score = content['score']
        rationale = content['rationale']
        print("Response from llm:")
        print("LLM Judge score: ", score)
        print("Rationale: ", rationale)
        return score
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
        return 0

def openai_llm_ttx_v2_judge(gold, pred, trace=None):
    print("############## evaluating open ai llm judge ###############")
    print(gold)
    print(pred)
    print("--------------------------------")
    print("\n")
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."}, 
            {"role": "user", "content": f"Expected output: " + str(gold)},
            {"role": "user", "content": f"Predicted output: " + str(pred.output.medication_recommendations)},
            {"role": "user", "content": """Evaluate the semantic similarity between the predicted and expected outputs. Consider the following:
            1. Is the expected medication present in the top 5 medications predicted. Consider semantic similarity for medication names?
            2. Is the strength, route form, dosage, frequency, number of days to take the medication and the reason for the medication relevant to the patient history and also matching expected output?
            3. Are there any significant omissions or additions in the predicted output?

            Provide output as valid JSON with field `score` as '1' for similar and '0' for not similar and field `rationale` having the reasoning string for this score."""}
        ],
        response_format = TTxResponse
    )
    # print(response)
    # Extract the content from the first choice
    # content = response.choices[0].message.content
    content = response.choices[0].message.parsed
    print("Response from llm:")
    print("LLM Judge score: ", content.score)
    score = content.score
    rationale = content.rationale
    print("Rationale: ", content.rationale)

    # time.sleep(2)
    return score

def openai_llm_ttx_judge(gold, pred, trace=None):
    print("############## evaluating open ai llm judge ###############")
    print(gold.medications_gt)
    print(pred.output.medications)
    print("--------------------------------")
    print("\n")
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."}, 
            {"role": "user", "content": f"Expected output: " + gold.medications_gt},
            {"role": "user", "content": f"Predicted output: " + str(pred.output.medications)},
            {"role": "user", "content": """Evaluate the semantic similarity between the predicted and expected outputs. Consider the following:
            1. Is the expected medication present in the top 5 medications predicted. Consider semantic similarity for medication names?
            2. Is the rationale for medications relevant to the patient history.
            3. Are there any significant omissions or additions in the predicted output?

            Provide output as valid JSON with field `score` as '1' for similar and '0' for not similar and field `rationale` having the reasoning string for this score."""}
        ],
        response_format = TTxResponse
    )
    # print(response)
    # Extract the content from the first choice
    # content = response.choices[0].message.content
    content = response.choices[0].message.parsed
    print("Response from llm:")
    print("LLM Judge score: ", content.score)
    score = content.score
    rationale = content.rationale
    print("Rationale: ", content.rationale)

    # time.sleep(2)
    return score

def openai_llm_reasoning_judge(gold, pred, trace=None):
    print("############## evaluating open ai llm judge ###############")
    print(gold.diagnosis)
    pred_diagnosis = pred
    print(pred_diagnosis)


    print("\n")
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."},
            {"role": "user", "content": f"Expected output: " + gold.diagnosis},
            {"role": "user", "content": f"Predicted output: " + str(pred_diagnosis) },
            {"role": "user", "content": """Evaluate the semantic similarity between the predicted and expected outputs. Consider the following: 
             1. Is the expected diagnosis present in the top 5 diagnosises predicted?
             2. Is the core meaning preserved even if the wording differs from medical terminologies and synonyms for the matching expected and predicted diagnosis?
             3. Are there any significant omissions or additions in the predicted output?

             Provide output as valid JSON with field `score` as '1' for similar and '0' for not similar and field `rationale` having the reasoning string for this score."""}
        ],
        response_format = DDxResponse
    )
    # print(response)
    # Extract the content from the first choice
    # content = response.choices[0].message.content
    content = response.choices[0].message.parsed

    print("Response from llm:")
    print("LLM Judge score: ", content.score)
    score = content.score
    rationale = content.rationale
    print("Rationale: ", content.rationale)

    return score

def openai_llm_judge(gold, pred, trace=None):
    print("############## evaluating open ai llm judge ###############")
    print(gold.diagnosis)
    pred_diagnosis = pred.output
    print(pred_diagnosis)


    print("\n")
    response = client.beta.chat.completions.parse(
        
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."},   
            {"role": "user", "content": f"Expected output: " + gold.diagnosis},
            {"role": "user", "content": f"Predicted output: " + str(pred_diagnosis) },
            {"role": "user", "content": """Evaluate the semantic similarity between the predicted and expected outputs. Consider the following: 
             1. Is the expected diagnosis present in the top 5 diagnosises predicted?
             2. Is the core meaning preserved even if the wording differs from medical terminologies and synonyms for the matching expected and predicted diagnosis?
             3. Are there any significant omissions or additions in the predicted output?
             
             Provide output as valid JSON with field `score` as '1' for similar and '0' for not similar and field `rationale` having the reasoning string for this score."""}
        ],
        response_format = DDxResponse
    )
    # print(response)
    # Extract the content from the first choice
    # content = response.choices[0].message.content
    content = response.choices[0].message.parsed

    print("Response from llm:")
    print("LLM Judge score: ", content.score)
    score = content.score
    rationale = content.rationale
    print("Rationale: ", content.rationale)
    

    # time.sleep(2)

    return score

def openai_qwen_local_llm_judge(gold, pred, trace=None):
    print("############## evaluating open ai llm judge ###############")
    print(gold.differential_diagnoses)
    pred_diagnosis = pred
    print(pred_diagnosis)


    print("\n")
    response = client.beta.chat.completions.parse(
        
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that helps in evaluating the similarity between two diagnosis for qiven case history of a patient for a doctor in rural India."},   
            {"role": "user", "content": f"Expected output: " + gold.differential_diagnoses},
            {"role": "user", "content": f"Predicted output: " + str(pred_diagnosis) },
            {"role": "user", "content": """Evaluate the semantic similarity between the predicted and expected outputs. Consider the following: 
             1. Is the expected diagnosis present in the top 5 diagnosises predicted?
             2. Is the core meaning preserved even if the wording differs from medical terminologies and synonyms for the matching expected and predicted diagnosis?
             3. Are there any significant omissions or additions in the predicted output?
             
             Provide output as valid JSON with field `score` as '1' for similar and '0' for not similar and field `rationale` having the reasoning string for this score."""}
        ],
        response_format = DDxResponse
    )
    # print(response)
    # Extract the content from the first choice
    # content = response.choices[0].message.content
    content = response.choices[0].message.parsed

    print("Response from llm:")
    print("LLM Judge score: ", content.score)
    score = content.score
    rationale = content.rationale
    print("Rationale: ", content.rationale)
    

    # time.sleep(2)

    return score

def load_groq_llama_3_1():
    llama = dspy.GROQ(model="llama-3.1-8b-instant", api_key = GROQ_API_KEY)
    dspy.settings.configure(lm=llama, max_tokens=10000, top_k=3)

def load_hyperbolic_llama_3_3_70b_instruct():
    lm = dspy.LM('openai/meta-llama/Llama-3.2-11B-Vision-Instruct', api_key=HYPERBOLIC_API_KEY, api_base=HYPERBOLIC_API_BASE)
    dspy.configure(lm=lm, max_tokens=10000, top_k=5)

def load_gemini_lm():
    gemini = dspy.Google("models/gemini-1.5-pro", api_key=GEMINI_API_KEY,)
    dspy.settings.configure(lm=gemini, max_tokens=10000, top_k=5)


def load_gemini_lm_prod():
    gemini = dspy.Google("models/gemini-1.5-pro", api_key=GEMINI_API_KEY, temperature=0.01)
    dspy.settings.configure(lm=gemini, max_tokens=10000, top_k=5)
    # dspy.settings.configure(lm=gemini, top_k=5)

def load_gemini2_lm():
    gemini = dspy.Google("models/gemini-2.0-flash", api_key=GEMINI_API_KEY)
    dspy.settings.configure(lm=gemini, max_tokens=10000, top_k=5)

def load_gemini2_5_lm():
    gemini = dspy.Google("models/gemini-2.5-flash-preview-04-17", api_key=GEMINI_API_KEY)
    dspy.settings.configure(lm=gemini, max_tokens=50000, top_k=5)

def load_gemini2_5_lm_1():
    gemini = dspy.LM("openai/gemini-2.5-flash-preview-04-17", api_key=GEMINI_API_KEY,
                     api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                     max_tokens=50000)
    dspy.settings.configure(lm=gemini, top_k=5)

def load_gemini_2_5_pro_lm():
    gemini = dspy.LM("openai/gemini-2.5-pro-preview-05-06", api_key=GEMINI_API_KEY, api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                     max_tokens=50000)
    dspy.settings.configure(lm=gemini, top_k=5)

def load_gemini_2_5_vertex_lm():
    gemini = dspy.Google("models/gemini-2.5-flash-preview-04-17", api_key=GEMINI_API_KEY)

    # gemini = dspy.Google(
    #             model='vertex_ai/gemini-2.5-flash-preview-04-17"',
    #             api_key=GEMINI_API_KEY,
    #             vertex_project=VERTEXAI_PROJECT,
    #             vertex_location="us-central1",
    #             vertex_credentials=MEDLM_PROJECT_JSON)
    dspy.settings.configure(lm=gemini, temperature=0.1)

def load_gemini_vertex_finetuned_lm():
    # gemini = dspy.LM(
    #             model="openai/top1-ddx",
    #             base_model="vertex_ai/gemini-2.0-flash",
    #             api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    #             api_key=GEMINI_API_KEY,
    #             vertex_project=VERTEXAI_PROJECT,
    #             vertex_location="us-central1",
    #              vertex_credentials=MEDLM_PROJECT_JSON)
    # gemini = dspy.GoogleVertexAI(
    #         #   model_name=MODEL_ENDPOINT,
    #           model_name="vertex_ai/top1-ddx",
    #           api_key=GEMINI_API_KEY,
    #           project=VERTEXAI_PROJECT,
    #           location="us-central1",  # Match deployment region
    #           credentials=MEDLM_PROJECT_JSON)
    
    gemini = dspy.LM(
            model="vertex_ai/gemini/top1-ddx",
            base_model="vertex_ai/gemini/gemini-2.0-flash",
            api_key=GEMINI_API_KEY,
            project=VERTEXAI_PROJECT,
            location="us-central1",  # Match deployment region
            vertex_credentials=MEDLM_PROJECT_JSON)
    #
    # gemini = dspy.GoogleVertexAI(
    #     model="gemini-2.0-flash",
    #     project=VERTEXAI_PROJECT,
    #     location="us-central1",
    #     credentials=MEDLM_PROJECT_JSON
    #     )
    dspy.configure(lm=gemini, max_tokens=8192, temperature=1.0, top_k=5)

def load_gemini_vertexai_lm():
    print("loading gemini vertex ai")
    try:
        gemini = dspy.LM('vertex_ai/gemini-1.5-pro', api_key=GEMINI_API_KEY, api_base=MEDLM_API_BASE,
                 vertex_credentials=MEDLM_PROJECT_JSON)
        # gemini = dspy.GoogleVertexAI(
        #     model=MEDLM_MODEL,
        #     project=MEDLM_PROJECT_ID,
        #     location="us-central1",
        #     credentials=MEDLM_PROJECT_JSON
        #     )
        dspy.settings.configure(lm=gemini, max_tokens=10000, temperature=1.0, top_k=7)
    except:
        print("cannot load")

def load_open_ai_lm():
    lm = dspy.LM('openai/gpt-4o', api_key=OPENAI_API_KEY, temperature=1.0)
    dspy.configure(lm=lm, top_k=5)

def load_open_ai_lm_4_1():
    lm = dspy.LM('openai/gpt-4.1', api_key=OPENAI_API_KEY, temperature=1.0)
    dspy.configure(lm=lm, top_k=5)

def load_open_ai_o1_mini_lm():
    lm = dspy.LM('o1-mini', api_key=OPENAI_API_KEY, max_tokens=8000, temperature=1.0)
    dspy.configure(lm=lm,  top_k=5)

def load_vertex_ai_url_lm():
    lm = dspy.LM('vertex_ai/gemini-1.5-pro', api_key=GEMINI_API_KEY, api_base=MEDLM_API_BASE,
                 vertex_credentials=MEDLM_PROJECT_JSON)
    dspy.configure(lm=lm)

def load_ollama_openbio_70b_llm_url():
    lm = dspy.LM('ollama_chat/taozhiyuai/openbiollm-llama-3:70b_q2_k', api_base='http://localhost:11434', api_key='', model_type="chat", temperature=1.0, stop=["## completed ##"])
    dspy.configure(lm=lm, top_k=5)


def load_ollama_meditron_70b_url():
    lm = dspy.LM('ollama_chat/meditron:70b', api_base='http://localhost:11434', api_key='', model_type="chat", temperature=1.0)

    dspy.configure(lm=lm, top_k=5)


def load_ollama_meditron_7b_url():
    lm = dspy.LM('ollama_chat/meditron:latest', api_base='http://localhost:11434', api_key='', model_type="chat", temperature=1.0)

    dspy.configure(lm=lm, top_k=5)


def load_ollama_deepseek_70b_llm_url():
    lm = dspy.LM('ollama_chat/deepseek-r1:70b', api_base='http://localhost:11434', api_key='', model_type="chat", temperature=1.0)
    dspy.configure(lm=lm, top_k=5)

def load_ollama_deepseek_r1_70b_llm_url_non_chat():
    lm = dspy.LM('ollama_chat/deepseek-r1:70b', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm, top_k=5)


def load_ollama_deepseek_r1_32b_llm_url_non_chat():
    lm = dspy.LM('ollama_chat/deepseek-r1:32b', api_base='http://localhost:11434', model_type="chat", api_key='', stop=["## completed ##"])
    dspy.configure(lm=lm, top_k=5)
    # return lm


def load_openAI_direct_client():    # Initialize OpenAI client that points to the local LM Studio server
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )

    return client


def load_lm_studio_medllama3_v20_url():
    lm = dspy.LM('lm_studio/probemedicalyonseimailab-medllama3-v20', api_base='http://localhost:1234/v1', api_key='', model_type="chat", temperature=1.0)
    dspy.configure(lm=lm, top_k=5)

def load_lm_studio_deepseek_r1_qwen_distil_32b():
    lm = dspy.LM('lm_studio/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF', api_base='http://localhost:1234/v1', api_key='', model_type="chat", temperature=1.0)
    dspy.configure(lm=lm, top_k=5)

def load_lm_studio_qwen_qwq_32b():
    lm = dspy.LM('lm_studio/Qwen/QwQ-32B-GGUF', api_base='http://localhost:1234/v1', api_key='', model_type="chat", temperature=1.0)
    dspy.configure(lm=lm, top_k=5)

def load_lm_studio_qwen_merged_3b_grpo():
    lm = dspy.LM('lm_studio/Qwen2.5-3B-Instruct-Qlora-GGUF', api_base='http://localhost:1234/v1', api_key='', model_type="chat", temperature=1.0)
    dspy.configure(lm=lm, top_k=5)

def load_lm_studio_medgemma_27b_text_it():
    lm = dspy.LM(model="openai/google/medgemma-27b-text-it", api_base='http://localhost:1234/v1/', api_key="local", model_type="chat", temperature=1.0)
    dspy.configure(lm=lm, top_k=5)

def load_aws_bedrock_lm():
    """Configures DSPy to use AWS Bedrock with credentials from environment variables."""
    if not BEDROCK_MODEL_ID:
        raise ValueError("BEDROCK_MODEL_ID environment variable not set.")
    # litellm, used by dspy.LM, will automatically pick up AWS credentials
    # from environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME)
    # or from ~/.aws/credentials if they are set.

    bedrock_lm = dspy.LM(
        model=f"bedrock/{BEDROCK_MODEL_ID}"
    )
    dspy.settings.configure(lm=bedrock_lm, max_tokens=4096, temperature=1.0)
    print(f"DSPy configured to use AWS Bedrock model: {BEDROCK_MODEL_ID}")
