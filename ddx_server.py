import dspy
import time
import json
from utils.metric_utils import load_gemini_lm_prod, load_open_ai_lm, load_gemini_lm, load_gemini2_lm, load_gemini2_5_lm
from dotenv import load_dotenv
from modules.DDxModule import DDxModule
from modules.DDxMulModule import DDxMulModule
from modules.TelemedicineDDxModule import TelemedicineDDxModule
from google import genai
import os
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prompt_config import prompt_config
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Custom JSON Formatter for structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
        }
        if hasattr(record, 'extra_data'):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

class CustomLogger:
    def __init__(self, name='ddx_logger', log_file='ddx.log', max_bytes=10485760, backup_count=5):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_file)

        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        console_handler = logging.StreamHandler()

        formatter = JSONFormatter()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)

logger = CustomLogger()

load_dotenv(
    "ops/.env"
)

# MLflow setup for tracking DSPy calls
logger.info("Setting up MLflow for DSPy tracking...")
# Set tracking URI to local file store with database backend
import os
os.makedirs("mlflow_data", exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlflow_data/mlflow.db')}")
mlflow.set_experiment("ddx-server-tracking")
mlflow.dspy.autolog(
    log_traces=True,
    log_traces_from_compile=True,
    log_traces_from_eval=True,
    log_compiles=True,
    log_evals=True,
    silent=False
)
logger.info("MLflow DSPy autologging configured successfully!")

load_gemini2_lm()

# Configure OpenAI API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

class BaseDDxRequest(BaseModel):
    case: str
    model_name: str
    prompt_version: int
    tracker: str

class DDxRequestV1(BaseDDxRequest):
    pass

class DDxRequestV2(BaseDDxRequest):
    pass


async def transform_diagnosis_output(llm_output: dict) -> dict:
    """Transform the LLM output into the desired format using another LLM call."""
    transform_prompt = f"""Given the following diagnosis output from a medical LLM, transform it into a structured format.
    The input is:
    {str(llm_output)}

    Transform this into a response with the following structure:
    {{
        "result": [
            {{
                "diagnosis": "diagnosis name",
                "summarised_rationale":
                [
                    {{
                        "1": "summarised rationale line for the diagnosis in a crisp format from the rationale field limited to 1 line without losing relevance."
                    }},
                    {{
                        "2": "summarised rationale line for the diagnosis in a crisp format from the rationale field limited to 1 line without losing relevance."
                    }},
                    {{
                        "3": "summarised rationale line for the diagnosis in a crisp format from the rationale field limited to 1 line without losing relevance."
                    }}
                ]
                "rationale": [
                    {{
                        "Clinical Relevance and Features": "detailed explanation of clinical features"
                    }},
                    {{
                        "Lack of Fit Reasoning": "explanation of why this diagnosis might not fit"
                    }},
                    {{
                        "Relevance to Rural India": "explanation of rural context if applicable"
                    }},
                    {{
                        "Clinical Relevance": "general medical explanation"
                    }}
                ],
                "likelihood": "likelihood score"
            }},
            ...
        ],
        "conclusion": "conclusion text",
        "further_questions": [
        {{
            1: "example question 1?"
        }},
        {{
            2: "example question 2?"

        }}]
    }}

    For each diagnosis in the input:
    - diagnosis: The name of the diagnosis
    - summarised_rationale: A list of exactly 3 rationale points maximum, each being a single crisp line that summarizes key aspects of the diagnosis from the detailed rationale
    - rationales: A list of rationale objects, each containing:
        * any of these fields below:
            - "Clinical Relevance and Features": Details about how symptoms match the diagnosis
            - "Lack of Fit Reasoning": Why certain symptoms or factors make the diagnosis less likely
            - "Relevance to Rural India": Context specific to rural healthcare settings
            - "Clinical Relevance": General medical explanation of the condition
        * each field will have the detailed explanation for that field
    - likelihood: The likelihood score (High, Moderate-High, Moderate, Low-Moderate, Low)
    
    IMPORTANT: The summarised_rationale field must contain at most 3 lines ensuring clinical relevance is not lost. Each line should be concise and capture the most important aspects of the diagnosis rationale.

    For the further_questions field break the string into a list of questions with each item in the list being a question.
    - each key is the question number
    - each value is the question

    Note: Break down any multi-line rationales into separate rationale objects by category. Include "Lack of Fit Reasoning" when the diagnosis is less likely or when there are factors that make it less probable.
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=transform_prompt,
        )
        # Log the transformation request to MLflow
        mlflow.log_param("transformation_model", "gemini-2.0-flash")
        mlflow.log_param("transformation_prompt_length", len(transform_prompt))
        
        # Log the transformation prompt as an artifact
        with open("transformation_prompt.txt", "w") as f:
            f.write(transform_prompt)
                # Log the raw transformation response
        with open("raw_transformation_response.txt", "w") as f:
            f.write(str(response))
        mlflow.log_text(transform_prompt, "transformation_prompt.txt")
        mlflow.log_text(str(response), "raw_transformation_response.txt")
        
        # Log transformation success metric
        mlflow.log_metric("transformation_success", 1)
        print("response: -----")
        print(response)
        print("-----")
        
        # Extract JSON from the markdown-formatted response
        response_text = response.text
        if "```json" in response_text:
            # Find the content between ```json and ```
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        else:
            # If no markdown formatting, try to find JSON between any backticks
            json_str = response_text.split("```")[1].strip() if "```" in response_text else response_text.strip()
            
        # Parse the extracted JSON
        transformed = json.loads(json_str)
        return transformed
    except Exception as e:
        print(f"Error transforming diagnosis output: {e}")
        # Fallback to a basic transformation if parsing fails
        return {
            "result": [],
            "conclusion": "Error processing diagnosis output. Please try again."
        }

@app.post("/predict/v1")
async def ddx_v1(request_body: DDxRequestV1):
    start_time = time.time()
    log_extra = {
        'extra_data': {
            'tracker': request_body.tracker,
            'model_name': request_body.model_name,
            'prompt_version': request_body.prompt_version
        }
    }
    logger.info("Starting prediction request...", extra=log_extra)

    # Start MLflow run to track this prediction
    with mlflow.start_run(run_name=f"ddx_prediction_{int(time.time())}"):
        # Log request parameters
        mlflow.log_param("model_name", request_body.model_name)
        mlflow.log_param("case_length", len(request_body.case))
        mlflow.log_param("prompt_version", request_body.prompt_version)
        mlflow.log_param("tracker", request_body.tracker)

        cot = None
        prompt = ""
        if request_body.model_name == "gemini-2.0-flash":
            cot = DDxModule()
            cot.load("outputs/" + "10_02_2025_ddx_gemini2_only_num_trials_20_ayu_data_top_k5_single_diagnosis.json")
            # Create a new, fully-configured LM instance.
            gemini_for_tracking = dspy.LM(
                "gemini/gemini-2.0-flash",
                api_key=GEMINI_API_KEY,
                max_tokens=10000,
                temperature=0.7,
                top_k=5
            )
            # Get the current DSPy configuration to extract LLM parameters
            current_lm = dspy.settings.lm
            if hasattr(current_lm, 'model'):
                mlflow.log_param("llm_model", current_lm.model)
            if hasattr(current_lm, 'max_tokens'):
                mlflow.log_param("llm_max_tokens", current_lm.max_tokens)
            if hasattr(current_lm, 'temperature'):
                mlflow.log_param("llm_temperature", current_lm.temperature)
            if hasattr(current_lm, 'top_k'):
                mlflow.log_param("llm_top_k", current_lm.top_k)
            else:
                # Fallback to default values if attributes not found
                mlflow.log_param("llm_model", "gemini/gemini-2.0-flash")
                mlflow.log_param("llm_max_tokens", 10000)
                mlflow.log_param("llm_temperature", 0.7)
                mlflow.log_param("llm_top_k", 5)

            # Directly assign the configured LM to the predictor within the loaded module.
            cot.generate_answer.llm = gemini_for_tracking

            prompt = prompt_config[1]
            mlflow.log_param("prompt_config_index", 1)
        else:
            raise HTTPException(status_code=400, detail="Invalid model name for v1")

        dspy_program = dspy.asyncify(cot)

        try:
            logger.info("About to call DSPy program...", extra=log_extra)
            print("prompt selected: ", prompt)
            # Log input data
            mlflow.log_text(request_body.case, "input_case.txt")
            mlflow.log_text(prompt, "prompt_used.txt")

            dspy_start_time = time.time()
            result = await dspy_program(case=request_body.case, question=prompt)
            dspy_latency = time.time() - dspy_start_time
            log_extra['extra_data']['dspy_latency'] = dspy_latency
            logger.info(f"DSPy program completed in {dspy_latency:.2f} seconds", extra=log_extra)
            mlflow.log_metric("dspy_latency", dspy_latency)
            print(result)

            # Log the raw result
            mlflow.log_text(str(result), "raw_dspy_output.txt")

            if hasattr(result, 'output') and hasattr(result.output, 'diagnosis') and result.output.diagnosis == "NA":
                print("no diagnosis possible")
                response_data = {
                    "status": "success",
                    "data": "The Input provided does not have enough clinical details for AI based assessment."
                }
                mlflow.log_text(str(response_data), "final_response.txt")
                return response_data

            # Transform the diagnosis output
            transform_start_time = time.time()
            transformed_output = await transform_diagnosis_output(result.toDict())
            transform_latency = time.time() - transform_start_time
            log_extra['extra_data']['transformation_latency'] = transform_latency
            logger.info(f"Transformation completed in {transform_latency:.2f} seconds", extra=log_extra)
            mlflow.log_metric("transformation_latency", transform_latency)

            # Log the transformed output
            mlflow.log_text(json.dumps(transformed_output, indent=2), "transformed_output.json")

            response_data = {
                "status": "success",
                "data": transformed_output
            }

            # Log final response
            mlflow.log_text(json.dumps(response_data, indent=2), "final_response.json")
            mlflow.log_metric("prediction_success", 1)

            total_latency = time.time() - start_time
            mlflow.log_metric("total_latency", total_latency)
            log_extra['extra_data']['total_latency'] = total_latency
            logger.info(f"Prediction successful. Total latency: {total_latency:.2f} seconds", extra=log_extra)

            return response_data

        except Exception as e:
            logger.error(f"Error in prediction: {e}", extra=log_extra)
            mlflow.log_param("error", str(e))
            mlflow.log_metric("prediction_success", 0)
            print(e)
            raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")


@app.post("/predict/v2")
async def ddx_v2(request_body: DDxRequestV2):
    cot = None
    prompt = ""
    if request_body.model_name == "gemini-2.0-flash-001":
        cot = TelemedicineDDxModule()
        cot.load("outputs/" + "19_03_2025_21_31_ddx_gemini_cot_ayu_cleaned_data_llm_judge.json")
        prompt = prompt_config[2]
    else:
        raise HTTPException(status_code=400, detail="Invalid model name for v2")

    dspy_program = dspy.asyncify(cot)

    try:
        print("prompt selected: ", prompt)
        result = await dspy_program(case=request_body.case, question=prompt)
        print(result)
        if hasattr(result, 'output') and hasattr(result.output, 'diagnosis') and result.output.diagnosis == "NA":
            print("no diagnosis possible")
            return {
                "status": "success",
                "data": "The Input provided does not have enough clinical details for AI based assessment."
            }

        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")


@app.get("/health-status")
async def health_status():
    return {
        "status": "AVAILABLE",
        "description": "Service status for DDX server"
    }