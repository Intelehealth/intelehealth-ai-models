import json

multiple_diagnosis_diff_schema =  {
    "type": "json_schema",
    "json_schema": {
        "name": "mulitple_diagnosis_with_rationale",
        "schema": {
            "type": "object",
            "properties": {
                "primary_diagnosis": {
                    "type": "string",
                        "properties": {
                            "disease": {"type": "string"},
                            "rationale": {"type": "string"}
                        },
                    "minItems": 1
                },
                "diagnoses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "disease": {"type": "string"},
                            "rationale": {"type": "string"}
                        },
                         "required": ["disease", "rationale"]
                    },
                    "maxItems": 4
                },
                "further_questions" : {
                    "type": "string"
                }
            },
            "required": ["diagnoses", "primary_diagnosis", "further_questions"]
        }
    }
}

user_prompt = "what are the top 5 differential diagnosis for indian doctor in rural setting for the case given by user. Order by likelhiood of each diagnosis from top to bottom."

# Get response from AI
def lm_studio_openai_client_infer(client, case):
    messages = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": user_prompt + case

            }
        ]
    response = client.chat.completions.create(
            model="TheBloke/meditron-70B-GGUF",
            messages=messages,
            response_format=multiple_diagnosis_diff_schema,
            max_tokens=-1,
            temperature=0.2 # Add temperature here
        )
    return json.loads(response.choices[0].message.content)