
from google import genai
import os, sys

from dotenv import load_dotenv

load_dotenv(
    "ops/.env"
)
# Create a client

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
 
# Define the model you are going to use
model_id =  "gemini-2.0-flash" # or "gemini-2.0-flash-lite-preview-02-05"  , "gemini-2.0-pro-exp-02-05"

file_path = "./data/standard-treatment-guidelines.pdf"
stg_pdf = client.files.upload(file=file_path, config={'display_name': 'STG'})

file_size = client.models.count_tokens(model=model_id,contents=stg_pdf)


subtopics_by_topic = {
    # "Common Conditions": ['   Acute fever', '   Fever of unknown origin', '   Acute rheumatic fever', '   Anaemia', '   Typhoid fever', '   Malaria', '   Dengue', '   Chikungunya', '   Tuberculosis and RNTCP', '   Epilepsy', '   Status epilepticus', '   Urinary tract infection'],
    "Emergency Conditions": ['   Cardiopulmonary resuscitation', '   Anaphylaxis', '   Acute airway obstruction', '   Stridor', '   Shock', '   Fluid and electrolyte imbalance', '   Septicemia', '   Organophosphorus poisoning', '   Kerosene and petrol poisoning', '   Datura poisoning', '   Abdominal injury', '   Foreign body in ear', '   Chemical injuries of the eye', '   Corneal and conjunctival foreign bodies', '   Traumatic hyphema', '   Animal bites- Dog bite', '   Snake bite', '   Insect and Arachnid bites', '   Scorpion bite']
}

chapter = 2

section_prompt = """
For this topic - {} in the chapter {}, extract all the text for subtopic {}, organized by each sub section heading into json.
Json should also contain description of the subtopic {}. Include references in the sectin if any also in the json. 
so if there is a table in the section, store the table information within tables field in json with rows and columns of each table as list. 
"""
for topic, subtopics in subtopics_by_topic.items():
    for subtopic in subtopics:
        formatted_section_prompt = section_prompt.format(topic, chapter, subtopic, subtopic)
        try:
            heading_response = client.models.generate_content(
                model=model_id, 
                contents=[formatted_section_prompt, stg_pdf]
            )
            heading_text = "".join([part.text for part in heading_response.candidates[0].content.parts])
            print(f"Section headings for sub-topic '{subtopic}' under topic '{topic}':", heading_text)
            # Create directory if it doesn't exist
            output_dir = f"data/stg/chapters/2/sections/{topic.strip().replace(' ', '_')}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Clean filename by removing invalid characters and spaces
            clean_subtopic = subtopic.strip().replace('/', '_').replace(' ', '_')
            filename = f"{output_dir}/{clean_subtopic}.json"
            
            try:
                # Try to parse the text as JSON first
                import json
                # Strip any "```json" or "```" markers from the text
                heading_text = heading_text.strip()
                if heading_text.startswith('```json'):
                    heading_text = heading_text[7:]
                elif heading_text.startswith('```'):
                    heading_text = heading_text[3:]
                if heading_text.endswith('```'):
                    heading_text = heading_text[:-3]
                heading_text = heading_text.strip()
                json_data = json.loads(heading_text)
                
                # Write formatted JSON to file
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
                print(f"Successfully saved JSON to {filename}")
                
            except json.JSONDecodeError:
                # If parsing fails, wrap the text in a basic JSON structure
                json_data = {
                    "topic": topic,
                    "subtopic": subtopic,
                    "raw_content": heading_text
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
                print(f"Saved raw content as JSON to {filename}")
        except Exception as e:
            print(f"Error processing sub-topic '{subtopic}' under topic '{topic}': {str(e)}")
