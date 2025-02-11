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
print(f'File: {stg_pdf.display_name} equals to {file_size.total_tokens} tokens')


extract_topic_prompt = "From this PDF for STG, extract all the topics from index table"


response = client.models.generate_content(model=model_id, contents=[extract_topic_prompt, stg_pdf])


print(response)

# Access the first candidate
candidate = response.candidates[0]

# Access the content
content = candidate.content

# Access the parts
parts = content.parts

# Extract the text from each part and join them
extracted_text = "".join([part.text for part in parts])

# Create a list of topics from the extracted text
topics = []
for line in extracted_text.split('\n')[1:]:
    # Skip empty lines and the initial text
    if line.strip() and not line.startswith('Here are'):
        # Remove the number and dot at the start and strip whitespace
        topic = line.split('. ', 1)[1].strip() if '. ' in line else line.strip()
        topics.append(topic)

print("List of topics:", topics)

# Extract sub-topics for each topic
subtopics_by_topic = {}
for topic in topics:
    subtopic_prompt = f"From this PDF for STG, extract all the sub-topics under the topic '{topic}'"
    
    try:
        subtopic_response = client.models.generate_content(
            model=model_id, 
            contents=[subtopic_prompt, stg_pdf]
        )
        
        # Extract text from response
        subtopic_text = "".join([part.text for part in subtopic_response.candidates[0].content.parts])
        
        # Process the subtopics similar to topics
        subtopics = []
        for line in subtopic_text.split('\n'):
            if line.strip() and not line.startswith('Here are'):
                # Clean up the subtopic text
                subtopic = line.split('. ', 1)[1].strip().replace('*', '') if '. ' in line else line.strip().replace('*', '')
                subtopics.append(subtopic)
                
        subtopics_by_topic[topic] = subtopics
        print(f"\nTopic: {topic}")
        print("Sub-topics:", subtopics)
        
    except Exception as e:
        print(f"Error processing topic '{topic}': {str(e)}")

print(subtopics_by_topic)
