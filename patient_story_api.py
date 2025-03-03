from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Optional, List, Any
import os
import pandas as pd
from google import genai
from dotenv import load_dotenv
from patient_story_generation.past_patients_visit import get_patient_data as get_patient_visits
import json
import sys
import os


# Load environment variables
load_dotenv("ops/.env")

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)

# Create FastAPI app
app = FastAPI(
    title="Patient Story Generator API",
    description="""
    API for generating patient stories using templates and patient data.
    
    This API allows you to:
    - List available templates and patients
    - Generate patient stories based on templates and patient data
    - Retrieve detailed patient visit information
    
    All stories are generated using real patient data without hallucination or fabrication.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Define models
class StoryRequest(BaseModel):
    template_id: int = Field(..., description="ID of the template to use for story generation", example=1)
    patient_id: str = Field(..., description="ID of the patient to generate story for", example="P001")
    
    class Config:
        schema_extra = {
            "example": {
                "template_id": 1,
                "patient_id": "P001"
            }
        }

class StoryResponse(BaseModel):
    patient_id: str = Field(..., description="ID of the patient")
    template_id: int = Field(..., description="ID of the template used")
    story: str = Field(..., description="Generated patient story")
    metadata: Dict = Field(..., description="Additional metadata about the story generation")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "template_id": 1,
                "story": "This is John Doe, a 45-year-old male from New York, who can be reached at (555) 123-4567. He has visited our facility 3 times.\n\nVisit 1 (2022-01-15): He presented with fever and cough. Diagnosed with acute bronchitis. Treatment: Prescribed antibiotics and rest.\nVisit 2 (2022-03-20): He presented with joint pain. Diagnosed with early arthritis. Treatment: Anti-inflammatory medication and physical therapy.\nVisit 3 (2022-07-10): He presented with follow-up for arthritis. Diagnosed with improving condition. Treatment: Continued medication with reduced dosage.\n\nCurrent Status: John is scheduled for a follow-up visit on 2023-12-15.",
                "metadata": {
                    "template_name": "Default",
                    "template_description": "Standard patient story template with basic information and all visits",
                    "visit_count": 3,
                    "data_sources": ["basic_patient_data", "past_patients_visit.py"]
                }
            }
        }

class TemplateInfo(BaseModel):
    id: int = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    template: str = Field(..., description="Template string")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Default",
                "description": "Standard patient story template with basic information and all visits",
                "template": "...."
            }
        }

class PatientVisit(BaseModel):
    visit_number: int = Field(..., description="Visit number")
    visit_date: Optional[str] = Field(None, description="Date of the visit")
    complaints: Optional[str] = Field(None, description="Chief complaints during the visit")
    diagnosis: Optional[str] = Field(None, description="Diagnosis made during the visit")
    treatment: Optional[str] = Field(None, description="Treatment prescribed during the visit")
    vitals: Optional[Dict] = Field(None, description="Vital signs recorded during the visit")
    additional_info: Optional[Dict] = Field(None, description="Additional information recorded during the visit")

class PatientInfo(BaseModel):
    patient_id: str = Field(..., description="Patient ID")
    name: Optional[str] = Field(None, description="Patient name")
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    location: Optional[str] = Field(None, description="Patient location")
    contact: Optional[str] = Field(None, description="Patient contact information")
    visits: List[PatientVisit] = Field([], description="List of patient visits")

class PatientVisitsResponse(BaseModel):
    patient_id: str = Field(..., description="Patient ID")
    basic_info: Dict = Field(..., description="Basic patient information")
    visits: Dict = Field(..., description="Patient visit data")
    visit_count: int = Field(..., description="Number of visits")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "basic_info": {
                    "name": "John Doe",
                    "age": 45,
                    "gender": "Male",
                    "location": "New York",
                    "contact": "(555) 123-4567"
                },
                "visits": {
                    "1": {
                        "Visit_Date": "2022-01-15",
                        "Chief_Complaints": "Fever and cough",
                        "Diagnosis": "Acute bronchitis",
                        "Treatment": "Prescribed antibiotics and rest"
                    },
                    "2": {
                        "Visit_Date": "2022-03-20",
                        "Chief_Complaints": "Joint pain",
                        "Diagnosis": "Early arthritis",
                        "Treatment": "Anti-inflammatory medication and physical therapy"
                    },
                    "3": {
                        "Visit_Date": "2022-07-10",
                        "Chief_Complaints": "Follow-up for arthritis",
                        "Diagnosis": "Improving condition",
                        "Treatment": "Continued medication with reduced dosage"
                    }
                },
                "visit_count": 3
            }
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    templates_available: int = Field(..., description="Number of available templates")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "templates_available": 3
            }
        }

# Template storage - in a real app, this would be in a database
template_1 = "This is ABC 👩🏽, a 27-year-old from Nasik, Maharashtra🇮🇳, living at 444, Kathe Lane. She was born on August 6, 1997🎂, and can be reached at 9876543210. Muskan first visited us a few months ago with complaints of recurring headaches and fatigue, which we traced to stress and mild anemia. On her second visit, she reported dizziness and nausea, so we adjusted her treatment with a nutritional plan and supplements."

templates = {
    1: {
        "name": "Default",
        "description": "Standard patient story template with basic information and all visits",
        "template": template_1
    },
    
    2: {
        "name": "Brief",
        "description": "Concise summary of patient information and visits",
        "template": "{name}, {age}, {gender} from {location}. Contact: {contact}. {visit_count} visits.\n{visit_summary}\nNext follow-up: {next_followup_date}."
    },
    
    3: {
        "name": "Detailed",
        "description": "Comprehensive patient profile with structured format and detailed visit information",
        "template": "Patient Profile: {name} ({age})\nGender: {gender}\nLocation: {location}\nContact: {contact}\n\nMedical History:\n{visit_details_formatted}\n\nCurrent Status: Patient is currently following the prescribed treatment plan and is scheduled for a follow-up visit on {next_followup_date}."
    }
}

# Load patient data from CSV
def load_patient_data():
    try:
        csv_file_path = './data/Unified_Patient_Data.csv'
        df = pd.read_csv(csv_file_path)
        print(df.head())
        # Convert to dictionary for easier access
        patient_data = {}
        for _, row in df.iterrows():
            patient_id = str(row['Patient_id'])
            if patient_id not in patient_data:
                patient_data[patient_id] = {
                    'Patient_id': patient_id,
                    'visits': []
                }
            
            # Add basic patient info if available
            for field in ['Name', 'Age', 'Gender', 'Location', 'Contact']:
                if field in row and not pd.isna(row[field]) and field.lower() not in patient_data[patient_id]:
                    patient_data[patient_id][field.lower()] = row[field]
            
            # Add this visit to the patient's visits
            visit_data = {col: row[col] for col in df.columns if not pd.isna(row[col])}
            patient_data[patient_id]['visits'].append(visit_data)
            
        return patient_data
    except Exception as e:
        print(f"Error loading patient data: {e}")
        # Return empty dict if file not found, will be handled in the endpoint
        return {}

# Get comprehensive patient data with all visits
async def get_comprehensive_patient_data(patient_id: str) -> Dict:
    """
    Combines basic patient data with detailed visit information from past_patients_visit.py
    """
    # Get basic patient data from our local function
    basic_data = await get_patient_data()
    
    if patient_id not in basic_data:
        raise HTTPException(status_code=404, detail=f"Patient ID '{patient_id}' not found in basic data")
    
    patient_info = basic_data[patient_id].copy()
    
    # Get detailed visit data from past_patients_visit.py
    visit_data = get_patient_visits(patient_id)
    
    if not visit_data:
        print(f"Warning: No visit data found for patient {patient_id} in past_patients_visit.py")
    
    # Combine the data
    patient_info['visit_data'] = visit_data
    patient_info['visit_count'] = len(visit_data) if visit_data else len(patient_info.get('visits', []))
    
    return patient_info

# Format visit data for templates
def format_visit_data(patient_info: Dict) -> Dict:
    """
    Formats visit data for use in templates
    """
    result = patient_info.copy()
    print(result)

    # Set gender pronouns
    # gender = patient_info.get('gender', '').lower()
    # if gender == 'male':
    #     result['gender_pronoun'] = 'He'
    #     result['gender_possessive'] = 'his'
    # elif gender == 'female':
    #     result['gender_pronoun'] = 'She'
    #     result['gender_possessive'] = 'her'
    
    # Format visit details
    visit_details = []
    visit_summary = []
    visit_details_formatted = []
    
    # Use visit_data from past_patients_visit.py if available
    if 'visit_data' in patient_info and patient_info['visit_data']:
        for visit_num, visit in patient_info['visit_data'].items():
            visit_date = visit.get('Visit_Date', f"Visit {visit_num}")
            complaints = visit.get('Chief_Complaints', 'No complaints recorded')
            diagnosis = visit.get('Diagnosis', 'No diagnosis recorded')
            treatment = visit.get('Treatment', 'No treatment recorded')
            
            visit_details.append(f"Visit {visit_num} ({visit_date}): Patient presented with {complaints}. Diagnosed with {diagnosis}. Treatment: {treatment}.")
            visit_summary.append(f"V{visit_num}: {diagnosis}")
            visit_details_formatted.append(f"- Visit {visit_num} ({visit_date}):\n  Complaints: {complaints}\n  Diagnosis: {diagnosis}\n  Treatment: {treatment}")
    
    # Fallback to basic visit data if needed
    elif 'visits' in patient_info and patient_info['visits']:
        for i, visit in enumerate(patient_info['visits'], 1):
            visit_date = visit.get('Visit_Date', f"Visit {i}")
            complaints = visit.get('Chief_Complaints', 'No complaints recorded')
            diagnosis = visit.get('Diagnosis', 'No diagnosis recorded')
            treatment = visit.get('Treatment', 'No treatment recorded')
            
            visit_details.append(f"Visit {i} ({visit_date}): Patient presented with {complaints}. Diagnosed with {diagnosis}. Treatment: {treatment}.")
            visit_summary.append(f"V{i}: {diagnosis}")
            visit_details_formatted.append(f"- Visit {i} ({visit_date}):\n  Complaints: {complaints}\n  Diagnosis: {diagnosis}\n  Treatment: {treatment}")
    
    # Add formatted visit data to result
    result['visit_details'] = "\n".join(visit_details)
    result['visit_summary'] = ", ".join(visit_summary)
    result['visit_details_formatted'] = "\n".join(visit_details_formatted)
    
    # Add placeholder for next follow-up date if not provided
    if 'next_followup_date' not in result:
        result['next_followup_date'] = "the next available appointment"
    
    return result

# Generate story using Gemini
async def generate_story(template: str, patient_data: Dict) -> str:
    try:
        # Format the patient data for the template
        formatted_data = format_visit_data(patient_data)
        template = templates[1]["template"]
        # print(formatted_data)
        # # Format template with patient data
        # # Use safe formatting to handle missing fields
        # formatted_template = template
        # for key, value in formatted_data.items():
        #     placeholder = "{" + key + "}"
        #     if placeholder in formatted_template:
        #         formatted_template = formatted_template.replace(placeholder, str(value))
        
        # print(formatted_template)

        # Use Gemini to enhance the story
        prompt = f"""
        Given this patient's real medical visit data:
        
        {patient_data}
        
        Please generate a story matching the template {template} to make it more natural and readable for the doctor. 
        
        Fill in any missing placeholders 
        with plausible information based ONLY on the data provided. 
        
        DO NOT add any medical facts, diagnoses, 
        or treatments that are not explicitly mentioned in the original text. Keep all medical information 
        exactly as provided, but make the narrative flow better. Do not include any non english words.

        If there is any pending follow-up, please mention it in the story in the last. 
        
        If there are placeholders like {{field_name}} that couldn't be filled, do not include them in the story inventing specific information.
        """
        
        response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error generating story: {e}")
        return ""  # Fallback to the template if generation fails

# Dependency to get patient data
async def get_patient_data():
    data = load_patient_data()
    if not data:
        raise HTTPException(status_code=500, detail="Failed to load patient data")
    return data

# Endpoints
@app.get("/templates", response_model=List[TemplateInfo], 
         summary="List all templates",
         description="Returns a list of all available templates with their IDs, names, descriptions, and template strings.",
         responses={
             200: {"description": "List of available templates", "model": List[TemplateInfo]},
             500: {"description": "Internal server error"}
         })
async def list_templates():
    """List all available templates with their IDs and descriptions"""
    return [
        TemplateInfo(
            id=template_id,
            name=template_data["name"],
            description=template_data["description"],
            template=template_data["template"]
        )
        for template_id, template_data in templates.items()
    ]

@app.get("/patients", response_model=List[str], 
         summary="List all patients",
         description="Returns a list of all available patient IDs.",
         responses={
             200: {"description": "List of available patient IDs"},
             500: {"description": "Internal server error"}
         })
async def list_patients(patient_data: Dict = Depends(get_patient_data)):
    """List all available patient IDs"""
    return list(patient_data.keys())

@app.post("/generate-story", response_model=StoryResponse, 
          summary="Generate a patient story",
          description="Generates a patient story using the specified template and patient data. The story includes information from all patient visits without hallucination.",
          responses={
              200: {"description": "Successfully generated story", "model": StoryResponse},
              404: {"description": "Template ID or Patient ID not found"},
              500: {"description": "Error retrieving patient data or generating story"}
          })
async def create_story(
    request: StoryRequest
):
    """Generate a patient story using the specified template and patient data"""
    
    # Check if template exists
    if request.template_id not in templates:
        raise HTTPException(status_code=404, detail=f"Template ID '{request.template_id}' not found")
    
    # Get comprehensive patient data with all visits
    try:
        patient_info = await get_comprehensive_patient_data(request.patient_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving patient data: {str(e)}")
    
    # Get template
    template_data = templates[request.template_id]
    template = template_data["template"]
    
    # Generate story
    story = await generate_story(template, patient_info)
    
    # Return response
    return StoryResponse(
        patient_id=request.patient_id,
        template_id=request.template_id,
        story=story,
        metadata={
            "template_name": template_data["name"],
            "template_description": template_data["description"],
            "visit_count": patient_info.get('visit_count', 0),
            "data_sources": ["basic_patient_data", "past_patients_visit.py" if 'visit_data' in patient_info and patient_info['visit_data'] else ""]
        }
    )

@app.get("/template/{template_id}", response_model=TemplateInfo, 
         summary="Get template details",
         description="Returns details for a specific template by ID.",
         responses={
             200: {"description": "Template details", "model": TemplateInfo},
             404: {"description": "Template ID not found"}
         })
async def get_template(template_id: int):
    """Get details for a specific template"""
    if template_id not in templates:
        raise HTTPException(status_code=404, detail=f"Template ID '{template_id}' not found")
    
    template_data = templates[template_id]
    return TemplateInfo(
        id=template_id,
        name=template_data["name"],
        description=template_data["description"],
        template=template_data["template"]
    )

@app.get("/patient/{patient_id}/visits", response_model=PatientVisitsResponse, 
         summary="Get patient visits",
         description="Returns all visits for a specific patient, combining data from both the local CSV and past_patients_visit.py.",
         responses={
             200: {"description": "Patient visit data", "model": PatientVisitsResponse},
             404: {"description": "Patient ID not found"},
             500: {"description": "Error retrieving patient visits"}
         })
async def get_patient_visits_endpoint(patient_id: str):
    """Get all visits for a specific patient"""
    try:
        patient_info = await get_comprehensive_patient_data(patient_id)
        return PatientVisitsResponse(
            patient_id=patient_id,
            basic_info={k: v for k, v in patient_info.items() if k not in ['visits', 'visit_data']},
            visits=patient_info.get('visit_data', {}),
            visit_count=patient_info.get('visit_count', 0)
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving patient visits: {str(e)}")

@app.get("/health", response_model=HealthResponse, 
         summary="Health check",
         description="Returns the health status of the API.",
         responses={
             200: {"description": "API is healthy", "model": HealthResponse}
         })
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", 
        templates_available=len(templates)
    )

# API Documentation
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run("patient_story_api:app", host="0.0.0.0", port=8000, reload=True) 