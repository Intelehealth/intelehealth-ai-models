import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
from google import genai
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Initialize Faker
fake = Faker()

# Configure Gemini

# Load environment variables
load_dotenv("ops/.env")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Read the original dataset
df = pd.read_csv('data/NAS_UnseenData_02042025_v3_medications_judged_filtered.csv')

def get_llm_diagnosis(clinical_notes):
    """Get diagnosis from Gemini based on clinical notes"""
    try:
        prompt = f"""You are a medical expert. Based on the clinical notes provided, 
        determine the most likely diagnosis. Return the response in the following format:
        diagnosis_name|icd_code|diagnosis_description
        
        For example:
        Acute Pharyngitis|J02.9|Acute Pharyngitis
        
        Only return the diagnosis in the specified format, nothing else.
        
        Clinical notes:
        {clinical_notes}"""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        diagnosis = response.text.strip()
        diagnosis_name, icd_code, diagnosis_desc = diagnosis.split('|')
        return (diagnosis_name, icd_code, diagnosis_desc)
    except Exception as e:
        print(f"Error getting Gemini diagnosis: {e}")
        # Fallback to a default diagnosis if Gemini fails
        return ("Acute Upper Respiratory Infection", "J06.9", "Acute Upper Respiratory Infection")

# Function to generate synthetic data
def generate_synthetic_data(num_records=200):
    synthetic_data = []
    
    # Common medications and their patterns
    medications = [
        ("Paracetamol Tablet 500 mg", "mg", "1 Tablet Twice daily-दिवसातून दोनदा for 3 DAYS - दिवस"),
        ("Omeprazole Capsule 20 mg", "null mg", "1 Capsule Once daily, in the morning - दिवसातून एकदा, सकाळी for 3 DAYS - दिवस"),
        ("Levocetirizine* Tablet 5 mg", "null mg", "0.5 Tablet Once daily, at bedtime- दिवसातून एकदा, झोपेच्या वेळी for 3 DAYS - दिवस"),
        ("Amoxicillin Capsule 500 mg", "mg", "1 Capsule Twice daily-दिवसातून दोनदा for 5 DAYS - दिवस"),
        ("Azithromycin Tablet 500 mg", "mg", "1 Tablet Once daily-दिवसातून एकदा for 3 DAYS - दिवस")
    ]
    
    # Symptom-Diagnosis mappings with ICD codes (as fallback)
    symptom_diagnosis_map = {
        "Cold, Sneezing, Headache": [
            ("Acute Upper Respiratory Infection", "J06.9", "Acute Upper Respiratory Infection"),
            ("Acute Pharyngitis", "J02.9", "Acute Pharyngitis"),
            ("Acute Sinusitis", "J01.9", "Acute Sinusitis")
        ],
        "Fever, Body pain": [
            ("Acute Upper Respiratory Infection", "J06.9", "Acute Upper Respiratory Infection"),
            ("Viral Fever", "R50.9", "Viral Fever"),
            ("Acute Gastroenteritis", "A09", "Acute Gastroenteritis")
        ],
        "Cough, Sore throat": [
            ("Acute Pharyngitis", "J02.9", "Acute Pharyngitis"),
            ("Acute Laryngitis", "J04.0", "Acute Laryngitis"),
            ("Acute Bronchitis", "J20.9", "Acute Bronchitis")
        ],
        "Skin disorder": [
            ("Acute Urticaria", "L50.9", "Acute Urticaria"),
            ("Contact Dermatitis", "L25.9", "Contact Dermatitis"),
            ("Acute Eczema", "L30.9", "Acute Eczema")
        ],
        "Ear pain, Headache": [
            ("Acute Otitis Media", "H66.9", "Acute Otitis Media"),
            ("Acute Otitis Externa", "H60.9", "Acute Otitis Externa"),
            ("Acute Sinusitis", "J01.9", "Acute Sinusitis")
        ],
        "Abdominal pain, Nausea": [
            ("Acute Gastroenteritis", "A09", "Acute Gastroenteritis"),
            ("Acute Gastritis", "K29.7", "Acute Gastritis"),
            ("Acute Colitis", "K52.9", "Acute Colitis")
        ],
        "Joint pain, Swelling": [
            ("Acute Arthritis", "M13.9", "Acute Arthritis"),
            ("Acute Bursitis", "M71.9", "Acute Bursitis"),
            ("Acute Tendinitis", "M77.9", "Acute Tendinitis")
        ],
        "Eye irritation, Redness": [
            ("Acute Conjunctivitis", "H10.9", "Acute Conjunctivitis"),
            ("Acute Blepharitis", "H01.0", "Acute Blepharitis"),
            ("Acute Iritis", "H20.9", "Acute Iritis")
        ]
    }
    
    # Medical history components
    medical_conditions = [
        "None",
        "Hypertension",
        "Diabetes Mellitus",
        "Asthma",
        "Hypothyroidism",
        "Arthritis",
        "Migraine",
        "Gastritis"
    ]
    
    allergies = [
        "No known allergies",
        "Penicillin",
        "Sulfa drugs",
        "Dust",
        "Pollen",
        "Shellfish",
        "Peanuts",
        "Latex"
    ]
    
    lifestyle_factors = {
        'smoking': [
            "Patient denied/has no h/o smoking",
            "Former smoker - quit 2 years ago",
            "Occasional smoker",
            "Regular smoker - 10 cigarettes/day"
        ],
        'alcohol': [
            "No",
            "Occasional - social drinking",
            "Regular - 2-3 drinks/week",
            "Former drinker - quit 1 year ago"
        ],
        'pregnancy': [
            "Not pregnant",
            "Not applicable",
            "Pregnant - 20 weeks",
            "Not pregnant - post-menopausal"
        ]
    }
    
    # Generate records with progress bar
    for i in tqdm(range(num_records), desc="Generating synthetic records", unit="record"):
        # Generate visit dates
        visit_date = fake.date_time_between(start_date='-30d', end_date='+30d')
        visit_start = visit_date
        visit_creation = visit_date + timedelta(minutes=random.randint(5, 20))
        visit_end = visit_creation + timedelta(hours=random.randint(4, 8))
        
        # Generate patient ID
        patient_id = f"9522{random.randint(100000, 999999)}"
        
        # Generate vitals
        sbp = random.randint(110, 140)
        dbp = random.randint(70, 90)
        pulse = random.randint(60, 100)
        temp = round(random.uniform(36.5, 37.5), 2)
        weight = round(random.uniform(40, 80), 2)
        height = random.randint(150, 180)
        bmi = round(weight / ((height/100) ** 2), 2)
        rr = random.randint(16, 24)
        spo2 = random.randint(95, 100)
        
        # Select random symptoms
        symptom = random.choice(list(symptom_diagnosis_map.keys()))
        
        # Generate medical history
        medical_condition = random.choice(medical_conditions)
        allergy = random.choice(allergies)
        smoking_status = random.choice(lifestyle_factors['smoking'])
        alcohol_status = random.choice(lifestyle_factors['alcohol'])
        pregnancy_status = random.choice(lifestyle_factors['pregnancy'])
        
        # Generate clinical notes
        gender = random.choice(['Male', 'Female'])
        age = random.randint(5, 70)
        clinical_notes = f"""Gender: {gender}

 Age: {age} years

 Chief_complaint: ► **{symptom}** :  
• Duration - {random.randint(1, 7)} Days.  
• Precipitating factors - {random.choice(['Cold weather', 'Wind', 'Dust', 'None'])}.  
• Prior treatment sought - None.  

 Physical_examination: **General exams:**  
• Eyes: Jaundice-no jaundice seen, [picture taken].  
• Eyes: Pallor-normal pallor, [picture taken].  
• Arm-Pinch skin* - pinch test normal.  
• Nail abnormality-nails normal, [picture taken].  
• Nail anemia-Nails are not pale, [picture taken].  
• Ankle-no pedal oedema, [picture taken].  

 Patient_medical_history: • Pregnancy status - {pregnancy_status}.  
• Allergies - {allergy}.  
• Alcohol use - {alcohol_status}.  
• Smoking history - {smoking_status}.  
• Medical History - {medical_condition}.  
• Drug history - {random.choice(['No recent medication', 'On regular medication for ' + medical_condition if medical_condition != 'None' else 'No recent medication'])}.  

 Family_history: •Do you have a family history of any of the following? : {random.choice(['None', 'Hypertension', 'Diabetes', 'Heart disease', 'Cancer'])}.  

 Vitals:- 

Sbp: {sbp}

 Dbp: {dbp}

 Pulse: {pulse}

 Temperature: {temp} 'C

 Weight: {weight} Kg

 Height: {height} cm

 BMI: {bmi}

 RR: {rr}

 SPO2: {spo2}

 HB: Null

 Sugar_random: Null

 Blood_group: Null

 Sugar_pp: Null

 Sugar_after_meal: Null

"""
        
        # Get diagnosis from Gemini
        diagnosis = get_llm_diagnosis(clinical_notes)
        
        # Select random medications
        num_meds = random.randint(1, 3)
        selected_meds = random.sample(medications, num_meds)
        
        # Create record
        record = {
            'Visit_id': f"{random.randint(11117000, 11117999)}_sd",
            'Patient_id': patient_id,
            'Gender': gender[0],  # First letter of gender
            'Age': age,
            'Visit_started_date': visit_start.strftime('%Y-%m-%d %H:%M:%S'),
            'Visit_creation_date': visit_creation.strftime('%Y-%m-%d %H:%M:%S'),
            'Visit_ended_date': visit_end.strftime('%Y-%m-%d %H:%M:%S'),
            'Specialty': 'General Physician',
            'Sbp': sbp,
            'Dbp': dbp,
            'Pulse': pulse,
            'Temperature': temp,
            'Weight': weight,
            'Height': height,
            'BMI': bmi,
            'RR': rr,
            'SPO2': spo2,
            'Symptoms': symptom,
            'Diagnosis_provided': 'yes',
            'Diagnosis': f"{diagnosis[0]}:Primary & Provisional",
            'ICD_Diagnosis': f"{diagnosis[1]}:{diagnosis[2]}",
            'Primary & Provisional': diagnosis[2],
            'Medications': ';'.join([f"{med[0]}: {med[1]}, {med[2]}" for med in selected_meds]),
            'Medicines': ', '.join([med[0] for med in selected_meds]),
            'Strength': ', '.join([med[1] for med in selected_meds]),
            'Dosage': '. '.join([med[2] for med in selected_meds]),
            'Medical_advice': 'GURGLE WITH LUKEWARM SALINE WATER - कोमट खारट पाण्याने गुळणे करा',
            'Referral_advice': 'AVOID OILY AND SPICY FOODS - तेलकट आणि मसालेदार पदार्थ टाळा',
            'Follow_up_date': (visit_date + timedelta(days=3)).strftime('%d-%m-%Y'),
            'Sign_submit_time': (visit_creation + timedelta(minutes=random.randint(5, 15))).strftime('%Y-%m-%d %H:%M:%S'),
            'Patient_waiting_time_(in_hours)': round(random.uniform(0.2, 1.0), 4),
            'Additional_comments': 'clinical condition and vital parameters noted. WNL.',
            'Clinical_notes': clinical_notes
        }
        
        synthetic_data.append(record)
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    return pd.DataFrame(synthetic_data)

# Generate synthetic data
synthetic_df = generate_synthetic_data(200)

# Generate filename with timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f'data/synthetic_medical_visits_{timestamp}.csv'

# Save to CSV using the generated filename
synthetic_df.to_csv(filename, index=False)
print(f"Generated 200 synthetic medical visit records and saved to '{filename}'") 