import requests
import json
import re

patient_test_case = """
Gender: Female

 Age: 70 years

 Chief_complaint: ► **Cold, Sneezing** :  
• 1 Days.  
• Precipitating factors - Cold weather, Wind.  
• Prior treatment sought - None.  
► **Headache** :  
• Duration - 1 Days.  
• Site - Localized - कपाळावरती डोकं दुखतंय .  
• Severity - Moderate.  
• Onset - Acute onset (Patient can recall exact time when it started).  
• Character of headache - Throbbing.  
• Radiation - pain does not radiate.  
• Timing - Day.  
• Exacerbating factors - bending.  
• Prior treatment sought - None.  
► **Leg, Knee or Hip Pain** :  
• Site - Right leg, Hip, Thigh, Knee, Site of knee pain - Front, Back,
Lateral/medial. Swelling - No, Calf, Left leg, Hip, Thigh, Knee, Site of knee
pain - Front, Back, Lateral/medial. Swelling - No, Calf, Hip.  
• Duration - 6 Days.  
• Pain characteristics - Sharp shooting.  
• Onset - Gradual.  
• Progress - Static (Not changed).  
• Pain does not radiate.  
• Aggravating

 Physical_examination: **General exams:**  
• Eyes: Jaundice-no jaundice seen, [picture taken].  
• Eyes: Pallor-normal pallor, [picture taken].  
• Arm-Pinch skin* - pinch test normal.  
• Nail abnormality-nails normal, [picture taken].  
• Nail anemia-Nails are not pale, [picture taken].  
• Ankle-no pedal oedema, [picture taken].  
**Joint:**  
• non-tender.  
• no deformity around joint.  
• full range of movement is seen.  
• joint is not swollen.  
• no pain during movement.  
• no redness around joint.  
**Back:**  
• tenderness observed.  
**Head:**  
• No injury.

 Patient_medical_history: • Pregnancy status - Not pregnant.  
• Allergies - No known allergies.  
• Alcohol use - No.  
• Smoking history - Patient denied/has no h/o smoking.  
• Medical History - None.  
• Drug history - No recent medication.  

 Family_history: •Do you have a family history of any of the following? : None.  

 Vitals:- 

Sbp: 140.0

 Dbp: 90.0

 Pulse: 83.0

 Temperature: 36.78 'C

 Weight: 44.75 Kg

 Height: 152.0 cm

 BMI: 19.37

 RR: 21.0

 SPO2: 97.0

 HB: Null

 Sugar_random: Null

 Blood_group: Null

 Sugar_pp: Null

 Sugar_after_meal: Null
"""

diagnosis = "Acute Pharyngitis"

def process_medications(medication_text):
    """
    Process the medication recommendations text into a structured array
    
    Args:
        medication_text (str): Text containing medication recommendations
        
    Returns:
        list: Array of structured medication objects with the following fields:
            - name: Medication name
            - confidence: High/Moderate/Low confidence level
            - strength: Dosage strength (e.g., "500 mg")
            - route_form: Administration form (e.g., "tablet, oral")
            - frequency: How often to take (e.g., "4-6 hours")
            - duration: How long to take (e.g., "3-5 days")
            - rationale: Reason for the medication
    """
    medications = []
    
    # Split text into individual medication entries (each starting with a number)
    med_entries = []
    current_entry = ""
    
    for line in medication_text.strip().split("\n"):
        if line.strip() and line.strip()[0].isdigit() and "**" in line:
            if current_entry:
                med_entries.append(current_entry)
            current_entry = line
        elif current_entry:
            current_entry += " " + line
    
    if current_entry:
        med_entries.append(current_entry)
    
    # Process each entry
    for entry in med_entries:
        med = {}
        
        # Extract name - remove the colon if it exists
        if "**" in entry and ":" in entry:
            name_part = entry.split("**")[1].split(":")[0]
            if "(" in name_part:
                name = name_part.split("(")[0].strip()
            else:
                name = name_part.strip()
            med["name"] = name
        else:
            continue  # Skip if we can't find a name
        
        # Extract confidence
        if "High Likelihood" in entry:
            med["confidence"] = "High"
        elif "Moderate Likelihood" in entry:
            med["confidence"] = "Moderate"
        elif "Low Likelihood" in entry:
            med["confidence"] = "Low"
        else:
            med["confidence"] = "Unknown"
        
        # Extract strength
        if "mg" in entry:
            try:
                # Find the number before "mg"
                mg_matches = re.findall(r'(\d+)\s*mg', entry)
                if mg_matches:
                    med["strength"] = f"{mg_matches[0]} mg"
                else:
                    med["strength"] = ""
            except:
                med["strength"] = ""
        elif "IU" in entry:
            try:
                # Find the number before "IU"
                iu_matches = re.findall(r'(\d+)\s*IU', entry)
                if iu_matches:
                    med["strength"] = f"{iu_matches[0]} IU"
                else:
                    med["strength"] = ""
            except:
                med["strength"] = ""
        else:
            med["strength"] = ""
        
        # Extract route form
        if "tablet" in entry.lower() and "oral" in entry.lower():
            med["route_form"] = "tablet, oral"
        elif "tablet" in entry.lower():
            med["route_form"] = "tablet"
        elif "oral" in entry.lower():
            med["route_form"] = "oral"
        elif "apply" in entry.lower() or "topical" in entry.lower():
            med["route_form"] = "topical"
        else:
            med["route_form"] = ""
        
        # Extract frequency
        if "every" in entry:
            try:
                freq_match = re.search(r'every\s+([\w\-]+\s+[\w\-]+)', entry)
                if freq_match:
                    med["frequency"] = freq_match.group(1)
                else:
                    med["frequency"] = ""
            except:
                med["frequency"] = ""
        elif "once daily" in entry.lower():
            med["frequency"] = "once daily"
        elif "2-3 times daily" in entry.lower():
            med["frequency"] = "2-3 times daily"
        elif "daily" in entry.lower():
            med["frequency"] = "daily"
        else:
            med["frequency"] = ""
        
        # Extract duration
        if "For" in entry and "days" in entry:
            try:
                dur_match = re.search(r'For\s+([\w\-]+\s+[\w\-]+)', entry)
                if dur_match:
                    med["duration"] = dur_match.group(1)
                else:
                    med["duration"] = ""
            except:
                med["duration"] = ""
        else:
            med["duration"] = ""
        
        # Extract rationale
        if "Rationale:" in entry:
            rationale = entry.split("Rationale:")[1].strip()
            med["rationale"] = rationale
        else:
            med["rationale"] = ""
        
        medications.append(med)
    
    return medications


def get_treatment_recommendations(patient_case, diagnosis, api_url="http://127.0.0.1:8000/ttx/v1", model="gemini-2.0-flash"):
    """
    Get treatment recommendations from the TTX API and return structured medication data
    
    Args:
        patient_case (str): Patient case description
        diagnosis (str): Diagnosis for the patient
        api_url (str): URL for the TTX API
        model (str): Model name to use for recommendations
        
    Returns:
        dict: Dictionary with keys:
            - success (bool): Whether the API call succeeded
            - medications (list): Structured medication recommendations
            - error (str, optional): Error message if the call failed
            - raw_response (dict, optional): Raw API response
    """
    result = {
        "success": False,
        "medications": []
    }
    
    try:
        response = requests.post(
            api_url,
            json={"model_name": model, "case": patient_case, "diagnosis": diagnosis}
        )
        
        if response.status_code == 200:
            response_json = response.json()
            # result["raw_response"] = response_json
            
            # Process medication recommendations if present
            if "data" in response_json and "output" in response_json["data"] and "medication_recommendations" in response_json["data"]["output"]:
                med_text = response_json["data"]["output"]["medication_recommendations"]
                result["medications"] = process_medications(med_text)
                result["medical_advice"] = response_json["data"]["output"]["medical_advice"]
                result["success"] = True
            else:
                result["error"] = "No medication recommendations found in API response"
        else:
            result["error"] = f"API error: {response.status_code} - {response.text}"
    except Exception as e:
        result["error"] = f"Exception: {str(e)}"
    
    return result


if __name__ == "__main__":
    # Example patient case
    patient_test_case = """
    Gender: Female
    
     Age: 70 years
    
     Chief_complaint: ► **Cold, Sneezing** :  
    • 1 Days.  
    • Precipitating factors - Cold weather, Wind.  
    • Prior treatment sought - None.  
    ► **Headache** :  
    • Duration - 1 Days.  
    • Site - Localized - कपाळावरती डोकं दुखतंय .  
    • Severity - Moderate.  
    • Onset - Acute onset (Patient can recall exact time when it started).  
    • Character of headache - Throbbing.  
    • Radiation - pain does not radiate.  
    • Timing - Day.  
    • Exacerbating factors - bending.  
    • Prior treatment sought - None.  
    ► **Leg, Knee or Hip Pain** :  
    • Site - Right leg, Hip, Thigh, Knee, Site of knee pain - Front, Back,
    Lateral/medial. Swelling - No, Calf, Left leg, Hip, Thigh, Knee, Site of knee
    pain - Front, Back, Lateral/medial. Swelling - No, Calf, Hip.  
    • Duration - 6 Days.  
    • Pain characteristics - Sharp shooting.  
    • Onset - Gradual.  
    • Progress - Static (Not changed).  
    • Pain does not radiate.  
    • Aggravating
    
     Physical_examination: **General exams:**  
    • Eyes: Jaundice-no jaundice seen, [picture taken].  
    • Eyes: Pallor-normal pallor, [picture taken].  
    • Arm-Pinch skin* - pinch test normal.  
    • Nail abnormality-nails normal, [picture taken].  
    • Nail anemia-Nails are not pale, [picture taken].  
    • Ankle-no pedal oedema, [picture taken].  
    **Joint:**  
    • non-tender.  
    • no deformity around joint.  
    • full range of movement is seen.  
    • joint is not swollen.  
    • no pain during movement.  
    • no redness around joint.  
    **Back:**  
    • tenderness observed.  
    **Head:**  
    • No injury.
    
     Patient_medical_history: • Pregnancy status - Not pregnant.  
    • Allergies - No known allergies.  
    • Alcohol use - No.  
    • Smoking history - Patient denied/has no h/o smoking.  
    • Medical History - None.  
    • Drug history - No recent medication.  
    
     Family_history: •Do you have a family history of any of the following? : None.  
    
     Vitals:- 
    
    Sbp: 140.0
    
     Dbp: 90.0
    
     Pulse: 83.0
    
     Temperature: 36.78 'C
    
     Weight: 44.75 Kg
    
     Height: 152.0 cm
    
     BMI: 19.37
    
     RR: 21.0
    
     SPO2: 97.0
    
     HB: Null
    
     Sugar_random: Null
    
     Blood_group: Null
    
     Sugar_pp: Null
    
     Sugar_after_meal: Null
    """
    
    diagnosis = "Acute Pharyngitis"
    
    # Get recommendations
    print("Getting treatment recommendations...")
    result = get_treatment_recommendations(patient_test_case, diagnosis)
    
    if result["success"]:
        print(json.dumps(result, indent=2))
        
        # print("\nMedication Summary:")
        # for med in result["medications"]:
        #     print(f"Name: {med.get('name', 'N/A')}")
        #     print(f"Confidence: {med.get('confidence', 'N/A')}")
        #     print(f"Strength: {med.get('strength', 'N/A')}")
        #     print(f"Route/Form: {med.get('route_form', 'N/A')}")
        #     print(f"Frequency: {med.get('frequency', 'N/A')}")
        #     print(f"Duration: {med.get('duration', 'N/A')}")
        #     print(f"Rationale: {med.get('rationale', 'N/A')}")
        #     print("---")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")



