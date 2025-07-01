import dspy
from utils.metric_utils import load_gemini2_lm, load_gemini2_5_lm_1, \
    metric_fun, openai_llm_judge, load_gemini_lm, load_open_ai_lm, \
    load_hyperbolic_llama_3_3_70b_instruct, load_open_ai_o1_mini_lm, load_open_ai_lm_4_1, \
    load_gemini_2_5_pro_lm, load_lm_studio_medgemma_27b_text_it, load_gemini_2_5_vertex_lm, load_aws_bedrock_lm
from dotenv import load_dotenv
import pandas as pd
import argparse
import os

from modules.DDxModule import DDxModule
from modules.TelemedicineDDxModule import TelemedicineDDxModule
from modules.TelemedicineTenDDxModule import TelemedicineTenDDxModule
from modules.DDxKBModule import DDxKBModule
from modules.TelemedicineICD11DDxModule import TelemedicineICD11DDxModule
from modules.TelemedicineSnomedCTDDxModule import TelemedicineSnomedCTDDxModule
from modules.DDxTelemedicineModule import DDxTelemedicineModule
# Parse command line arguments
parser = argparse.ArgumentParser(description='Run differential diagnosis inference on NAS unseen data')
parser.add_argument('--input_csv', type=str, required=True,
                    help='Input CSV file containing the patient data')
parser.add_argument('--output_csv', type=str, required=True,
                    help='Output CSV file to save the results')
parser.add_argument('--model', type=str, choices=['gemini2', 'openai', 'openai_4_1', 'llama', 'gemini2_5', 'gemini_2_5_pro', 'medgemma-27-it', 'gemini_2_5_vertex', 'aws_bedrock_llama_3_2_11b'], default='gemini2',
                    help='LLM model to use')
parser.add_argument('--trained_file', type=str, required=True,
                    help='Trained model file to load')
parser.add_argument('--module', type=str, choices=['ddx', 'telemedicine', 'icd_snowmed_kb', 'telemedicine_ten', 'telemedicine_icd_11', 'telemedicine_snomed_ct', 'ddx_telemedicine'], default='ddx',
                    help='Module type to use (ddx or telemedicine)')
args = parser.parse_args()

load_dotenv(
    "ops/.env"
)

# Configure model based on command line argument
if args.model == 'gemini2':
    load_gemini2_lm()
elif args.model == 'gemini2_5':
    load_gemini2_5_lm_1()
elif args.model == 'gemini_2_5_pro':
    load_gemini_2_5_pro_lm()
elif args.model == 'openai':
    load_open_ai_lm()
elif args.model == 'gemini':
    load_gemini_lm()
elif args.model == 'llama':
    load_hyperbolic_llama_3_3_70b_instruct()
elif args.model == 'openai_4_1':
    load_open_ai_lm_4_1()
elif args.model == 'medgemma-27-it':
    load_lm_studio_medgemma_27b_text_it()
elif args.model == 'gemini_2_5_pro':
    load_gemini_2_5_pro_lm()
elif args.model == 'gemini_2_5_vertex':
    load_gemini_2_5_vertex_lm()
elif args.model == 'aws_bedrock_llama_3_2_11b':
    load_aws_bedrock_lm()

cot = ""
df = pd.read_csv(args.input_csv)
print(df.columns)

new_df = df[df.columns]

print(new_df)
print(df.shape)

question = ""
# Initialize appropriate module based on argument
if args.module == 'telemedicine':
    cot = TelemedicineDDxModule()
    question = """
        You are a doctor conducting a telemedicine consultation with a patient in rural India.
        Based on patient history, symptoms, physical exam findings, and demographics:
        1. Provide top 5 differential diagnoses ranked by likelihood
        2. For each: include likelihood score (high/moderate/low) and brief rationale
        3. For high/moderate: mention key features, infections, and rural India relevance
        4. For low: briefly explain why it doesn't fit
        Keep all responses concise and to the point.
    """
elif args.module == 'icd_snowmed_kb':
    cot = DDxKBModule()
    question = """
        Based on given patient history, symptoms, physical exam findings, and demographics please provide top 5 differential diagnoses ranked by likelihood. Refer to ICD 11 and SNOMED CT terminologies to provide the diagnosis.
        
        For each diagnosis: include likelihood score (high/moderate/low) and brief rationale
        For high/moderate: mention key features, infections, and rural India relevance
        For low: briefly explain why it doesn't fit

        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Keep all responses concise and to the point.
    """
elif args.module == 'telemedicine_ten':
    cot = TelemedicineTenDDxModule()
    question = """
        You are a doctor conducting a telemedicine consultation with a patient in rural India.
        Based on patient history, symptoms, physical exam findings, and demographics:
        1. Provide top 10 differential diagnoses ranked by likelihood
        2. For each: include likelihood score (high/moderate/low) and brief one line rationale. Do not include any other case or question in the rationale.

        Keep all responses concise and to the point.
    """
elif args.module == 'telemedicine_icd_11':
    cot = TelemedicineICD11DDxModule()
    question = """
        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Based on patient history, symptoms, physical exam findings, and demographics:
        1. Provide the top 5 differential diagnoses, with highest confidence ranked in order of likelihood, picked from the ICD-11 database.
        2. Ensure diagnoses are relevant to a telemedicine context in India.
        3. For each diagnosis: include a brief rationale.
        4. Exclude diagnoses from the following ICD-11 chapters:
            - Chapter 20 to 26
            - Chapter V
            - Chapter X
        Keep all responses concise and to the point.
    """
elif args.module =='ddx':
    cot = DDxModule()
    question = "You are a doctor with the following patient from rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? For each diagnosis, include the likelihood score and the brief rationale for that diagnosis. For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural india. For a low likelihood diagnosis, include lack of fit reasoning under the rationale for that diagnosis. Please rank the differential diagnoses based on the likelihood and provide a brief explanation for each diagnosis. Please don't include  CASE and Question in the rationale.Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient."
elif args.module == 'telemedicine_snomed_ct':
    cot = TelemedicineSnomedCTDDxModule()
    question = """
        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Based on patient history, symptoms, physical exam findings, and demographics:
        1. Provide the top 5 differential diagnoses, with highest confidence ranked in order of likelihood, picked from the snomed ct database.
        2. Ensure diagnoses are relevant to a telemedicine context in India.
        3. For each diagnosis: include a brief rationale and the confidence of prediction - high, moderate, low etc.
        4. do not include any snomed ct codes in the response.
        5. do not repeat case history in the rationale. keep the rationale concise and to the point.

        Keep all responses concise and to the point.
    """
elif args.module == 'ddx_telemedicine':
    cot = DDxTelemedicineModule()
    question = """
You are a doctor consulting a patient in rural India via telemedicine.
        Here is their case with the history of presenting illness, their physical exams, and demographics based on the information provided through the telemedicine consultation.
        What would be the top 5 differential diagnosis for this patient?
        For each diagnosis, include the likelihood score and the rationale for that diagnosis.
        The top 5 differential diagnoses you predict is in this list add that, otherwise map it to the closest equivalent in the list as per the patient case history.
        If there is no suitable diagnosis in the list that matches the patient's case history, add a diagnosis that is most likely to be the cause of the patient's symptoms.

        Standard Diagnoses List:
        - Abnormal Uterine Bleeding
        - Acne Vulgaris
        - Acute Cholecystitis
        - Acute Conjunctivitis
        - Acute Diarrhea
        - Acute Gastritis
        - Acute Gastroenteritis
        - Acute Heart Failure
        - Acute Otitis Media
        - Acute Pharyngitis
        - Acute Pulpitis
        - Acute Renal Failure
        - Acute Rheumatic Fever
        - Acute Rhinitis
        - Acute Sinusitis
        - Acute Viral Hepatitis
        - Allergic Rhinitis
        - Alzheimer disease
        - Amoebic Liver Abscess
        - Anemia
        - Anorexia Nervosa
        - Acute Appendicitis
        - Atopic Dermatitis
        - Atrial Fibrillation
        - Blunt injury of foot
        - Breast Cancer
        - Bronchial Asthma
        - Bronchiectasis
        - Burns
        - Candidiasis
        - Carcinoma of Stomach
        - Cellulitis
        - Cerebral Malaria
        - Cervical Spondylosis
        - Chancroid
        - Chicken pox
        - Cholera
        - Chronic Active Hepatitis
        - Chronic Bronchitis
        - Chronic Constipation
        - Chronic Duodenal Ulcer
        - Chronic Heart Failure
        - Chronic Kidney Disease
        - Chronic Kidney Disease due to Hypertension
        - Chronic Liver Disease
        - Chronic Renal Failure
        - Cirrhosis
        - Cluster Headache
        - Colitis
        - Collapse of Lung
        - Colon Cancer
        - Complete Heart Block
        - Congestive Heart Failure
        - Consolidation of Lung
        - COPD
        - Cor Pulmonale
        - Dementia
        - Dengue Fever
        - Dental Caries
        - Diabetes Insipidus
        - Diabetes Mellitus
        - Diabetic Ketoacidosis
        - Diabetic Neuropathy
        - Drug Reaction
        - Ectopic Pregnancy
        - Emphysema
        - Epilepsy
        - Esophageal Carcinoma
        - Fibroid Uterus
        - Folliculitis
        - Frozen Shoulder
        - Functional Constipation
        - Functional Dyspepsia
        - Gallstones
        - Gastro-esophageal Reflux Disease (GERD)
        - Gastrointestinal Tuberculosis
        - Giardiasis
        - Gingivitis
        - Glaucoma
        - Glossitis
        - Gout
        - Graves Disease
        - Hand Foot Mouth Disease (HFMD)
        - Head Injury
        - Hemophilia
        - Hepatitis E Infection
        - Herpes Simplex
        - HIV
        - Hypertension
        - Hypothyroidism
        - Impetigo
        - Infectious Mononucleosis
        - Inflammatory Bowel Disease
        - Influenza
        - Injury
        - Injury of Sclera
        - Insect bite
        - Insomnia
        - Iron Deficiency Anemia
        - Kala Azar
        - Laceration
        - Lead Poisoning
        - Leg Ulcer
        - Leprosy
        - Liver Abscess
        - Liver Cancer
        - Liver Secondaries
        - Lower Respiratory Tract Infection (LRTI)
        - Ludwig's Angina
        - Lung Abscess
        - Lymphoma
        - Malaria
        - Malnutrition
        - Mastitis
        - Meningitis
        - Menorrhagia
        - Migraine
        - Mitral Regurgitation
        - Mitral Stenosis
        - Muscle Sprain
        - Myocardial Infarction
        - Myxedema
        - Neonatal Herpes Simplex
        - Nephrotic Syndrome
        - Nevi
        - Obesity
        - Obstructive Jaundice
        - Oligomenorrhea
        - Osteoarthritis
        - Otitis Externa
        - Pancreatic Cancer
        - Parkinsonism
        - Parotitis
        - Pelvic Inflammatory Disease
        - Pemphigoid
        - Pemphigus
        - Peptic Ulcer
        - Pericardial Effusion
        - Pityriasis Alba
        - Plantar Faciitis
        - Pneumonia
        - Pneumonia with HIV Infection
        - Pneumothorax
        - Polycystic Ovary
        - Post-streptococcal Glomerulonephritis
        - Pregnancy
        - Presbyacusis
        - Primary Biliary Cirrhosis
        - Primary Dysmenorrhea
        - Primary Dysmenorrhoea
        - Primary Infertility
        - Psoriasis
        - Psychogenic Erectile Dysfunction
        - Pustule
        - Rheumatoid Arthritis
        - Rhythm Disorders
        - Scabies
        - Sciatica
        - Scrub Typhus
        - Secondary Amenorrhoea
        - Shingles
        - Smell Disorder
        - Squamous Cell Carcinoma
        - Stress Headache
        - Stroke
        - Syncope
        - Syphilis
        - Tension Headache
        - Tetralogy of Fallot (Cyanotic Congenital Heart Disease)
        - Thrombophlebitis
        - Tinea Capitis
        - Tinea Corporis
        - Tinea Cruris
        - Tinea Mannum
        - Tinea Pedis
        - Tinea Versicolor
        - Tuberculosis
        - Tuberculous Lymphadenitis co-occurent with HIV
        - Tuberculous Meningitis
        - Tuberculous Pleural Effusion
        - Typhoid Fever
        - Unstable Angina
        - Upper Gatrointestinal Bleeding
        - Upper Respiratory Tract Infection (URTI)
        - Urinary Tract Infection
        - Vaginitis
        - Viral Fever

        For high to moderate likelihood diagnosis under the rationale mention the clinical relevance and features, any recent infection or preceeding infection, and relevance to rural India, keeping in mind the limitations of a telemedicine consultation.
        For a low likelihood diagnosis, include lack of fit reasoning under the rationale for that diagnosis.
        Please rank the differential diagnoses based on the likelihood and provide a detailed explanation for each diagnosis.
        Please remember this is a patient in rural India and use this as a consideration for the diseases likely for the patient and dont output a json
    """
else:
    question_1 = "You are a doctor with the following patient from rural India. Here is their case with the history of presenting illness, their physical exams, and demographics. What would be the top 5 differential diagnosis for this patient? For each diagnosis, include the likelihood score and a brief rationale focusing on the most relevant clinical features and relevance to a rural Indian context. Please rank the differential diagnoses based on the likelihood and provide a brief explanation for each diagnosis. DO NOT include case and question in the output anywhere. Keep rationale and conclusion very brief."

cot.load("outputs/" + args.trained_file)

def receive_new_data(new_df):
    for index,row in new_df[:].iterrows():
        # if index < 190:
        #     continue
        # else:
            print("############################################")
            case_id = row["visit_id"]
            gt_diagnosis = row["Diagnosis"]
            patient_case = row["Clinical Notes"]
            print("ROW -----> ", case_id, index)
            # Initialize response dictionary with fields matching module signatures
            if args.module == 'telemedicine':
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                    "LLM_rationale": "",
                    "LLM_conclusion": "",
                    "further_questions": "",
                    "follow_up_recommendations": ""
                }
            elif args.module == 'icd_snowmed_kb':
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                    "LLM_rationale": ""
                }
            elif args.module == 'telemedicine_icd_11':
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                    "LLM_conclusion": "",
                    "further_questions": "",
                    "follow_up_recommendations": ""

                }
            elif args.module == 'telemedicine_snomed_ct':
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                    "LLM_conclusion": "",
                    "further_questions": "",
                    # "follow_up_recommendations": ""
                }
            elif args.module == 'ddx':
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                    "LLM_rationale": "",
                    "LLM_conclusion": "",
                    "further_questions": ""
                    # "follow_up_recommendations": ""
                }
            elif args.module == 'ddx_telemedicine':
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                }
            else:
                resp = {
                    "case_id": case_id,
                    "patient_case": patient_case,
                    "gt_diagnosis": gt_diagnosis,
                    "LLM_diagnosis": "",
                    "LLM_rationale": ""
                }
            
            try:
                output = cot(case=patient_case, question=question)
                
                diagnosis_text = output.output.diagnosis

                if "diagnosis=" in diagnosis_text:
                    diagnosis_text = diagnosis_text.split("diagnosis=")[-1]
                elif "diagnosis:" in diagnosis_text:
                    diagnosis_text = diagnosis_text.split("diagnosis:")[-1]

                if '\n\n' in diagnosis_text:
                    parts = diagnosis_text.split('\n\n', 1)
                    if len(parts) > 1 and parts[1].lstrip().startswith('1.'):
                        diagnosis_text = parts[1]

                diagnosis_text = diagnosis_text.strip()
                if diagnosis_text.startswith("'") and diagnosis_text.endswith("'"):
                    diagnosis_text = diagnosis_text[1:-1]
                if diagnosis_text.startswith('"') and diagnosis_text.endswith('"'):
                    diagnosis_text = diagnosis_text[1:-1]

                print("diagnosis: ", diagnosis_text)
                
                # Set basic fields that exist in both modules
                resp["LLM_diagnosis"] = diagnosis_text
                

                # Set additional fields for telemedicine module
                if args.module == 'telemedicine' or args.module == 'icd_snowmed_kb':
                    resp["LLM_rationale"] = output.output.rationale
                    resp["further_questions"] = output.output.further_questions
                    resp["follow_up_recommendations"] = output.output.follow_up_recommendations
                    resp["LLM_conclusion"] = output.output.conclusion
                elif args.module == 'telemedicine_icd_11':
                    resp["LLM_conclusion"] = output.output.conclusion
                    resp["further_questions"] = output.output.further_questions
                    resp["follow_up_recommendations"] = output.output.follow_up_recommendations
                elif args.module == 'telemedicine_ten':
                    resp["further_questions"] = output.output.further_questions
                    resp["follow_up_recommendations"] = output.output.follow_up_recommendations
                elif args.module == 'telemedicine_snomed_ct':
                    resp["LLM_rationale"] = output.output.rationale
                    resp["LLM_conclusion"] = output.output.conclusion
                    resp["further_questions"] = output.output.further_questions
                    # resp["follow_up_recommendations"] = output.output.follow_up_recommendations
                elif args.module == 'ddx':
                    resp["LLM_rationale"] = output.output.rationale
                    resp["LLM_conclusion"] = output.output.conclusion
                    resp["further_questions"] = output.output.further_questions
                elif args.module == 'ddx_telemedicine':
                    resp["LLM_rationale"] = output.output.rationale
                    resp["LLM_conclusion"] = output.output.conclusion
                    resp["further_questions"] = output.output.further_questions
            except Exception as e:
                # Set module-specific empty fields based on module type
                if args.module == 'telemedicine':
                    resp["LLM_rationale"] = ""
                    resp["further_questions"] = ""
                    resp["follow_up_recommendations"] = ""
                    resp["LLM_conclusion"] = ""
                elif args.module == 'telemedicine_ten':
                    resp["further_questions"] = ""
                    resp["follow_up_recommendations"] = ""
                elif args.module == 'ddx':
                    resp["further_questions"] = ""
                    resp["LLM_conclusion"] = ""
                    resp["LLM_rationale"] = ""
                elif args.module == 'ddx_telemedicine':
                    resp["further_questions"] = ""
                    resp["LLM_conclusion"] = ""
                    resp["LLM_rationale"] = ""
                print(e)
                print("exception happened")
                resp["LLM_diagnosis"] =  ""
            yield resp

# Write data as soon as a new row is available
for new_data in receive_new_data(new_df):
    # Append the new row to the DataFrame
    new_row_df = pd.DataFrame([new_data])
    # Concatenate the new row DataFrame with the existing DataFrame
    fdf = pd.concat([df, new_row_df], ignore_index=True)
    # Write the DataFrame to a CSV file after each row
    new_row_df.to_csv(args.output_csv, mode='a', header=False, index=False)

    # Print the current state of the DataFrame
    print(fdf)
