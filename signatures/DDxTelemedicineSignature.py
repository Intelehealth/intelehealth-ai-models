import dspy

class DDxTelemedicineSignature(dspy.Signature):
    """
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
    case = dspy.InputField(desc="case with the history of presenting illness, physical exams, and demographics of a patient")
    question = dspy.InputField(desc="the patient prompt question")
    diagnosis = dspy.OutputField(desc="Top five differential diagnosis for the patient ranked by likelhood")
    rationale = dspy.OutputField(desc="detailed chain of thought reasoning for each of these top 5 diagnosis predicted. don't include the case and question in this one.")
    conclusion = dspy.OutputField(desc="Final conclusion on top likely diagnosis considering all rationales")
    further_questions = dspy.OutputField(desc="further questions to ask the patient ONLY if the data was not sufficient. don't include any questions outside scope of provided patient data.") 