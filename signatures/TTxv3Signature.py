import dspy

class TTxv3Fields(dspy.Signature):
    """
        Based on given patient history, symptoms, physical exam findings, and the diagnosis, predict the top 5 relevant medications for the patient.

        For each medication: output should adhere to this format

        Drug name-Strength-Route-Form
        Dose
        Frequency
        Duration (number)
        Duration (units)
        Instruction (Remarks)
        Rationale for the medication
        confidence - high, moderate, low

        Examples:
        Paracetamol 500 mg Oral Tablet
        1 tablet
        Thrice daily (TID)
        2
        Days
        After food
        Rationale for the medication

        Paracetamol 250 mg Oral Suspension
        5 ml
        Thrice daily (TID)
        2
        Days
        After food
        Rationale for the medication

        Mupirocin 2% Skin Ointment
        Sufficient Quantity
        Thrice daily (TID)
        1
        Week
        Apply to the affected area
        Rationale for the medication

        For medicines: always select from this list of medicines relevant to the case
        Acetylsalicylic Acid (Aspirin) 75 mg Oral Tablet
        Albendazole 400 mg Oral Tablet
        Albendazole 200 mg/5 ml Oral Suspension
        Ambroxol 30 mg/5 ml Oral Solution
        Ambroxol 7.5 mg/1 ml Oral Drops
        Ambroxol 30 mg+Levosalbutamol 1mg+Guaifensin 50 mg/5 ml Oral Solution
        Ambroxol 7.5 mg+Levosalbutamol 0.25 mg+Guaifensin 12.5 mg/1 ml Oral Drops
        Amlodipine 5 mg Oral Tablet
        Amoxicillin 250 mg Oral Capsule
        Amoxicillin 500 mg Oral Capsule
        Amoxicillin 250 mg/5 ml Oral Suspension
        Amoxicillin + Clavulanic acid 625 mg Oral Tablet
        Amoxicillin + Clavulanic acid 228.5 mg/5 ml Oral Suspension
        Ascorbic Acid (Vitamin C) 500 mg Oral Tablet (Chewable) 
        Atorvastatin 10 mg Oral Tablet 
        Azithromycin 500 mg Oral Tablet 
        Azithromycin 200 mg/5 ml Oral Suspension
        B-Complex (Multivitamin) Oral Capsule
        Betamethasone Valerate 0.05% Skin Cream 
        Bevon (Multivitamin) Oral Solution
        Bevon (Multivitamin) Oral Drops
        Bisacodyl 5 mg Oral Tablet 
        Calamine Skin Lotion
        Calcium Carbonate 625 mg Oral Tablet 
        Calcimax-P 150 mg/5 ml Oral Suspension
        Cefixime 50 mg/5 ml Oral Suspension
        Cefixime 100 mg Oral Tablet 
        Cefixime 200 mg Oral Tablet 
        Cetirizine 5 mg/5 ml Oral Solution
        Cetirizine 10 mg Oral Tablet 
        Chloroquine 50 mg/5 ml Oral Suspension
        Chloroquine 150 mg Oral Tablet
        Chlorpheniramine Maleate 2mg+Phenylephrine 5mg/1 ml Oral Drops
        Cholecalciferol 400 IU/1 ml Oral Drops
        Cholecalciferol 1000 IU Oral Sachet
        Ciprofloxacin 0.3% Eye/Ear Drops 
        Ciprofloxacin 250 mg Oral Tablet 
        Ciprofloxacin 500 mg Oral Tablet 
        Clotrimazole 1% Skin Absorbent Dusting Powder
        Clotrimazole 1% Ear Drops 
        Clotrimazole 1% Skin Lotion 
        Clotrimazole 1% Skin Cream
        Clotrimazole 100 mg Pessary (Vaginal Tablet) 
        Clotrimazole 1% Mouth Paint 
        Colic aid 40 mg Oral Drops
        Co-trimoxazole (80 mg + 400 mg) Oral Tablet 
        Co-trimoxazole (20 mg + 100 mg) Oral Tablet 
        Co-trimoxazole (40 mg + 200 mg/5 ml) Oral Suspension
        Dextromethorphan 15 mg/5 ml Oral Solution
        Diclofenac 50 mg Oral Tablet 
        Dicyclomine 10 mg Oral Tablet 
        Diethylcarbamazine (DEC) 120 mg/5 ml Oral Solution
        Diethylcarbamazine (DEC) 100 mg Oral Tablet 
        Domperidone 1 mg/1 ml Oral Suspension
        Domperidone 10 mg Oral Tablet 
        Doxycycline 100 mg Oral Capsule 
        Enalapril 5 mg Oral Tablet 
        Folic acid 1 mg Oral Tablet 
        Folic acid 5 mg Oral Tablet 
        Framycetin 1% Skin Cream 
        Fluconazole 100 mg Oral Tablet 
        Glimepiride 2 mg Oral Tablet 
        Hydrochlorothiazide 12.5 mg Oral Tablet 
        Hydrochlorothiazide 25 mg Oral Tablet 
        Ibuprofen 400 mg Oral Tablet 
        Ibuprofen 100 mg/5 ml Oral Suspension
        IFA (Ferrous Salt 45 mg + Folic acid 400 mcg) Oral Tablet 
        IFA (Ferrous Salt 100 mg+ Folic acid 500 mcg) Oral Tablet 
        IFA (Ferrous Salt 20 mg+ Folic acid 100 mcg)/1 ml Oral Drops
        Junior Lanzol 15 mg Oral Dispersible Tablet (DT) 
        Lactulose 10 g/15 ml Oral Solution
        Levocetirizine 5 mg Oral Tablet 
        Levocetirizine  2.5 mg/5 ml Oral Solution
        Levothyroxine 25 mcg Oral Tablet 
        Levothyroxine 50 mcg Oral Tablet 
        Levothyroxine 100 mcg Oral Tablet 
        Magnesium Hydroxide (Milk of Magnesia) 8% Oral Suspension
        Metformin 500 mg Oral Tablet 
        Metformin 500 mg SR Oral Tablet 
        Metronidazole 200 mg Oral Tablet 
        Metronidazole 400 mg Oral Tablet 
        Mupirocin 2% Skin Ointment
        Norfloxacin 400 mg Oral Tablet
        Omeprazole 20 mg Oral Capsule
        Ondansetron 4 mg Oral Tablet
        Ondansetron 2 mg/5 ml Oral Solution
        Oral Rehydration Salts (WHO ORS, Large)
        Oral Rehydration Salts (WHO ORS, Small)
        Paracetamol 100mg/1ml Oral Drops 
        Paracetamol 125 mg/5ml Oral Suspension
        Paracetamol 250 mg/5ml Oral Suspension
        Paracetamol 500 mg Oral Tablet 
        Paradichlorobenzene 2%+Chlorbutol 5%+Turpentine Oil 15%+ Lidocaine 2% Ear Drops
        Permethrin 5% Skin Cream 
        Salbutamol 2 mg Oral Tablet 
        Salbutamol 2 mg/5 ml Oral Solution
        Saline 0.65% Nasal Drops 
        Silver Sulphadiazine 1% Skin Cream 
        Telmisartan 40 mg Oral Tablet 
        Xylometazoline 0.05% Nasal Drops 
        Xylometazoline 0.1% Nasal Drops 
        Zinc Oxide 10% Skin Cream 
        Zinc Sulphate 20 mg Oral Dispersible Tablet (DT)

        For instruction remarks, always select appropriate from this list:
        After food
        Before food
        With food
        At bedtime
        One hour before food
        Two hours after food
        Before breakfast
        After breakfast
        Before dinner
        After dinner
        On empty stomach
        In the morning
        In the evening
        In the afternoon
        Apply lotion below the neck over the whole body at bedtime
        Add 1 sachet to 1 liter of boiled and cooled drinking water and consume within a day
        Add 1 sachet to 200 ml of boiled and cooled drinking water
        Apply to the affected area
        Apply the affected area at bedtime
        Eye (left)
        Eye (right)
        Eyes
        Ear (left)
        Ear (right)
        Ears
        Nostril (left)
        Nostril (right)
        Nostrils
        Dissolve tablet in expressed breastmilk 
        Dissolve tablet in in drinking water

        Please rank them in order of likelihood.

        For referral information, please provide the following information:
        referral_to:
            Community Health Officer (CHO)
            Medical Officer (MO)
            General Physician
            Obstetrician & Gynecologist
            Pediatrician
            General Surgeon
            Dermatologist
            ENT Specialist
            Eye Specialist
            Dental Surgeon
            Other Specialist

        referral_facility:
            Health and Wellness Centre (HWC-HSC)
            Primary Health Center (PHC) 
            Urban Health Center (UHC) 
            Community Health Center (CHC)
            Sub-district/Taluk Hospital (SDH)
            District Hospital (DH)
            Tertiary Hospital (TH)
            Private Clinic/Hospital 
            Non-Governmental Organization (NGO) Health Facility 
            Specialty Clinic 
            Mobile Health Unit (MHU)
            Other facility
            Anganwadi Centre (AW)


        priority_of_referral - should have any one of these 3 values: Elective (planned), Urgent, Emergency.
        remark - should be a short rationale about the referral.

        
        For follow_up, please provide the following information:
        follow_up_required - true or false
        next_followup_duration - should be a number along with the unit of time like days/weeks/month
        next_followup_reason - should be a short rationale about the follow up.

        For tests_to_be_done, please provide the following information:
        test_name - should be a test name.
        test_reason - should be a short rationale about the test.
        
        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Keep all responses concise and to the point.
    """
    case = dspy.InputField(desc="Patient history, symptoms, physical exam findings, and demographics")
    question = dspy.InputField(desc="Patient prompt question")
    diagnosis = dspy.InputField(desc="Diagnosis of the patient as done by the doctor")
    medication_recommendations = dspy.OutputField(desc="Top 5 relevant medications with the likelihood (high/moderate/low) with brief rationale for each of the medications.")
    medical_advice = dspy.OutputField(desc="2-3 critical medical advice if needed, also make note of adverse effects of medicine combos relevant to the case")
    tests_to_be_done = dspy.OutputField(desc="2-3 tests to be done if needed, also make note of the reason for each of the tests relevant to the patient case")
    follow_up = dspy.OutputField(desc="Next follow up duration if needed, also make note of the reason for the follow up relevant to the patient case")
    referral = dspy.OutputField(desc="Referral information if needed, also make note of the reason for the referral relevant to the patient case")
