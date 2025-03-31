# GRPO_Trainer/inference_grpo_tfidf.py

import joblib
import logging
import os
import torch
from sklearn.feature_extraction.text import TfidfVectorizer # Needed for loading

import config # To get hidden dims and device
from agent import PolicyNetwork # Load the network definition
from inference_grpo_sentemb import load_model_and_artifacts, predict_diagnosis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_DIR = "models" # For vectorizer and encoder
GRPO_MODEL_FILENAME = "grpo_model_tfidf.pth" # Relative to script location
VECTORIZER_FILENAME = "grpo_tfidf_vectorizer.joblib"
LABEL_ENCODER_FILENAME = "grpo_label_encoder.joblib"

GRPO_MODEL_PATH = GRPO_MODEL_FILENAME # Path relative to script execution
VECTORIZER_PATH = os.path.join(MODEL_DIR, VECTORIZER_FILENAME)
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, LABEL_ENCODER_FILENAME)

# --- Load Preprocessing Artifacts ---
try:
    logging.info(f"Loading TF-IDF vectorizer from: {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("Vectorizer loaded successfully.")
    state_dim = len(vectorizer.vocabulary_)

    logging.info(f"Loading label encoder from: {LABEL_ENCODER_PATH}")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    num_actions = len(label_encoder.classes_)
    logging.info("Label encoder loaded successfully.")

    logging.info(f"Determined state_dim={state_dim}, num_actions={num_actions}")

except FileNotFoundError as e:
    logging.error(f"Error loading vectorizer/encoder: {e}. Please ensure '{VECTORIZER_FILENAME}' and '{LABEL_ENCODER_FILENAME}' exist in the '{MODEL_DIR}' directory and were generated consistently.")
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred during preprocessing artifact loading: {e}")
    exit()

# --- Load GRPO Model ---
# Determine device
device = torch.device(config.DEVICE)
logging.info(f"Using device: {device}")

# Instantiate the policy network
policy_net = PolicyNetwork(state_dim, num_actions, config.HIDDEN_DIMS).to(device)

try:
    logging.info(f"Loading GRPO model state from: {GRPO_MODEL_PATH}")
    # Load the state dict, ensuring it maps to the correct device
    policy_net.load_state_dict(torch.load(GRPO_MODEL_PATH, map_location=device))
    policy_net.eval() # Set the network to evaluation mode
    logging.info("GRPO model loaded successfully.")

except FileNotFoundError:
    logging.error(f"Error: GRPO model file not found at '{GRPO_MODEL_PATH}'. Ensure it was saved correctly by train.py.")
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred loading the GRPO model state: {e}")
    exit()


def predict_diagnosis_grpo(clinical_note_text):
    """
    Predicts the diagnosis for a given clinical note using the loaded GRPO policy network.

    Args:
        clinical_note_text (str): The raw clinical note text.

    Returns:
        str: The predicted diagnosis name, or None if prediction fails.
    """
    if not isinstance(clinical_note_text, str) or not clinical_note_text.strip():
        logging.error("Input clinical note must be a non-empty string.")
        return None

    try:
        # 1. Vectorize text
        logging.info("Vectorizing input text...")
        vectorized_note_sparse = vectorizer.transform([clinical_note_text])
        # Convert sparse matrix to dense tensor for the policy network
        vectorized_note_dense = torch.tensor(vectorized_note_sparse.toarray(), dtype=torch.float32).to(device)
        logging.info(f"Input tensor shape: {vectorized_note_dense.shape}")

        # 2. Get action probabilities from the policy network
        logging.info("Getting action probabilities from GRPO model...")
        with torch.no_grad(): # Disable gradient calculations for inference
            action_dist = policy_net.get_action_distribution(vectorized_note_dense)
            probabilities = action_dist.probs # Shape (1, num_actions)

        # 3. Find the action with the highest probability
        predicted_label_numeric = torch.argmax(probabilities, dim=1).item() # Get index
        logging.info(f"Predicted numeric label: {predicted_label_numeric}")
        logging.info(f"Confidence (probability): {probabilities[0, predicted_label_numeric].item():.4f}")


        # 4. Convert numeric label to diagnosis name
        predicted_diagnosis_name = label_encoder.inverse_transform([predicted_label_numeric])

        return predicted_diagnosis_name[0]

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # Example usage (same as before):
    example_note = """
"Gender: Female

 Age: 43 years

 Chief_complaint: ► **Fatigue and General weakness** :  
• Duration - 2 महिने.  
• Timing - All day.  
• Eating habits - 2, patient is irregular in taking meals, Amount, Small, पोळी
भाजी .  
• Stressful condition - No.  
• Prior treatment sought - None.  
► **Associated symptoms** :  
• Patient reports -  
Heat / Cold intolerance - Cold intolerance, Muscle weakness, Daytime
sleepiness, Disturbed sleep, हात पायाला मुंग्या येतात  
• Patient denies -  
Fever, Chills, Night sweats, Breathlessness on exertion, Jaundice, Bleeding,
Paresthesia, Gait abnormalities, Drooping eyelids, Depressed mood, Anxiety,
Joint pain, Increase in quantity of urine output, Increase in frequency of
urination, Polydipsia, Polyphagia  

 Physical_examination: **General exams:**  
• Eyes: Jaundice-no jaundice seen, [picture taken].  
• Eyes: Pallor-normal pallor, [picture taken].  
• Arm-Pinch skin* - pinch test normal.  
• Nail abnormality-nails normal, [picture taken].  
• Nail anemia-Nails are not pale, [picture taken].  
• Ankle-no pedal oedema.  
**Neck:**  
• Thyroid swelling-no swelling in front of neck.  
**Joint:**  
• no deformity around joint.  
• joint is not swollen.  
• pain during movement.  
**Face:**  
• face appears normal.  
**Leg:**  
• Strength-both legs have equal strength.  
**Any Location:**  
• Skin Bruise:-no bruises seen.  
• Skin Rash:-no rash.  
**Hands:**  
• Holding a paper tightly between thumb and curled fingers-grip equal on both
sides.  
• Gripping two fingers tightly-grip equal on both sides.  
• Ability to spread 4 fingers apart when they are held together-grip equal on
both sides.  
**Head:** <br

 Patient_medical_history: • Pregnancy status - Not pregnant.  
• Allergies - No known allergies.  
• Alcohol use - No.  
• Smoking history - Patient denied/has no h/o smoking.  
• Medical History - None.  
• Drug history - No recent medication.  

 Family_history: •Do you have a family history of any of the following? : None.  

 Vitals:- 

Sbp: 110.0

 Dbp: 80.0

 Pulse: 102.0

 Temperature: 36.94 'C

 Weight: 47.35 Kg

 Height: 155.0 cm

 BMI: 19.71

 RR: 21.0

 SPO2: 100.0

 HB: Null

 Sugar_random: Null

 Blood_group: Null

 Sugar_pp: Null

 Sugar_after_meal: Null


    """

    print("-" * 30)
    print(f"Input Clinical Note:\n{example_note}")
    print("-" * 30)

    predicted_diagnosis = predict_diagnosis_grpo(example_note)

    if predicted_diagnosis:
        print(f"Predicted Diagnosis (GRPO): {predicted_diagnosis}")
    else:
        print("Prediction failed.")
    print("-" * 30)

    example_note_2 = """
"Gender: Female

 Age: 25 years

 Chief_complaint: ► **Fatigue and General weakness** :  
• Duration - 1 महिने.  
• Timing - All day.  
• Eating habits - 2 - patient is irregular in taking meals. Amount - Small.  
• Stressful condition - No.  
• Prior treatment sought - None.  
• Additional information - जेवण जात नाहीत थकवा येतो .  
► **Associated symptoms** :  
• Patient reports -  
Muscle weakness  
• Patient denies -  
Fever, Chills, Night sweats, Breathlessness on exertion, Heat / Cold
intolerance, Jaundice, Daytime sleepiness, Bleeding, Paresthesia, Gait
abnormalities, Drooping eyelids, Depressed mood, Anxiety, Joint pain, Increase
in quantity of urine output, Increase in frequency of urination, Polydipsia,
Polyphagia  

 Physical_examination: **General exams:**  
• Eyes: Jaundice-no jaundice seen, [picture taken].  
• Eyes: Pallor-normal pallor, [picture taken].  
• Arm-Pinch skin* - pinch test normal.  
• Nail abnormality-nails normal, [picture taken].  
• Nail anemia-Nails are not pale.  
• Ankle-no pedal oedema, [picture taken].  
**Joint:**  
• no deformity around joint.  
• joint is not swollen.  
• no pain during movement.  
**Leg:**  
• Strength-both legs have equal strength.  
**Any Location:**  
• Skin Bruise:-no bruises seen.  
• Skin Rash:-no rash.  
**Hands:**  
• Holding a paper tightly between thumb and curled fingers-grip equal on both
sides.  
• Gripping two fingers tightly-grip equal on both sides.  
• Ability to spread 4 fingers apart when they are held together-grip equal on
both sides.  
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

Sbp: 100.0

 Dbp: 67.0

 Pulse: 95.0

 Temperature: 36.72 'C

 Weight: 33.1 Kg

 Height: 150.0 cm

 BMI: 14.71

 RR: 21.0

 SPO2: 99.0

 HB: Null

 Sugar_random: Null

 Blood_group: Null

 Sugar_pp: Null

 Sugar_after_meal: Null

    """

    print(f"Input Clinical Note:\n{example_note_2}")
    print("-" * 30)
    predicted_diagnosis_2 = predict_diagnosis_grpo(example_note_2)
    if predicted_diagnosis_2:
        print(f"Predicted Diagnosis (GRPO): {predicted_diagnosis_2}")
    else:
        print("Prediction failed.")
    print("-" * 30)

    # Load models
    agent, sentence_transformer, label_encoder = load_model_and_artifacts()

    # Make a prediction
    clinical_notes = "Your clinical notes here..."
    predicted_label, confidence = predict_diagnosis(
        agent, sentence_transformer, label_encoder, clinical_notes
    )
    print(f"Predicted: {predicted_label}")
    print(f"Confidence: {confidence:.2f}") 