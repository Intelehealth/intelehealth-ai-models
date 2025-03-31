# predict.py

import torch
import joblib
import numpy as np
import logging
import argparse
import os
from GRPO_Trainer.agent import PolicyNetwork # Assuming agent.py is in GRPO_Trainer
import GRPO_Trainer.config as config # Assuming config.py is in GRPO_Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration --- #
# Ensure paths are relative to where the script is run or use absolute paths
MODEL_DIR = "models" # Directory where vectorizer and encoder are saved
VECTORIZER_PATH = os.path.join(MODEL_DIR, f"{config.STATE_REPRESENTATION}_vectorizer.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
MODEL_WEIGHTS_PATH = config.MODEL_SAVE_PATH # From config.py
DEVICE = torch.device(config.DEVICE)

def load_artifacts(vectorizer_path, encoder_path):
    """Loads the saved TF-IDF vectorizer and label encoder."""
    logging.info(f"Loading vectorizer from {vectorizer_path}...")
    try:
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        logging.error(f"Error: Vectorizer file not found at {vectorizer_path}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading vectorizer: {e}")
        return None, None, None

    logging.info(f"Loading label encoder from {encoder_path}...")
    try:
        label_encoder = joblib.load(encoder_path)
        # Reconstruct action_map
        action_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    except FileNotFoundError:
        logging.error(f"Error: Label encoder file not found at {encoder_path}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading label encoder: {e}")
        return None, None, None

    logging.info("Artifacts loaded successfully.")
    return vectorizer, label_encoder, action_map

def load_model(weights_path, state_dim, num_actions, hidden_dims, device):
    """Loads the trained PolicyNetwork model."""
    logging.info(f"Loading model weights from {weights_path}...")
    model = PolicyNetwork(state_dim, num_actions, hidden_dims).to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval() # Set model to evaluation mode
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Error: Model weights file not found at {weights_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        return None
    return model

def predict_diagnosis(text, vectorizer, model, action_map, device):
    """Preprocesses text and predicts the most likely diagnosis."""
    if not text or not isinstance(text, str):
        logging.error("Input text must be a non-empty string.")
        return None, None

    # 1. Preprocess text using the loaded vectorizer
    # Vectorizer expects a list/iterable of documents
    try:
        text_vector = vectorizer.transform([text])
        # Convert sparse matrix to dense tensor for the model
        # Ensure it's float32 as expected by nn.Linear
        state_tensor = torch.tensor(text_vector.toarray(), dtype=torch.float32).to(device)
    except Exception as e:
        logging.error(f"Error transforming input text: {e}")
        return None, None

    # 2. Predict using the model
    with torch.no_grad(): # Disable gradient calculations for inference
        try:
            action_dist = model.get_action_distribution(state_tensor)
            probabilities = action_dist.probs.squeeze() # Remove batch dim
            # Get the index of the highest probability
            predicted_action_index = torch.argmax(probabilities).item()
            # Get the probability of the predicted action
            predicted_probability = probabilities[predicted_action_index].item()
        except Exception as e:
            logging.error(f"Error during model prediction: {e}")
            return None, None

    # 3. Map index to diagnosis name
    predicted_diagnosis = action_map.get(predicted_action_index, "Unknown Diagnosis Index")

    logging.info(f"Predicted Diagnosis: {predicted_diagnosis} (Probability: {predicted_probability:.4f})")
    return predicted_diagnosis, predicted_probability


def main():
    parser = argparse.ArgumentParser(description="Predict diagnosis from clinical note text using a trained GRPO model.")
    parser.add_argument("clinical_note", type=str, help="The clinical note text to predict diagnosis for.")
    args = parser.parse_args()

    # --- Load Artifacts --- #
    vectorizer, label_encoder, action_map = load_artifacts(VECTORIZER_PATH, ENCODER_PATH)
    if not vectorizer or not label_encoder:
        return # Exit if artifacts failed to load

    # --- Determine Model Dimensions --- #
    # State dimension comes from the TF-IDF vectorizer
    state_dim = vectorizer.n_features_
    # Number of actions comes from the label encoder
    num_actions = len(label_encoder.classes_)
    logging.info(f"Determined state_dim={state_dim}, num_actions={num_actions}")

    # --- Load Model --- #
    model = load_model(MODEL_WEIGHTS_PATH, state_dim, num_actions, config.HIDDEN_DIMS, DEVICE)
    if not model:
        return # Exit if model failed to load

    # --- Predict --- #
    predict_diagnosis(args.clinical_note, vectorizer, model, action_map, DEVICE)


if __name__ == "__main__":
    main() 