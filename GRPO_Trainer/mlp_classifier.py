# GRPO_Trainer/mlp_classifier.py

import logging
import numpy as np
import torch # For setting seed potentially
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE # Import SMOTE
import joblib # To save the model
import os # Added for makedirs

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- Setup ---
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # --- Load and Prepare Data ---
    # Ensure config.STATE_REPRESENTATION is set to 'sentence-transformer'
    if config.STATE_REPRESENTATION != 'sentence-transformer':
        logging.warning(f"Expected STATE_REPRESENTATION='sentence-transformer' in config.py, but found '{config.STATE_REPRESENTATION}'. Proceeding anyway...")

    logging.info("Loading and preprocessing data (expecting sentence embeddings)...")
    try:
        # Use state_rep_type from config
        X_train, X_test, y_train, y_test, _, label_encoder, action_map = utils.load_and_preprocess_data(
            config.DATA_PATH,
            config.TEXT_COLUMN,
            config.LABEL_COLUMN,
            test_size=config.TEST_SPLIT_RATIO,
            random_state=config.RANDOM_SEED,
            state_rep_type=config.STATE_REPRESENTATION,
            max_features=config.TFIDF_MAX_FEATURES # Ignored for sentence-transformer
        )
        logging.info(f"Data loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        logging.info(f"Embedding dimension: {X_train.shape[1]}")
        logging.info(f"Number of classes: {len(action_map)}")
    except Exception as e:
        logging.error(f"Failed to load or preprocess data: {e}")
        return

    # --- Train MLP Classifier ---
    logging.info("Training MLP Classifier model...")
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=config.RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
        # verbose=True
    )

    # Apply SMOTE to handle class imbalance on training data
    logging.info("Applying SMOTE to the training data...")
    # Adjust k_neighbors: Must be less than the smallest class size (which is 2 after filtering)
    smote = SMOTE(random_state=config.RANDOM_SEED, k_neighbors=1)
    # SMOTE expects 2D array for y sometimes, ensure y_train is suitable or reshape
    # However, typically it works with 1D y_train directly
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logging.info(f"SMOTE applied. Original train size: {X_train.shape[0]}, Resampled train size: {X_train_resampled.shape[0]}")
    except Exception as e:
        logging.error(f"Error applying SMOTE: {e}")
        logging.warning("Proceeding without resampling.")
        X_train_resampled, y_train_resampled = X_train, y_train # Fallback

    # Fit the model on the resampled data
    # No sample_weight needed here
    model.fit(X_train_resampled, y_train_resampled)
    logging.info("Model training complete.")

    # --- Evaluate Model ---
    logging.info("Evaluating model on the test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Set Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    try:
        target_names = [label_encoder.inverse_transform([label])[0] for label in unique_labels]
        report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0)
    except Exception as e:
        logging.warning(f"Could not use class names for report: {e}. Using numeric labels.")
        report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=0)
    print(report)

    # --- Save Model ---
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/supervised_mlp_classifier_sent_embed.joblib"
    joblib.dump(model, model_save_path)
    logging.info(f"MLP model saved to {model_save_path}")

if __name__ == "__main__":
    main() 