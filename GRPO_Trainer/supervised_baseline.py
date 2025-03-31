# GRPO_Trainer/supervised_baseline.py

import logging
import numpy as np
import torch # For setting seed potentially
from sklearn.linear_model import LogisticRegression # Changed back
from sklearn.metrics import accuracy_score, classification_report
import joblib

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- Setup ---
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # --- Load and Prepare Data ---
    logging.info("Loading and preprocessing data for supervised baseline...")
    try:
        X_train, X_test, y_train, y_test, vectorizer, label_encoder, action_map = utils.load_and_preprocess_data(
            config.DATA_PATH,
            config.TEXT_COLUMN,
            config.LABEL_COLUMN,
            test_size=config.TEST_SPLIT_RATIO,
            random_state=config.RANDOM_SEED,
            state_rep_type=config.STATE_REPRESENTATION, # Reads from config
            max_features=config.TFIDF_MAX_FEATURES
        )
        logging.info(f"Data loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        if config.STATE_REPRESENTATION == 'tfidf':
             logging.info(f"Number of features: {X_train.shape[1]}")
        else:
             logging.info(f"Embedding dimension: {X_train.shape[1]}")
        logging.info(f"Number of classes: {len(action_map)}")
    except Exception as e:
        logging.error(f"Failed to load or preprocess data: {e}")
        return

    # --- Train Supervised Model (Logistic Regression) ---
    logging.info("Training Logistic Regression model...")
    model = LogisticRegression(
        random_state=config.RANDOM_SEED,
        max_iter=1000,
        multi_class='ovr',
        solver='liblinear'
    )

    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # --- Evaluate Model ---
    logging.info("Evaluating model on the test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Set Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    try:
        # Attempt to get target names from label encoder
        target_names = [label_encoder.inverse_transform([label])[0] for label in unique_labels]
        report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0)
    except Exception as e:
        logging.warning(f"Could not use class names for report: {e}. Using numeric labels.")
        report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=0)
    print(report)

    # --- Save Model (Optional) ---
    # Use state representation in filename for clarity
    model_save_path = f"models/supervised_logreg_{config.STATE_REPRESENTATION}.joblib"
    joblib.dump(model, model_save_path)
    logging.info(f"Supervised model saved to {model_save_path}")

if __name__ == "__main__":
    main() 