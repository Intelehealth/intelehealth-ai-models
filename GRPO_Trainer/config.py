# config.py

import torch

# --- Data Configuration ---
DATA_PATH = "/Users/bsb/work/intelehealth-ai-models/data/NAS_V2_CleanUpDataset_v0.2.csv"
TEXT_COLUMN = "Clinical Notes"
LABEL_COLUMN = "Diagnosis"
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42 # For reproducibility

# --- Environment Configuration ---
# Define cost/reward structure if needed (beyond simple accuracy)
CORRECT_DIAGNOSIS_REWARD = 1.0
INCORRECT_DIAGNOSIS_PENALTY = -1.0 # Or a negative value for cost

# --- Agent/Model Configuration ---
# State representation details
# Options: 'tfidf', 'sentence-transformer'
STATE_REPRESENTATION = 'sentence-transformer'
# Max features for TF-IDF (ignored if using sentence-transformer)
TFIDF_MAX_FEATURES = 5000

# Policy Network (Actor) architecture
HIDDEN_DIMS = [512, 256]  # Reverted to original architecture
DROPOUT_RATE = 0.1  # Reduced dropout rate

# --- GRPO Hyperparameters ---
LEARNING_RATE = 1e-4  # Restored original learning rate
GROUP_SIZE = 3  # Reverted to original group size
KL_CONSTRAINT_EPSILON = 0.01  # Reverted to original KL constraint

# Optimizer Configuration
OPTIMIZER_CONFIG = {
    'name': 'adamw',  # Options: 'adam', 'adamw'
    'params': {
        'lr': LEARNING_RATE,
        'weight_decay': 0.01,  # L2 regularization for AdamW
        'betas': (0.9, 0.999),  # Default Adam/AdamW betas
        'eps': 1e-8,  # Default epsilon
    }
}

# --- Training Configuration ---
NUM_EPISODES = 100000
BATCH_SIZE = 64  # Restored original batch size
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
LOG_INTERVAL = 100
MODEL_SAVE_PATH = "./grpo_model_sentemb.pth"

# --- Evaluation Configuration ---
EVAL_BATCH_SIZE = 128 