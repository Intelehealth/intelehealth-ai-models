# utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib # For saving/loading sklearn objects
import logging
import os
import numpy as np # Needed for embeddings
from sentence_transformers import SentenceTransformer # Import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the sentence transformer model name (can be configured later)
# Using a common, efficient model to start
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

def load_and_preprocess_data(file_path, text_col, label_col, test_size=0.2, random_state=42, state_rep_type='sentence-transformer', max_features=None):
    """
    Loads the dataset, preprocesses text and labels, splits into train/test sets,
    and creates state representations using TF-IDF or Sentence Transformers.

    Args:
        file_path (str): Path to the CSV data file.
        text_col (str): Name of the column containing clinical notes.
        label_col (str): Name of the column containing diagnosis labels.
        test_size (float): Proportion of the dataset to use for the test set.
        random_state (int): Random seed for splitting.
        state_rep_type (str): Type of state representation ('tfidf' or 'sentence-transformer').
        max_features (int, optional): Max features for TF-IDF (ignored for sentence-transformer).

    Returns:
        tuple: (X_train, X_test, y_train, y_test, state_vectorizer/model, label_encoder, action_map)
               X represents state vectors (sparse for TFIDF, dense for sentence-transformer).
               state_vectorizer is the TFIDF vectorizer or None for sentence-transformer.
               label_encoder is the fitted LabelEncoder.
               action_map is a dictionary mapping encoded actions to diagnosis names.
    """
    logging.info(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Handle potential missing values
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str) # Ensure text column is string
    df[label_col] = df[label_col].astype(str) # Ensure label column is string
    logging.info(f"Data shape after dropping NA: {df.shape}")

    # Filter out classes with fewer than 3 samples before splitting
    label_counts = df[label_col].value_counts()
    min_samples_required = 3 # Changed from 2 to 3
    labels_to_keep = label_counts[label_counts >= min_samples_required].index
    original_rows = len(df)
    df_filtered = df[df[label_col].isin(labels_to_keep)]
    removed_rows = original_rows - len(df_filtered)
    if removed_rows > 0:
        logging.warning(f"Removed {removed_rows} rows belonging to {len(label_counts) - len(labels_to_keep)} classes with fewer than {min_samples_required} samples to allow for stratified splitting.")

    if df_filtered.empty:
        raise ValueError("No data remaining after filtering out single-instance classes.")

    # Encode labels using only the filtered data
    label_encoder = LabelEncoder()
    # Use the filtered DataFrame for fitting the encoder and splitting
    df_filtered['encoded_labels'] = label_encoder.fit_transform(df_filtered[label_col])
    num_actions = len(label_encoder.classes_)
    action_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    logging.info(f"Found {num_actions} unique diagnoses (actions) after filtering.")

    # Split filtered data
    X = df_filtered[text_col]
    y = df_filtered['encoded_labels']
    # Split the text data first
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X.tolist(), # Convert Series to list for sentence transformer
        y.tolist(),   # Convert Series to list
        test_size=test_size,
        random_state=random_state,
        stratify=y # Stratify based on original y list
    )
    logging.info(f"Data split into train/test sets. Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

    # Convert y back to pandas Series/numpy arrays if needed by downstream code
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # Create state representation
    state_vectorizer = None # TF-IDF vectorizer (set to None for sentence-transformer)
    embedding_model = None  # Sentence transformer model

    if state_rep_type == 'tfidf':
        # This part remains unchanged if you want to keep TF-IDF as an option
        logging.info(f"Creating TF-IDF state representation with max_features={max_features}...")
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_train = vectorizer.fit_transform(X_train_text) # Fit on text
        X_test = vectorizer.transform(X_test_text)       # Transform text
        state_vectorizer = vectorizer
        logging.info(f"TF-IDF transformation complete. Input dimension: {X_train.shape[1]}")

    elif state_rep_type == 'sentence-transformer':
        logging.info(f"Creating Sentence Transformer embeddings using model: {SENTENCE_TRANSFORMER_MODEL}...")
        # Load the model (will download if needed)
        # Use config device if possible
        device = config.DEVICE if hasattr(config, 'DEVICE') else 'cpu'
        embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=device)

        # Encode the text data (this might take some time)
        logging.info("Encoding training text...")
        X_train = embedding_model.encode(X_train_text, show_progress_bar=True)
        logging.info("Encoding testing text...")
        X_test = embedding_model.encode(X_test_text, show_progress_bar=True)
        logging.info(f"Sentence Transformer embedding complete. Embedding dimension: {X_train.shape[1]}")
        # We return the embedding_model itself instead of a vectorizer for potential future use
        # state_vectorizer = embedding_model # Or just return None if model not needed later

    else:
        raise ValueError(f"Unsupported state_representation_type: {state_rep_type}")

    # Return None for vectorizer if using sentence transformers
    returned_vectorizer = state_vectorizer if state_rep_type == 'tfidf' else None

    return X_train, X_test, y_train, y_test, returned_vectorizer, label_encoder, action_map

# Example usage (optional, for testing)
if __name__ == "__main__":
    try:
        # Update the call to test sentence-transformer
        X_train, X_test, y_train, y_test, _, encoder, act_map = load_and_preprocess_data(
            config.DATA_PATH,
            config.TEXT_COLUMN,
            config.LABEL_COLUMN,
            test_size=config.TEST_SPLIT_RATIO,
            random_state=config.RANDOM_SEED,
            state_rep_type='sentence-transformer' # Test sentence-transformer
            # max_features is ignored for sentence-transformer
        )
        print("Data loading and preprocessing (sentence-transformer) successful!")
        print(f"Train features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")
        print(f"Number of actions: {len(act_map)}")
        print(f"Sample action map entry: {list(act_map.items())[0]}")
        print(f"Sample y_train values: {y_train.head().tolist()}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, test TF-IDF again
        # print("\nTesting TF-IDF...")
        # try:
        #     X_train_tf, X_test_tf, y_train_tf, y_test_tf, _, _, _ = load_and_preprocess_data(
        #         config.DATA_PATH,
        #         config.TEXT_COLUMN,
        #         config.LABEL_COLUMN,
        #         test_size=config.TEST_SPLIT_RATIO,
        #         random_state=config.RANDOM_SEED,
        #         state_rep_type='tfidf',
        #         max_features=config.TFIDF_MAX_FEATURES
        #     )
        #     print("TF-IDF successful!")
        #     print(f"Train features shape: {X_train_tf.shape}")
        # except Exception as e_tf:
        #      print(f"TF-IDF test failed: {e_tf}") 