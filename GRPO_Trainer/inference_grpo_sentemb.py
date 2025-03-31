import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import logging
from tqdm import tqdm
import config
from agent import GRPOAgent, PolicyNetwork
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_artifacts():
    """Load the trained GRPO model, sentence transformer, and label encoder."""
    try:
        # Load sentence transformer model
        logging.info("Loading sentence transformer model...")
        sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        state_dim = sentence_transformer.get_sentence_embedding_dimension()
        logging.info(f"Sentence transformer loaded. Embedding dimension: {state_dim}")

        # Load label encoder
        logging.info("Loading label encoder...")
        label_encoder = joblib.load('models/grpo_label_encoder.joblib')
        num_actions = len(label_encoder.classes_)
        logging.info(f"Label encoder loaded. Number of classes: {num_actions}")

        # Initialize and load GRPO model
        logging.info("Initializing GRPO model...")
        agent = GRPOAgent(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dims=config.HIDDEN_DIMS,
            learning_rate=config.LEARNING_RATE,
            group_size=config.GROUP_SIZE,
            kl_epsilon=config.KL_CONSTRAINT_EPSILON,
            device=config.DEVICE
        )
        agent.load_model(config.MODEL_SAVE_PATH)
        logging.info("GRPO model loaded successfully")

        return agent, sentence_transformer, label_encoder

    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise

def predict_diagnosis(agent, sentence_transformer, label_encoder, clinical_notes):
    """
    Predict diagnosis for a single clinical note.
    
    Args:
        agent: Loaded GRPO agent
        sentence_transformer: Loaded sentence transformer model
        label_encoder: Loaded label encoder
        clinical_notes: Clinical notes text
    
    Returns:
        str: Predicted diagnosis
        float: Confidence score
    """
    # Generate sentence embedding
    with torch.no_grad():
        embedding = sentence_transformer.encode(clinical_notes, convert_to_tensor=True)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(config.DEVICE)

    # Get action distribution
    agent.policy_net.eval()
    with torch.no_grad():
        action_dist = agent.policy_net.get_action_distribution(embedding)
        action_probs = action_dist.probs.squeeze(0)
        
        # Get predicted action and confidence
        predicted_idx = torch.argmax(action_probs).item()
        confidence = action_probs[predicted_idx].item()
        
        # Convert index to label
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        
        return predicted_label, confidence

def evaluate_model(agent, sentence_transformer, label_encoder, test_data):
    """
    Evaluate the model on test data.
    
    Args:
        agent: Loaded GRPO agent
        sentence_transformer: Loaded sentence transformer model
        label_encoder: Loaded label encoder
        test_data: DataFrame containing test data
    
    Returns:
        float: Accuracy score
        dict: Detailed metrics
    """
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
        clinical_notes = row[config.TEXT_COLUMN]
        true_label = row[config.LABEL_COLUMN]
        
        predicted_label, confidence = predict_diagnosis(
            agent, sentence_transformer, label_encoder, clinical_notes
        )
        
        all_predictions.append(predicted_label)
        all_true_labels.append(true_label)
        
        if predicted_label == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    
    return accuracy, {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }

def process_input_file(input_file):
    """Process input file containing clinical notes."""
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
            if config.TEXT_COLUMN not in df.columns:
                raise ValueError(f"CSV file must contain a column named '{config.TEXT_COLUMN}'")
            return df[config.TEXT_COLUMN].tolist()
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                return [f.read().strip()]
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        raise

def main():
    """Main function to run inference."""
    parser = argparse.ArgumentParser(description='Run inference with GRPO model using sentence embeddings')
    parser.add_argument('--input', type=str, help='Input file containing clinical notes (CSV or text file)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on test set')
    args = parser.parse_args()

    try:
        # Load models and artifacts
        agent, sentence_transformer, label_encoder = load_model_and_artifacts()

        if args.evaluate:
            # Load test data and evaluate
            logging.info("Loading test data...")
            test_data = pd.read_csv(config.DATA_PATH)
            test_size = int(len(test_data) * config.TEST_SPLIT_RATIO)
            test_data = test_data.iloc[-test_size:]
            logging.info(f"Loaded {len(test_data)} test samples")

            accuracy, metrics = evaluate_model(agent, sentence_transformer, label_encoder, test_data)
            
            logging.info("\nDetailed Metrics:")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value}")

        elif args.input:
            # Process input file
            clinical_notes_list = process_input_file(args.input)
            
            logging.info(f"\nProcessing {len(clinical_notes_list)} clinical notes...")
            for i, notes in enumerate(clinical_notes_list, 1):
                print(f"\nClinical Note {i}:")
                print("-" * 50)
                print(notes[:200] + "..." if len(notes) > 200 else notes)
                print("-" * 50)
                
                predicted_label, confidence = predict_diagnosis(
                    agent, sentence_transformer, label_encoder, notes
                )
                print(f"Predicted Diagnosis: {predicted_label}")
                print(f"Confidence: {confidence:.4f}")
                print("-" * 50)

        else:
            # Interactive mode
            print("\nEnter clinical notes (Press Ctrl+D or Ctrl+Z to finish):")
            try:
                notes = []
                while True:
                    try:
                        line = input()
                        notes.append(line)
                    except EOFError:
                        break
                clinical_notes = "\n".join(notes)
                
                if clinical_notes.strip():
                    print("\nProcessing clinical notes...")
                    print("-" * 50)
                    print(clinical_notes[:200] + "..." if len(clinical_notes) > 200 else clinical_notes)
                    print("-" * 50)
                    
                    predicted_label, confidence = predict_diagnosis(
                        agent, sentence_transformer, label_encoder, clinical_notes
                    )
                    print(f"Predicted Diagnosis: {predicted_label}")
                    print(f"Confidence: {confidence:.4f}")
                    print("-" * 50)
                else:
                    print("No input provided.")
            except KeyboardInterrupt:
                print("\nInput interrupted.")

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise

if __name__ == "__main__":
    main() 