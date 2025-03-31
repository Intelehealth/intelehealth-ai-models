# train.py

import torch
import numpy as np
import logging
from tqdm import tqdm # For progress bars
import time
import os # For path joining and makedirs
import joblib # For saving artifacts

import config
import utils
from environment import DiagnosisEnvironment
from agent import GRPOAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_agent(agent, eval_env):
    """Evaluates the agent's performance on the evaluation environment."""
    logging.info("Starting evaluation...")
    agent.policy_net.eval()  # Set policy network to evaluation mode
    total_correct = 0
    total_cases = 0

    eval_env.set_case_index(0) # Reset environment to the start of the test set

    with torch.no_grad():
        for i in tqdm(range(eval_env.num_cases), desc="Evaluating"):
            state = eval_env.reset()
            if state is None:
                break # Reached end of evaluation data

            true_diagnosis = eval_env.get_current_true_diagnosis() # Get truth before env increments index in step()

            # In evaluation, typically select the most likely action
            action_dist = agent.policy_net.get_action_distribution(state.unsqueeze(0)) # Add batch dim
            predicted_action = torch.argmax(action_dist.probs, dim=1).item() # Get index of max probability

            # Simulate the step to increment index, reward calculation isn't strictly needed here
            # but good practice to keep flow similar, or just manually increment index
            _, _, _, info = eval_env.step(predicted_action) # action doesn't matter for index increment here
            # true_diagnosis_eval = info['ground_truth'] # Can also get truth here

            if predicted_action == true_diagnosis:
                total_correct += 1
            total_cases += 1

    accuracy = total_correct / total_cases if total_cases > 0 else 0
    logging.info(f"Evaluation Complete. Accuracy: {accuracy:.4f} ({total_correct}/{total_cases})")
    agent.policy_net.train() # Set back to training mode
    return accuracy


def main():
    # --- Setup ---
    logging.info(f"Using device: {config.DEVICE}")
    device = torch.device(config.DEVICE)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if config.DEVICE == "cuda" or config.DEVICE == "mps":
        torch.cuda.manual_seed_all(config.RANDOM_SEED) # if using CUDA

    # --- Load and Prepare Data ---
    try:
        # vectorizer will be None if using sentence-transformer
        X_train, X_test, y_train, y_test, vectorizer, label_encoder, action_map = utils.load_and_preprocess_data(
            config.DATA_PATH,
            config.TEXT_COLUMN,
            config.LABEL_COLUMN,
            test_size=config.TEST_SPLIT_RATIO,
            random_state=config.RANDOM_SEED,
            state_rep_type=config.STATE_REPRESENTATION,
            max_features=config.TFIDF_MAX_FEATURES
        )
    except Exception as e:
        logging.error(f"Failed to load or preprocess data: {e}")
        return # Exit if data loading fails

    # --- Save GRPO-specific Preprocessing Artifacts ---
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    # Save vectorizer (if it exists and is for TF-IDF)
    if config.STATE_REPRESENTATION == 'tfidf' and vectorizer is not None:
        vectorizer_save_path = os.path.join("models", "grpo_tfidf_vectorizer.joblib")
        joblib.dump(vectorizer, vectorizer_save_path)
        logging.info(f"GRPO TF-IDF vectorizer saved to {vectorizer_save_path}")
    # Save label encoder
    encoder_save_path = os.path.join("models", "grpo_label_encoder.joblib")
    joblib.dump(label_encoder, encoder_save_path)
    logging.info(f"GRPO label encoder saved to {encoder_save_path}")

    # --- Initialize Environment and Agent ---
    train_env = DiagnosisEnvironment(X_train, y_train, action_map, device)
    # Create a separate environment for evaluation if needed
    eval_env = DiagnosisEnvironment(X_test, y_test, action_map, device)

    # Log the number of training cases
    num_training_cases = train_env.num_cases
    logging.info(f"Number of training cases: {num_training_cases}")

    state_dim = train_env.get_state_dim()
    num_actions = train_env.get_num_actions()
    logging.info(f"State dimension: {state_dim}, Number of actions: {num_actions}")

    agent = GRPOAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=config.HIDDEN_DIMS,
        learning_rate=config.LEARNING_RATE,
        group_size=config.GROUP_SIZE,
        kl_epsilon=config.KL_CONSTRAINT_EPSILON,
        device=device
    )

    # --- Training Loop ---
    logging.info("Starting training...")
    start_time = time.time()
    total_steps = 0 # Can track total updates
    losses = []
    kl_divergences = []

    # We can train for a fixed number of "episodes" or passes (epochs)
    # Let's define an episode as processing one patient case
    # num_training_cases = train_env.num_cases # Moved up
    num_epochs = config.NUM_EPISODES // num_training_cases if num_training_cases > 0 else 0 # Approximate epochs
    logging.info(f"Training for {config.NUM_EPISODES} steps (~{num_epochs} epochs)...")

    for episode in tqdm(range(config.NUM_EPISODES), desc="Training Progress"):
        # Reset environment to the start if we've processed all cases in this epoch
        if train_env.current_case_index >= num_training_cases and num_training_cases > 0:
             # --- Shuffle data at the beginning of each epoch ---
             logging.info(f"Epoch finished. Shuffling training data...")
             permutation = np.random.permutation(num_training_cases)
             # Check if X_train is a sparse matrix or numpy array and handle appropriately
             if hasattr(X_train, 'tocsr'): # Check if it looks like a SciPy sparse matrix
                 X_train = X_train[permutation]
             elif isinstance(X_train, np.ndarray):
                 X_train = X_train[permutation]
             else:
                 # Assuming list or other indexable, handle potential errors if not
                 try:
                     X_train = [X_train[i] for i in permutation]
                 except TypeError:
                     logging.warning("X_train type not recognized for shuffling, attempting NumPy conversion.")
                     X_train = np.array(X_train)[permutation]
             # Assuming y_train is array-like (e.g., numpy array, list, pandas Series)
             if hasattr(y_train, 'iloc'): # Handle pandas Series
                 y_train = y_train.iloc[permutation]
             elif isinstance(y_train, np.ndarray):
                  y_train = y_train[permutation]
             else:
                 # Assuming list or other indexable
                 try:
                     y_train = [y_train[i] for i in permutation]
                 except TypeError:
                     logging.warning("y_train type not recognized for shuffling, attempting NumPy conversion.")
                     y_train = np.array(y_train)[permutation]

             # The environment uses the current_case_index to access data,
             # so modifying X_train/y_train in place and resetting the index should work.
             # If env caches data internally, this might need adjustment (e.g., env.update_data(X_train, y_train))
             # ------------------------------------------------------

             train_env.set_case_index(0) # Start next epoch
             logging.info(f"Starting next pass through shuffled training data.")
             # Optionally shuffle training data here if converting X_train/y_train to tensors earlier

        # 1. Get current state (patient case)
        state = train_env.reset() # Gets state at current_case_index
        if state is None: # Should not happen with the reset logic above, but safety check
             logging.warning("Environment returned None state during training. Skipping step.")
             continue

        true_diagnosis = train_env.get_current_true_diagnosis() # Get ground truth before env increments index

        # 2. Sample an action group from the policy
        # state might need unsqueeze(0) if agent expects batch dim, but let's check agent first
        # agent.select_action_group handles unsqueezing
        action_group_indices, _ = agent.select_action_group(state) # Indices are on device

        # 3. Simulate interaction & get rewards for the group
        # This requires knowing the ground truth for the *current* state
        rewards_in_group = []
        for action_index in action_group_indices:
            action_idx = action_index.item() # Convert tensor index to int
            if action_idx == true_diagnosis:
                reward = config.CORRECT_DIAGNOSIS_REWARD
            else:
                reward = config.INCORRECT_DIAGNOSIS_PENALTY
            rewards_in_group.append(reward)

        # 4. Update the policy using GRPO
        # The agent's update function requires the state, action indices, and rewards
        loss, kl_div = agent.update_policy(state, action_group_indices, rewards_in_group)
        losses.append(loss)
        kl_divergences.append(kl_div)
        total_steps += 1

        # Increment the environment's case index (implicitly done by agent update if step was called)
        # Since we didn't call env.step(), manually increment
        train_env.current_case_index += 1 # Manually move to next case

        # 5. Logging and Periodic Evaluation
        if episode % config.LOG_INTERVAL == 0 and episode > 0:
            avg_loss = np.mean(losses[-config.LOG_INTERVAL:])
            avg_kl = np.mean(kl_divergences[-config.LOG_INTERVAL:])
            elapsed_time = time.time() - start_time
            logging.info(f"Episode: {episode}/{config.NUM_EPISODES} | Avg Loss: {avg_loss:.4f} | Avg KL: {avg_kl:.6f} | Time: {elapsed_time:.2f}s")

            # Optional: Perform evaluation periodically
            evaluate_agent(agent, eval_env)
            agent.policy_net.train() # Ensure agent is back in training mode after eval

    # --- End of Training ---
    training_time = time.time() - start_time
    logging.info(f"Training finished in {training_time:.2f} seconds.")

    # --- Final Evaluation ---
    final_accuracy = evaluate_agent(agent, eval_env)
    logging.info(f"Final Test Accuracy: {final_accuracy:.4f}")

    # --- Save Model ---
    agent.save_model(config.MODEL_SAVE_PATH)
    logging.info(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main() 