# environment.py

import numpy as np
import torch
import logging
from scipy.sparse import csr_matrix

import config

class DiagnosisEnvironment:
    """
    A simplified environment for the differential diagnosis task using RL.
    Each episode corresponds to diagnosing a single patient case.
    The state is the preprocessed representation of the clinical notes.
    The action is the predicted diagnosis index.
    """
    def __init__(self, states, true_diagnoses, action_map, device):
        """
        Initializes the environment.

        Args:
            states (scipy.sparse.csr_matrix or np.ndarray): Preprocessed patient states (e.g., TF-IDF vectors).
            true_diagnoses (pd.Series or np.ndarray): The ground truth diagnosis labels (encoded) for each state.
            action_map (dict): Mapping from action index to diagnosis name.
            device (torch.device): Device to place tensors on.
        """
        logging.info("Initializing Diagnosis Environment...")
        if isinstance(states, csr_matrix):
            # Convert sparse matrix to dense numpy array for easier indexing,
            # handle potential memory issues if dataset is extremely large.
            # Alternatively, keep it sparse and handle slicing appropriately.
            # For moderate TF-IDF features, dense might be okay.
            logging.warning("Converting sparse TF-IDF matrix to dense. This might consume significant memory.")
            self.states = states.toarray()
        else:
             self.states = states # Assuming numpy array or similar if not sparse

        self.true_diagnoses = true_diagnoses.to_numpy() if hasattr(true_diagnoses, 'to_numpy') else true_diagnoses
        self.action_map = action_map
        self.num_actions = len(action_map)
        self.state_dim = self.states.shape[1]
        self.num_cases = self.states.shape[0]
        self.current_case_index = 0
        self.device = device

        if self.num_cases != len(self.true_diagnoses):
            raise ValueError("Mismatch between number of states and true diagnoses.")

        logging.info(f"Environment initialized with {self.num_cases} cases.")
        logging.info(f"State dimension: {self.state_dim}")
        logging.info(f"Number of actions (diagnoses): {self.num_actions}")

    def reset(self):
        """
        Resets the environment to the next patient case.
        In this setup, it just returns the state for the current case index.

        Returns:
            torch.Tensor: The state vector for the current patient case.
        """
        if self.current_case_index >= self.num_cases:
            # Optionally, could shuffle or restart from 0 if needed for multiple epochs
            # For now, just signal we're out of unique cases for one pass
            logging.warning("Reached end of dataset in environment.")
            return None # Or raise an error, or reset index

        state = self.states[self.current_case_index]
        # Convert state to a tensor and move to the specified device
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        return state_tensor

    def step(self, action):
        """
        Takes an action (diagnosis prediction) for the current patient case.

        Args:
            action (int): The index of the predicted diagnosis.

        Returns:
            tuple: (next_state, reward, done, info)
                   - next_state (None): Since each episode is one step, there's no next state.
                   - reward (float): Reward based on whether the diagnosis was correct.
                   - done (bool): Always True, as each episode ends after one action.
                   - info (dict): Contains the ground truth diagnosis for reference.
        """
        if self.current_case_index >= self.num_cases:
             raise IndexError("Step called after reaching the end of the dataset.")

        true_label = self.true_diagnoses[self.current_case_index]

        if action == true_label:
            reward = config.CORRECT_DIAGNOSIS_REWARD
        else:
            reward = config.INCORRECT_DIAGNOSIS_PENALTY

        done = True # Episode ends after one diagnosis attempt
        info = {'ground_truth': true_label, 'action_name': self.action_map.get(action, "Unknown Action")}

        # Move to the next case for the next reset() call
        self.current_case_index += 1

        # No real 'next_state' in this 1-step-per-episode setup
        next_state = None

        return next_state, reward, done, info

    def get_num_actions(self):
        """Returns the number of possible actions (diagnoses)."""
        return self.num_actions

    def get_state_dim(self):
        """Returns the dimensionality of the state vector."""
        return self.state_dim

    def get_current_true_diagnosis(self):
         """Returns the true diagnosis for the current case index (before step increments it)."""
         if self.current_case_index < self.num_cases:
             return self.true_diagnoses[self.current_case_index]
         else:
             return None

    def set_case_index(self, index):
        """Manually sets the current case index."""
        if 0 <= index < self.num_cases:
            self.current_case_index = index
        else:
            raise IndexError("Invalid case index provided.") 