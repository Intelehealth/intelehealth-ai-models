# agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
import numpy as np
import logging

import config

class PolicyNetwork(nn.Module):
    """
    Neural Network to parameterize the policy (Actor).
    Maps state representation to a probability distribution over actions.
    """
    def __init__(self, state_dim, num_actions, hidden_dims):
        super(PolicyNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.DROPOUT_RATE))  # Add dropout after each hidden layer
            input_dim = h_dim
        # Output layer: logits for each action
        layers.append(nn.Linear(input_dim, num_actions))
        self.network = nn.Sequential(*layers)
        logging.info(f"Policy Network initialized with hidden dims: {hidden_dims}")
        logging.info(f"Input dim: {state_dim}, Output dim: {num_actions}")


    def forward(self, state):
        """
        Forward pass to get action logits.

        Args:
            state (torch.Tensor): Input state tensor. Shape (batch_size, state_dim)

        Returns:
            torch.Tensor: Logits for each action. Shape (batch_size, num_actions)
        """
        return self.network(state)

    def get_action_distribution(self, state):
        """
        Gets the categorical distribution over actions for a given state.

        Args:
            state (torch.Tensor): Input state tensor. Shape (batch_size, state_dim)

        Returns:
            torch.distributions.Categorical: Probability distribution over actions.
        """
        logits = self.forward(state)
        # Use softmax to convert logits to probabilities
        action_probs = F.softmax(logits, dim=-1)
        # Add small epsilon for numerical stability
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        dist = Categorical(probs=action_probs)
        return dist


class GRPOAgent:
    """
    Implements the Group-Relative Policy Optimization (GRPO) agent.
    """
    def __init__(self, state_dim, num_actions, hidden_dims, learning_rate, group_size, kl_epsilon, device):
        """
        Initializes the GRPO agent.

        Args:
            state_dim (int): Dimensionality of the state space.
            num_actions (int): Number of possible actions.
            hidden_dims (list[int]): Dimensions of hidden layers in the policy network.
            learning_rate (float): Learning rate for the optimizer.
            group_size (int): Number of actions (K) to sample in each group.
            kl_epsilon (float): KL divergence constraint threshold.
            device (torch.device): Device to run computations on (CPU or GPU/MPS).
        """
        logging.info("Initializing GRPO Agent...")
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.group_size = group_size
        self.kl_epsilon = kl_epsilon
        self.device = device

        # Initialize policy network (actor)
        self.policy_net = PolicyNetwork(state_dim, num_actions, hidden_dims).to(device)

        # Initialize optimizer based on configuration
        optimizer_config = config.OPTIMIZER_CONFIG
        if optimizer_config['name'].lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.policy_net.parameters(),
                **optimizer_config['params']
            )
            logging.info(f"Using AdamW optimizer with weight decay: {optimizer_config['params']['weight_decay']}")
        else:  # Default to Adam
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(),
                **optimizer_config['params']
            )
            logging.info("Using Adam optimizer")

        logging.info(f"Agent configured with: LR={learning_rate}, Group Size={group_size}, KL Epsilon={kl_epsilon}")
        logging.info(f"Using device: {self.device}")


    def select_action_group(self, state):
        """
        Samples a group of K actions from the policy for the given state.

        Args:
            state (torch.Tensor): The current state tensor. Shape (state_dim,) or (1, state_dim).

        Returns:
            torch.Tensor: A tensor containing the indices of the K sampled actions. Shape (group_size,)
            torch.distributions.Categorical: The action distribution for the state.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0) # Add batch dimension if missing

        self.policy_net.eval() # Set to evaluation mode for sampling
        with torch.no_grad():
            action_dist = self.policy_net.get_action_distribution(state)
            # Sample group_size actions without replacement if possible and num_actions >= group_size
            # If group_size > num_actions, sample with replacement (though ideally K <= N)
            replace = self.group_size > self.num_actions
            if replace:
                 logging.warning(f"Group size ({self.group_size}) > number of actions ({self.num_actions}). Sampling with replacement.")
            # Multinomial expects probabilities, sample group_size times
            # The result is indices of actions
            # .squeeze(0) remove the batch dimension
            action_group_indices = action_dist.sample((self.group_size,)).squeeze(0)

            # If sampled with replacement, ensure unique actions if needed by resampling,
            # or accept duplicates based on algorithm interpretation.
            # For simplicity here, we accept potential duplicates if K > N.
            # If sampling without replacement is strictly needed and K > N, this needs adjustment.

        self.policy_net.train() # Set back to training mode
        return action_group_indices, action_dist # Return indices and the distribution it came from

    def _calculate_group_relative_advantages(self, action_group_indices, rewards, action_dist):
        """
        Calculates the group-relative advantage for each action in the sampled group.

        Args:
            action_group_indices (torch.Tensor): Indices of the sampled actions. Shape (group_size,).
            rewards (list or np.ndarray): Rewards received for each action in the group. Length group_size.
            action_dist (torch.distributions.Categorical): The action distribution used for sampling.

        Returns:
            torch.Tensor: Group-relative advantages for each action in the group. Shape (group_size,).
            torch.Tensor: Log probabilities of the sampled actions. Shape (group_size,).
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Calculate the average reward within the group
        average_group_reward = rewards_tensor.mean()

        # Calculate group-relative advantage: A_gr(s, a) = R(s, a) - avg_{a'~pi_k( |s)}[R(s, a')]
        advantages = rewards_tensor - average_group_reward

        # Get the log probability of the sampled actions under the *current* policy
        log_probs = action_dist.log_prob(action_group_indices)

        return advantages, log_probs


    def update_policy(self, state, action_group_indices, rewards):
        """
        Performs a single policy update step using GRPO.

        Args:
            state (torch.Tensor): The state for which the actions were sampled. Shape (state_dim,) or (1, state_dim).
            action_group_indices (torch.Tensor): Indices of the sampled actions. Shape (group_size,).
            rewards (list or np.ndarray): Rewards corresponding to the sampled actions.

        Returns:
            float: The calculated loss value for this update step.
            float: The calculated KL divergence for this update step.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # --- 1. Get old policy distribution (before update) ---
        self.policy_net.eval()
        with torch.no_grad():
            old_action_dist = self.policy_net.get_action_distribution(state)
        self.policy_net.train()

        # --- 2. Calculate Group-Relative Advantages ---
        advantages, old_log_probs = self._calculate_group_relative_advantages(
            action_group_indices, rewards, old_action_dist
        )

        # --- 3. Calculate the GRPO Loss ---
        current_action_dist = self.policy_net.get_action_distribution(state)
        current_log_probs = current_action_dist.log_prob(action_group_indices)
        
        # Calculate policy loss without entropy regularization
        policy_loss = -torch.mean(advantages * current_log_probs)

        # --- 4. Calculate KL Divergence (Constraint Check) ---
        kl_div = kl_divergence(old_action_dist, current_action_dist).mean()

        # --- 5. Perform Gradient Update ---
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item(), kl_div.item()


    def save_model(self, file_path):
        """Saves the policy network state dictionary."""
        logging.info(f"Saving model state to {file_path}")
        torch.save(self.policy_net.state_dict(), file_path)

    def load_model(self, file_path):
        """Loads the policy network state dictionary."""
        logging.info(f"Loading model state from {file_path}")
        try:
            self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
            self.policy_net.to(self.device) # Ensure model is on the correct device
            logging.info("Model loaded successfully.")
        except FileNotFoundError:
            logging.error(f"Error: Model file not found at {file_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading model state: {e}")
            raise 