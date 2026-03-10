import logging
import os
import json
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.rl.base_policy import BasePolicy



class SARSAPolicy(BasePolicy):
    """
    SARSA (State-Action-Reward-State-Action) Policy using a neural network.
    Implements SARSA algorithm with function approximation using Gymnasium.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        device: str = "cpu",

        target_skill_bundle: List[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the SARSA policy.

        Args:
            env: Gymnasium environment
            hidden_dims: Dimensions of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum value of epsilon
            device: Device to run the network on
    
            target_skill_bundle: List of target skills
        """
        super().__init__()

        # Environment info
        self.env = env
        self.state_dim = env.observation_space.shape[0]

        # Define skill-difficulty action space
        self.skills = target_skill_bundle if target_skill_bundle else ["default_skill"]
        self.num_skills = len(self.skills)
        
        # Get difficulties from environment
        self.difficulties = sorted(env.difficulty_to_idx.keys())
        self.num_difficulties = len(self.difficulties)

        # Action space: directly questions
        self.action_dim = 3  # 3 action types: Failed, Easy, High-Aptitude

        # Hyperparameters
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.device = device



        # Local random generator for policy
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Create Q-network
        self.q_network = self._build_network(self.state_dim, self.action_dim, self.hidden_dims)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "q_losses": [],
            "episode_lengths": [],
            "success_rates": [],
            "exploration_rates": [],
        }
    
    def set_seed(self, seed: int):
        """Set new seed for the policy's random generator."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _build_network(
        self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]
    ) -> nn.Module:
        """Build the neural network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions

        Returns:
            Neural network for Q-value approximation
        """
        layers = []
        prev_dim = state_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        return nn.Sequential(*layers).to(self.device)

    def _action_to_skill_difficulty(self, action: int) -> Tuple[str, float]:
        """Convert action index to (skill, difficulty) pair."""
        skill_idx = action // self.num_difficulties
        difficulty_idx = action % self.num_difficulties
        return self.skills[skill_idx], self.difficulties[difficulty_idx]

    def _skill_difficulty_to_action(self, skill: str, difficulty: float) -> int:
        """Convert (skill, difficulty) pair to action index."""
        skill_idx = self.skills.index(skill)
        difficulty_idx = self.difficulties.index(difficulty)
        return skill_idx * self.num_difficulties + difficulty_idx

    def select_action(self, state: np.ndarray, env=None, **kwargs) -> Dict:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state
            env: Environment instance (optional, for accessing failed questions)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing action information
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            # Epsilon-greedy exploration
            if self.rng.random() < self.epsilon:
                # Random action
                action = self.rng.integers(0, self.action_dim)
                exploration = True
            else:
                # Greedy action
                action = torch.argmax(q_values, dim=1).item()
                exploration = False

        return {
            "action": action,  # Now represents action type (0-2)
            "exploration": exploration,
            "q_values": q_values.cpu().numpy(),
        }

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Update the Q-network using SARSA algorithm.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(-1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        # Get next actions using current policy (on-policy SARSA)
        next_actions_list = []
        for next_state in next_states:
            next_action_info = self.select_action(next_state)
            next_actions_list.append(next_action_info["action"])
        next_actions_tensor = (
            torch.LongTensor(next_actions_list).unsqueeze(-1).to(self.device)
        )

        # Compute next Q values using SARSA
        with torch.no_grad():
            next_q_values = self.q_network(next_states_tensor).gather(
                1, next_actions_tensor
            )
        target_q_values = (
            rewards_tensor.unsqueeze(-1)
            + (1 - dones_tensor.unsqueeze(-1)) * self.gamma * next_q_values
        )

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, model_dir: str):
        """
        Save the SARSA policy to a directory with separate files for Q-network and config.
        
        Args:
            model_dir: Directory to save the model files
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Q-network
        q_network_path = os.path.join(model_dir, "q_network.pt")
        torch.save(self.q_network.state_dict(), q_network_path)
        
        # Save configuration
        config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "skills": self.skills,
            "difficulties": self.difficulties,
            "num_skills": self.num_skills,
            "num_difficulties": self.num_difficulties,

            "seed": self.seed,
        }
        
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"SARSA policy saved to {model_dir}")
        logging.info(f"Files saved: q_network.pt, config.json")

    def load(self, model_dir: str):
        """
        Load weights into the current SARSA policy instance.
        
        Args:
            model_dir: Directory containing the saved model files
        """
        # Load Q-network weights
        q_network_path = os.path.join(model_dir, "q_network.pt")
        self.q_network.load_state_dict(torch.load(q_network_path, map_location=self.device))
        
        # Set to evaluation mode
        self.q_network.eval()
        
        logging.info(f"SARSA policy weights loaded from {model_dir}")