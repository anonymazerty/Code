import logging
import os
import json
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.rl.base_policy import BasePolicy



class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for A2C algorithm.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        # Shared layers
        self.shared_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            self.shared_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        # Actor (policy) head
        self.actor_layers = []
        self.actor_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        self.actor_layers.append(nn.ReLU())
        self.actor_layers.append(nn.Linear(hidden_dims[-1], action_dim))

        # Critic (value) head
        self.critic_layers = []
        self.critic_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        self.critic_layers.append(nn.ReLU())
        self.critic_layers.append(nn.Linear(hidden_dims[-1], 1))

        # Convert to Sequential modules
        self.shared = nn.Sequential(*self.shared_layers)
        self.actor = nn.Sequential(*self.actor_layers)
        self.critic = nn.Sequential(*self.critic_layers)

        self.to(device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            Tuple of (action_logits, value)
        """
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def get_action_and_value(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for a given state.

        Args:
            state: Input state tensor

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class A2CPolicy(BasePolicy):
    """
    A2C (Advantage Actor-Critic) Policy implementation.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 1.0, 
        entropy_coef: float = 0.0,
        value_coef: float = 0.5, 
        max_grad_norm: float = 0.5,
        device: str = "cpu",

        target_skill_bundle: List[str] = None,
        normalize_advantages: bool = False, 
        use_rms_prop: bool = True, 
        rms_prop_eps: float = 1e-5, 
    ):
        """
        Initialize the A2C policy.

        Args:
            env: Gymnasium environment
            hidden_dims: Dimensions of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for advantage computation
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run the network on
    
            target_skill_bundle: List of target skills
            normalize_advantages: Whether to normalize advantages
        """
        super().__init__()

        # Environment info
        self.env = env
        self.state_dim = env.observation_space.shape[0]

        # Define skill-difficulty action space
        self.skills = target_skill_bundle if target_skill_bundle else ["default_skill"]
        self.num_skills = len(self.skills)
        
        # Get difficulties from environment
        if hasattr(env, 'difficulty_to_idx'):
            self.difficulties = sorted(env.difficulty_to_idx.keys())
            self.num_difficulties = len(self.difficulties)
        else:
            # Fallback to default difficulties if not set
            self.difficulties = [1, 2, 3]
            self.num_difficulties = 3

        # Action space: directly questions
        self.action_dim = 3  # 3 action types: Failed, Easy, High-Aptitude

        # Hyperparameters
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.lr = learning_rate
        self.normalize_advantages = normalize_advantages
        self.use_rms_prop = use_rms_prop
        self.rms_prop_eps = rms_prop_eps



        # Create actor-critic network
        self.network = ActorCriticNetwork(
            self.state_dim, self.action_dim, hidden_dims, device
        )

        # Initialize optimizer based on use_rms_prop parameter
        if use_rms_prop:
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.lr, eps=rms_prop_eps)
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Research tracking metrics
        self.training_metrics = {
            "episode_rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "total_losses": [],
            "advantages": [],
            "advantage_means": [],
            "advantage_stds": [],
            "exploration_rates": [],
            "action_entropies": [],
            "value_predictions": [],
            "episode_lengths": [],
            "success_rates": [],
        }

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

    def compute_advantages_with_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rewards: Batch of rewards
            values: Batch of value predictions
            next_values: Batch of next state value predictions
            dones: Batch of done flags

        Returns:
            Computed advantages
        """
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, device=self.device)

    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Normalize advantages for stable training.

        Args:
            advantages: Raw advantages

        Returns:
            Normalized advantages
        """
        if advantages.numel() > 1:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def select_action(self, state: np.ndarray, env=None, **kwargs) -> Dict:
        """
        Select an action using the current policy.

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
            action, log_prob, value = self.network.get_action_and_value(state_tensor)
            action = action.item()  # Direct question index

        return {
            "action": action,  # Now represents action type (0-2)
            "exploration": False,
            "log_prob": log_prob.item(),
            "value": value.item(),
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
        Update the policy using A2C algorithm.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            **kwargs: Additional arguments

        Returns:
            Total loss value
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current action logits and values
        action_logits, values = self.network(states)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        # Get next state values for advantage calculation
        with torch.no_grad():
            _, next_values = self.network(next_states)
            next_values = next_values.squeeze(-1)

        # Calculate advantages using simple TD (no GAE for A2C)
        advantages = (
            rewards + self.gamma * next_values * (1 - dones) - values.squeeze(-1)
        )

        # Normalize advantages if enabled
        if self.normalize_advantages:
            advantages = self._normalize_advantages(advantages)

        # Calculate losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(
            values.squeeze(-1), rewards + self.gamma * next_values * (1 - dones)
        )
        entropy_loss = -dist.entropy().mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Store research metrics
        self._store_training_metrics(
            policy_loss.item(),
            value_loss.item(),
            entropy_loss.item(),
            total_loss.item(),
            advantages.detach(),
            dist.entropy().mean().item(),
            values.squeeze(-1).detach(),
        )

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }

    def _store_training_metrics(
        self,
        policy_loss: float,
        value_loss: float,
        entropy_loss: float,
        total_loss: float,
        advantages: torch.Tensor,
        action_entropy: float,
        value_predictions: torch.Tensor,
    ):
        """Store training metrics for research analysis."""
        self.training_metrics["policy_losses"].append(policy_loss)
        self.training_metrics["value_losses"].append(value_loss)
        self.training_metrics["entropy_losses"].append(entropy_loss)
        self.training_metrics["total_losses"].append(total_loss)
        self.training_metrics["advantages"].extend(advantages.cpu().numpy().tolist())
        self.training_metrics["advantage_means"].append(advantages.mean().item())
        self.training_metrics["advantage_stds"].append(advantages.std().item())
        self.training_metrics["action_entropies"].append(action_entropy)
        self.training_metrics["value_predictions"].extend(
            value_predictions.cpu().numpy().tolist()
        )

    def update_episode_metrics(
        self,
        episode_reward: float,
        episode_length: int,
        success: bool = False,
        exploration_rate: float = 0.0,
    ):
        """Update episode-level metrics."""
        self.training_metrics["episode_rewards"].append(episode_reward)
        self.training_metrics["episode_lengths"].append(episode_length)
        self.training_metrics["success_rates"].append(1.0 if success else 0.0)
        self.training_metrics["exploration_rates"].append(exploration_rate)

    def get_training_metrics(self) -> Dict[str, List]:
        """Get training metrics for analysis."""
        return self.training_metrics

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics of training metrics."""
        summary = {}

        for key, values in self.training_metrics.items():
            if values:
                if key in [
                    "episode_rewards",
                    "policy_losses",
                    "value_losses",
                    "entropy_losses",
                    "total_losses",
                    "advantage_means",
                    "advantage_stds",
                    "action_entropies",
                    "episode_lengths",
                ]:
                    summary[f"{key}_mean"] = np.mean(values)
                    summary[f"{key}_std"] = np.std(values)
                    summary[f"{key}_min"] = np.min(values)
                    summary[f"{key}_max"] = np.max(values)
                elif key in ["success_rates", "exploration_rates"]:
                    summary[f"{key}_mean"] = np.mean(values)
                elif key in ["advantages", "value_predictions"]:
                    if values:
                        summary[f"{key}_mean"] = np.mean(values)
                        summary[f"{key}_std"] = np.std(values)

        return summary

    def save(self, model_dir: str):
        """
        Save the A2C policy to a directory with separate files for policy, value networks, and config.
        
        Args:
            model_dir: Directory to save the model files
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save policy network (actor)
        policy_path = os.path.join(model_dir, "policy.pt")
        torch.save(self.network.actor.state_dict(), policy_path)
        
        # Save value network (critic)
        value_path = os.path.join(model_dir, "value.pt")
        torch.save(self.network.critic.state_dict(), value_path)
        
        # Save shared layers
        shared_path = os.path.join(model_dir, "shared.pt")
        torch.save(self.network.shared.state_dict(), shared_path)
        
        # Save configuration
        config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.lr,
            "device": self.device,
            "skills": self.skills,
            "difficulties": self.difficulties,
            "num_skills": self.num_skills,
            "num_difficulties": self.num_difficulties,
            "normalize_advantages": self.normalize_advantages,
            "use_rms_prop": self.use_rms_prop,
            "rms_prop_eps": self.rms_prop_eps,

        }
        
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"A2C policy saved to {model_dir}")
        logging.info(f"Files saved: policy.pt, value.pt, shared.pt, config.json")

    def load(self, model_dir: str):
        """
        Load weights into the current A2C policy instance.
        
        Args:
            model_dir: Directory containing the saved model files
        """
        # Load network weights
        policy_path = os.path.join(model_dir, "policy.pt")
        value_path = os.path.join(model_dir, "value.pt")
        shared_path = os.path.join(model_dir, "shared.pt")
        
        self.network.actor.load_state_dict(torch.load(policy_path, map_location=self.device))
        self.network.critic.load_state_dict(torch.load(value_path, map_location=self.device))
        self.network.shared.load_state_dict(torch.load(shared_path, map_location=self.device))
        
        # Set to evaluation mode
        self.network.eval()
        
        logging.info(f"A2C policy weights loaded from {model_dir}")