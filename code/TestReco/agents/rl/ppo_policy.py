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



class PPONetwork(nn.Module):
    """
    PPO network with separate actor and critic heads.
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

    def get_log_prob_and_value(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probability and value for a given state-action pair.

        Args:
            state: Input state tensor
            action: Input action tensor

        Returns:
            Tuple of (log_prob, value)
        """
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action)
        return log_prob, value


class PPOPolicy(BasePolicy):
    """
    PPO (Proximal Policy Optimization) Policy implementation.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4, 
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        mini_batch_size: int = 64,
        n_epochs: int = 10,
        normalize_advantages: bool = True,
        device: str = "cpu",

        target_skill_bundle: List[str] = None,
    ):
        """
        Initialize the PPO policy.

        Args:
            env: Gymnasium environment
            hidden_dims: Dimensions of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_coef: Value function coefficient
            entropy_coef: Entropy coefficient for exploration
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping
            mini_batch_size: Mini-batch size for updates
            n_epochs: Number of epochs per update 
            normalize_advantages: Whether to normalize advantages
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
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device
        self.lr = learning_rate
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.normalize_advantages = normalize_advantages
        self.hidden_dims = hidden_dims



        # Create PPO network
        self.network = PPONetwork(self.state_dim, self.action_dim, hidden_dims, device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

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

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Array of rewards
            values: Array of values
            dones: Array of done flags
            next_value: Value of the next state

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = next_value

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = 0  # Reset advantage when episode ends
            else:
                delta = rewards[t] + self.gamma * last_value - values[t]
                last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        old_log_probs: np.ndarray = None,
        **kwargs,
    ) -> Dict:
        """
        Update the policy using PPO algorithm.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            old_log_probs: Batch of old log probabilities
            **kwargs: Additional arguments

        Returns:
            Dictionary containing loss information
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current values for GAE computation
        with torch.no_grad():
            _, values = self.network(states)
            values = values.squeeze(-1).cpu().numpy()
            _, next_value = self.network(next_states[-1:])
            next_value = next_value.item()

        # Compute GAE
        advantages, returns = self.compute_gae(
            rewards.cpu().numpy(), values, dones.cpu().numpy(), next_value
        )

        # Convert back to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        elif self.normalize_advantages and advantages.numel() == 1:
            advantages = advantages - advantages.mean()  # Just center if only one element

        # PPO update
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        current_log_probs = None
        
        # Mini-batch processing
        batch_size = states.shape[0]
        mini_batch_size = min(self.mini_batch_size, batch_size)
        num_mini_batches = max(1, batch_size // mini_batch_size)
        
        for epoch in range(self.n_epochs):  # Multiple epochs per update
            # Shuffle data for each epoch
            indices = torch.randperm(batch_size)
            shuffled_states = states[indices]
            shuffled_actions = actions[indices]
            shuffled_advantages = advantages[indices]
            shuffled_returns = returns[indices]
            
            epoch_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy_loss = 0
            
            # Process mini-batches
            for mb_idx in range(num_mini_batches):
                start_idx = mb_idx * mini_batch_size
                end_idx = min((mb_idx + 1) * mini_batch_size, batch_size)
                
                mb_states = shuffled_states[start_idx:end_idx]
                mb_actions = shuffled_actions[start_idx:end_idx]
                mb_advantages = shuffled_advantages[start_idx:end_idx]
                mb_returns = shuffled_returns[start_idx:end_idx]
                
                # Get current log probs and values for mini-batch
                log_probs, values = self.network.get_log_prob_and_value(mb_states, mb_actions)
                values = values.squeeze(-1)

                # Compute ratio
                if epoch == 0:
                    # First epoch: use the original old_log_probs from data collection
                    mb_old_log_probs = torch.FloatTensor(np.atleast_1d(old_log_probs[indices[start_idx:end_idx]])).to(self.device)
                    ratio = torch.exp(log_probs - mb_old_log_probs)
                else:
                    # Subsequent epochs: use log_probs from previous epoch as old_log_probs
                    if current_log_probs is not None:
                        mb_current_log_probs = current_log_probs[indices[start_idx:end_idx]]
                        ratio = torch.exp(log_probs - mb_current_log_probs)
                    else:
                        ratio = torch.ones_like(log_probs)
                
                # Store current log_probs for next epoch (for full batch)
                if mb_idx == 0:  # Only store once per epoch
                    current_log_probs = torch.zeros(batch_size, device=self.device)
                current_log_probs[indices[start_idx:end_idx]] = log_probs.detach()

                # Compute clipped surrogate loss
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns)

                # Entropy loss (compute once per mini-batch)
                action_logits, _ = self.network(mb_states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                entropy_loss = dist.entropy().mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy_loss += entropy_loss.item()

            # Early stopping if KL divergence is too high
            if epoch > 0 and current_log_probs is not None:
                # Compute KL divergence between consecutive epochs
                with torch.no_grad():
                    action_logits, _ = self.network(states)
                    action_probs = F.softmax(action_logits, dim=-1)
                    dist = torch.distributions.Categorical(action_probs)
                    
                    # Compute KL divergence between current and previous policy
                    # We need to compute the KL divergence properly
                    # Using a simpler approach: check if the policy changed significantly
                    if epoch == 1:  # Store initial policy for comparison
                        initial_log_probs = current_log_probs.clone()
                    else:
                        # Compute mean absolute difference in log probs
                        kl_div = torch.mean(torch.abs(current_log_probs - initial_log_probs))
                        if kl_div > self.target_kl:
                            logging.info(f"Early stopping at epoch {epoch} due to high KL divergence: {kl_div:.4f}")
                            break

            total_loss += epoch_loss / num_mini_batches
            total_policy_loss += epoch_policy_loss / num_mini_batches
            total_value_loss += epoch_value_loss / num_mini_batches
            total_entropy_loss += epoch_entropy_loss / num_mini_batches

        num_epochs = epoch + 1
        return {
            'total_loss': total_loss / num_epochs,
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy_loss': total_entropy_loss / num_epochs
        }

    def save(self, model_dir: str):
        """
        Save the PPO policy to a directory with separate files for policy, value networks, and config.
        
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
            "clip_ratio": self.clip_ratio,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "learning_rate": self.lr,
            "mini_batch_size": self.mini_batch_size,
            "n_epochs": self.n_epochs,
            "normalize_advantages": self.normalize_advantages,
            "device": self.device,
            "skills": self.skills,
            "difficulties": self.difficulties,
            "num_skills": self.num_skills,
            "num_difficulties": self.num_difficulties,

        }
        
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"PPO policy saved to {model_dir}")
        logging.info(f"Files saved: policy.pt, value.pt, shared.pt, config.json")

    def load(self, model_dir: str):
        """
        Load weights into the current policy instance.
        
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
        
        logging.info(f"PPO policy weights loaded from {model_dir}")