"""
Generic RL Training Module

This module provides a unified training interface for different RL algorithms
including PPO, A2C, and SARSA using Gymnasium.
"""

import logging
from typing import List, Tuple
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agents.rl.base_policy import BasePolicy


class RLTrainer:
    """
    Generic RL Training Manager

    Handles training for different RL agents with unified interface.
    """

    def __init__(self, agent: BasePolicy, env: gym.Env, args, reward_handler=None, model_dir=None):
        """
        Initialize the RL trainer.

        Args:
            agent: RL agent to train
            env: Gymnasium environment
            args: Command line arguments
            reward_handler: Optional reward handler for reward processing
        """
        self.agent = agent
        self.env = env
        self.args = args
        self.reward_handler = reward_handler

        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_exploration_rates = []
        self.max_q_values_history = []  # For Q-learning algorithms
        self.max_action_probs_history = []  # For policy-based algorithms
        self.value_predictions_history = []  # For actor-critic algorithms
        
        # PPO-specific metrics
        self.ppo_policy_losses = []
        self.ppo_value_losses = []
        self.ppo_entropy_losses = []

        # A2C-specific metrics
        self.a2c_policy_losses = []
        self.a2c_value_losses = []
        self.a2c_entropy_losses = []

        # Initialize on-policy algorithm parameters and buffers only if reward_handler is provided
        if reward_handler is not None:
            self._init_on_policy_system()
        
        # Reward machine state tracking
        self.rm_state_history = []

        # Model directory for saving all training outputs
        self.model_dir = model_dir
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
    
    def _init_on_policy_system(self):
        """Initialize parameters and buffers for on-policy algorithms (PPO, A2C)."""
        # Initialize parameters for each on-policy algorithm
        for agent_name in ["ppo", "a2c"]:
            if self.args.agent == agent_name:
                if agent_name == "ppo":
                    setattr(self, f"{agent_name}_n_steps", self.args.ppo_n_steps)
                    logging.info(f"PPO standard mode: collecting {self.args.ppo_n_steps} steps before update")
                else:  # A2C
                    setattr(self, f"{agent_name}_n_steps", self.args.a2c_n_steps)
                    logging.info(f"A2C standard mode: collecting {self.args.a2c_n_steps} steps before update")
        
        # Initialize buffers for on-policy algorithms
        buffer_template = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'old_log_probs': [],
            'episode_rewards': [],  # Store episode reward dictionaries for Pareto dominance check
            'step_count': 0  # Track total steps for PPO and A2C
        }
        
        for agent_name in ["ppo", "a2c"]:
            setattr(self, f"{agent_name}_buffer", __import__('copy').deepcopy(buffer_template))
    
    def _is_on_policy_algorithm(self, agent_name):
        """Check if the agent is an on-policy algorithm."""
        return agent_name in ["ppo", "a2c"]
    
    def _get_agent_buffer(self, agent_name):
        """Get the buffer for a specific agent."""
        return getattr(self, f"{agent_name}_buffer")
    
    def _get_agent_n_steps(self, agent_name):
        """Get the n_steps for a specific agent."""
        if agent_name == "ppo":
            return self.ppo_n_steps
        elif agent_name == "a2c":
            return self.a2c_n_steps
        return None
    
    def _add_episode_to_buffer(self, agent_name, episode_data):
        """Add episode data to the agent's buffer."""
        buffer = self._get_agent_buffer(agent_name)
        
        # Calculate episode total reward dictionary for Pareto dominance check
        episode_total_reward_dict = {}
        
        for obj in self.env.objectives:
            episode_total_reward_dict[obj] = sum(step_rewards[obj] for step_rewards in episode_data['reward_dicts'])
        
        buffer['states'].extend(episode_data['states'])
        buffer['actions'].extend(episode_data['actions'])
        buffer['rewards'].extend(episode_data['rewards'])
        buffer['next_states'].extend(episode_data['next_states'])
        buffer['dones'].extend(episode_data['dones'])
        buffer['old_log_probs'].extend(episode_data['old_log_probs'])
        buffer['episode_rewards'].append(episode_total_reward_dict)
        
        # Update step count for PPO and A2C (both use step-based collection)
        if agent_name in ["ppo", "a2c"]:
            buffer['step_count'] += len(episode_data['states'])
    
    def _add_step_to_buffer(self, agent_name, step_data):
        """Add single step data to the agent's buffer (for PPO step-based collection)."""
        buffer = self._get_agent_buffer(agent_name)
        
        buffer['states'].append(step_data['state'])
        buffer['actions'].append(step_data['action'])
        buffer['rewards'].append(step_data['reward'])
        buffer['next_states'].append(step_data['next_state'])
        buffer['dones'].append(step_data['done'])
        buffer['old_log_probs'].append(step_data['old_log_prob'])
        buffer['step_count'] += 1
    
    def _process_on_policy_update(self, agent_name, episode_data):
        """Process update for on-policy algorithms (PPO, A2C)."""
        buffer = self._get_agent_buffer(agent_name)
        
        # Add episode data and check if we have enough steps
        self._add_episode_to_buffer(agent_name, episode_data)
        n_steps = self._get_agent_n_steps(agent_name)
        steps_in_buffer = buffer['step_count']
            
        if steps_in_buffer >= n_steps:
            # For Pareto buffer modes, filter to keep only Pareto optimal episodes
            if self.args.reward_mode in ["pareto_buffer_s", "pareto_buffer_rm"]:
                filtered_buffer = self._filter_pareto_optimal_episodes(buffer)
                if filtered_buffer['states']:  # Only update if we have data after filtering
                    loss_info = self._perform_agent_update(agent_name, filtered_buffer)
                    logging.info(f"{agent_name.upper()} Pareto buffer update completed with {len(filtered_buffer['episode_rewards'])} Pareto optimal episodes, {len(filtered_buffer['states'])} steps")
                else:
                    loss_info = {'total_loss': 0.0}
                    logging.info("No Pareto optimal episodes found after filtering")
            else:
                # For standard modes, use all collected data
                loss_info = self._perform_agent_update(agent_name, buffer)
                logging.info(f"{agent_name.upper()} standard update completed with {steps_in_buffer} steps, {len(buffer['episode_rewards'])} episodes")
                logging.info(f"  Policy Loss: {loss_info['policy_loss']:.4f}")
                logging.info(f"  Value Loss: {loss_info['value_loss']:.4f}")
                logging.info(f"  Entropy Loss: {loss_info['entropy_loss']:.4f}")
            
            # Clear buffer after update
            setattr(self, f"{agent_name}_buffer", {
                'states': [],
                'actions': [],
                'rewards': [],
                'next_states': [],
                'dones': [],
                'old_log_probs': [],
                'episode_rewards': [],
                'step_count': 0
            })
        else:
            loss_info = {'total_loss': 0.0}
        
        return loss_info
    
    def _is_pareto_dominated(self, reward_dict, existing_rewards):
        """
        Check if a reward dictionary is Pareto dominated by any existing reward dictionaries.
        
        Args:
            reward_dict: The reward dictionary to check
            existing_rewards: List of existing reward dictionaries
            
        Returns:
            True if the reward dictionary is dominated, False otherwise
        """
        if not existing_rewards:
            return False
            
        # Convert dictionaries to arrays for comparison
        reward_array = np.array([reward_dict[obj] for obj in self.env.objectives])
        for existing_reward in existing_rewards:
            existing_array = np.array([existing_reward[obj] for obj in self.env.objectives])
            # Check if existing_reward dominates reward_dict
            # A dominates B if A is better or equal in all objectives and strictly better in at least one
            if np.all(existing_array >= reward_array) and np.any(existing_array > reward_array):
                return True
        return False
    
    def _filter_pareto_optimal_episodes(self, episode_data):
        """
        Filter episodes to keep only Pareto optimal ones.
        
        Args:
            episode_data: Dictionary containing episode data with 'episode_rewards' key
            
        Returns:
            Filtered episode data with only Pareto optimal episodes
        """
        if not episode_data['episode_rewards']:
            return episode_data
            
        # Find Pareto optimal episodes
        pareto_optimal_indices = []
        for i, reward_dict in enumerate(episode_data['episode_rewards']):
            if not self._is_pareto_dominated(reward_dict, episode_data['episode_rewards']):
                pareto_optimal_indices.append(i)
        
        # Filter data to keep only Pareto optimal episodes
        filtered_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'old_log_probs': [],
            'episode_rewards': []
        }
        
        # Calculate episode boundaries based on episode_rewards length
        # Each episode_reward corresponds to one episode
        episode_boundaries = []
        current_pos = 0
        for i in range(len(episode_data['episode_rewards'])):
            # Find the number of steps in this episode
            # We need to count steps until we reach the next episode
            episode_steps = 0
            for j in range(current_pos, len(episode_data['states'])):
                episode_steps += 1
                # Check if we've reached the end of this episode
                if j < len(episode_data['dones']) and episode_data['dones'][j]:
                    break
            episode_boundaries.append((current_pos, current_pos + episode_steps))
            current_pos += episode_steps
        
        # Keep only Pareto optimal episodes
        for idx in pareto_optimal_indices:
            start_idx, end_idx = episode_boundaries[idx]
            filtered_data['states'].extend(episode_data['states'][start_idx:end_idx])
            filtered_data['actions'].extend(episode_data['actions'][start_idx:end_idx])
            filtered_data['rewards'].extend(episode_data['rewards'][start_idx:end_idx])
            filtered_data['next_states'].extend(episode_data['next_states'][start_idx:end_idx])
            filtered_data['dones'].extend(episode_data['dones'][start_idx:end_idx])
            filtered_data['old_log_probs'].extend(episode_data['old_log_probs'][start_idx:end_idx])
            filtered_data['episode_rewards'].append(episode_data['episode_rewards'][idx])
        
        logging.info(f"Filtered {len(episode_data['episode_rewards'])} episodes to {len(filtered_data['episode_rewards'])} Pareto optimal episodes")
        return filtered_data
    
    def _perform_agent_update(self, agent_name, buffer):
        """Perform the actual agent update with buffer data."""
        states = np.array(buffer['states'])
        actions = np.array(buffer['actions'])
        rewards = np.array(buffer['rewards'])
        next_states = np.array(buffer['next_states'])
        dones = np.array(buffer['dones'])
        old_log_probs = np.array(buffer['old_log_probs'])
        
        loss_info = self.agent.update(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            old_log_probs=old_log_probs,
        )
        
        # Record algorithm-specific losses
        self._record_agent_losses(agent_name, loss_info)
        
        return loss_info
    
    def _record_agent_losses(self, agent_name, loss_info):
        """Record losses for specific agent types."""
        if agent_name == "ppo":
            self.ppo_policy_losses.append(loss_info['policy_loss'])
            self.ppo_value_losses.append(loss_info['value_loss'])
            self.ppo_entropy_losses.append(loss_info['entropy_loss'])
        elif agent_name == "a2c":
            self.a2c_policy_losses.append(loss_info['policy_loss'])
            self.a2c_value_losses.append(loss_info['value_loss'])
            self.a2c_entropy_losses.append(loss_info['entropy_loss'])

    def train(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Train the RL agent.

        Returns:
            Tuple of (episode_rewards, episode_losses, episode_exploration_rates, max_q_values_history)
        """
        logging.info(f"********* Starting Training *********")
        logging.info(
            f"Training {self.args.agent} agent for {self.args.train_episodes} episodes with max {self.args.train_steps} steps per episode..."
        )

        for episode in range(self.args.train_episodes):
            self._train_episode(episode)

            # Log progress
            if (episode + 1) % 100 == 0:
                self._log_training_progress(episode + 1)

        # Save training results
        self._save_training_results()

        return (
            self.episode_rewards,
            self.episode_losses,
            self.episode_exploration_rates,
            self.max_q_values_history,
        )

    def _update_agent(self, state, action, reward, next_state, done, old_log_prob=None):
        """Update the agent based on its type."""
        if hasattr(self.agent, "store_experience"):
            # For algorithms with experience replay
            self.agent.store_experience(state, action, reward, next_state, done)
            return self.agent.update()
        else:
            # For SARSA and other on-policy algorithms
            update_kwargs = {
                "states": np.array([state]),
                "actions": np.array([action]),
                "rewards": np.array([reward]),
                "next_states": np.array([next_state]),
                "dones": np.array([done]),
            }
            
            # Add old_log_probs for PPO if available
            if old_log_prob is not None:
                update_kwargs["old_log_probs"] = np.array([old_log_prob])
            
            return self.agent.update(**update_kwargs)

    def _log_training_progress(self, episode: int, episode_log_interval: int = 100):
        """Log training progress every 100 episodes."""
        start_idx = max(0, len(self.episode_rewards) - episode_log_interval)

        avg_reward = np.mean(self.episode_rewards[start_idx:], axis=0)
        avg_loss = np.mean(self.episode_losses[start_idx:])
        avg_exploration = np.mean(self.episode_exploration_rates[start_idx:])

        # Algorithm-specific metrics
        if self.max_q_values_history and any(self.max_q_values_history):
            avg_max_q = np.mean(self.max_q_values_history[start_idx:])
            q_metric_info = (
                f"Average Max Q Value (last {episode_log_interval}): {avg_max_q:.4f}"
            )
        elif self.max_action_probs_history and any(self.max_action_probs_history):
            avg_max_prob = np.mean(self.max_action_probs_history[start_idx:])
            q_metric_info = f"Average Action Confidence (last {episode_log_interval}): {avg_max_prob:.4f}"
        elif self.value_predictions_history and any(self.value_predictions_history):
            avg_value = np.mean(self.value_predictions_history[start_idx:])
            q_metric_info = f"Average Value Prediction (last {episode_log_interval}): {avg_value:.4f}"
        else:
            q_metric_info = "No algorithm-specific metrics available"

        # Format last and average reward for logging
        def _format_reward(r):
            if isinstance(r, (np.ndarray, list)) and not np.isscalar(r):
                return r.tolist()
            return float(r)

        logging.info(
            f"Episode {episode}/{self.args.train_episodes}\n"
            f"  Last Episode Reward: {_format_reward(self.episode_rewards[-1])}\n"
            f"  Average Reward (last {episode_log_interval}): {_format_reward(avg_reward)}\n"
            f"  Steps: {len(self.episode_rewards)}\n"
            f"  Average Loss (last {episode_log_interval}): {avg_loss:.4f}\n"
            f"  Exploration Rate: {self.episode_exploration_rates[-1]:.2%}\n"
            f"  Average Exploration (last {episode_log_interval}): {avg_exploration:.2%}\n"
            f"  Epsilon: {getattr(self.agent, 'epsilon', 'N/A')}\n"
            f"  {q_metric_info}"
        )

    def _save_training_results(self):
        """Save training results and generate plots."""
        
        # Save the trained policy
        self.agent.save(self.model_dir)
        logging.info(f"Saved trained policy to {self.model_dir}")

        # Save algorithm-specific metrics to model directory
        if self.max_q_values_history and any(self.max_q_values_history):
            # Q-learning algorithms
            q_values_file = os.path.join(self.model_dir, "q_values_history.npy")
            np.save(q_values_file, np.array(self.max_q_values_history))
            logging.info(f"Saved Q-values history to {q_values_file}")
            self._plot_q_values_trend()
        elif self.max_action_probs_history and any(self.max_action_probs_history):
            # Policy-based algorithms
            action_probs_file = os.path.join(self.model_dir, "action_probs_history.npy")
            np.save(action_probs_file, np.array(self.max_action_probs_history))
            logging.info(f"Saved action probabilities history to {action_probs_file}")
            self._plot_action_probs_trend()
        elif self.value_predictions_history and any(self.value_predictions_history):
            # Actor-critic algorithms
            values_file = os.path.join(self.model_dir, "value_predictions_history.npy")
            np.save(values_file, np.array(self.value_predictions_history))
            logging.info(f"Saved value predictions history to {values_file}")
            self._plot_value_predictions_trend()
        
        # PPO-specific loss plotting
        if self.args.agent == "ppo" and self.ppo_policy_losses:
            self._plot_actor_critic_losses_trend(self.args.agent)
        
        # A2C-specific loss plotting
        if self.args.agent == "a2c" and self.a2c_policy_losses:
            self._plot_actor_critic_losses_trend(self.args.agent)

    def _plot_q_values_trend(self):
        """Plot the trend of max Q(s,a) values during training."""
        plt.figure(figsize=(12, 6))

        # Plot max Q values over episodes
        episodes = range(1, len(self.max_q_values_history) + 1)
        plt.plot(
            episodes,
            self.max_q_values_history,
            "b-",
            linewidth=1,
            alpha=0.7,
            label="Max Q(s,a)",
        )

        # Add moving average for smoother trend
        if len(self.max_q_values_history) > 10:
            window_size = min(50, len(self.max_q_values_history) // 10)
            moving_avg = np.convolve(
                self.max_q_values_history,
                np.ones(window_size) / window_size,
                mode="valid",
            )
            moving_avg_episodes = range(window_size, len(self.max_q_values_history) + 1)
            plt.plot(
                moving_avg_episodes,
                moving_avg,
                "r-",
                linewidth=2,
                label=f"Moving Average (window={window_size})",
            )

        plt.xlabel("Training Episodes")
        plt.ylabel("Max Q(s,a) Value")
        plt.title(
            f"{self.args.agent.upper()} Training: Max Q(s,a) Trend\n{self.args.agent} {self.args.response_model} {self.args.data_type} {self.args.benchmark}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        q_values_plot_file = os.path.join(self.model_dir, "q_values_trend.png")
        plt.savefig(q_values_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Saved Q-values trend plot to {q_values_plot_file}")

    def _plot_action_probs_trend(self):
        """Plot the trend of action confidence during training."""
        plt.figure(figsize=(12, 6))

        # Plot action confidence over episodes
        episodes = range(1, len(self.max_action_probs_history) + 1)
        plt.plot(
            episodes,
            self.max_action_probs_history,
            "g-",
            linewidth=1,
            alpha=0.7,
            label="Action Confidence",
        )

        # Add moving average for smoother trend
        if len(self.max_action_probs_history) > 10:
            window_size = min(50, len(self.max_action_probs_history) // 10)
            moving_avg = np.convolve(
                self.max_action_probs_history,
                np.ones(window_size) / window_size,
                mode="valid",
            )
            moving_avg_episodes = range(
                window_size, len(self.max_action_probs_history) + 1
            )
            plt.plot(
                moving_avg_episodes,
                moving_avg,
                "r-",
                linewidth=2,
                label=f"Moving Average (window={window_size})",
            )

        plt.xlabel("Training Episodes")
        plt.ylabel("Action Confidence")
        plt.title(
            f"{self.args.agent.upper()} Training: Action Confidence Trend\n{self.args.agent} {self.args.response_model} {self.args.data_type} {self.args.benchmark}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        action_probs_plot_file = os.path.join(self.model_dir, "action_probs_trend.png")
        plt.savefig(action_probs_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Saved action confidence trend plot to {action_probs_plot_file}")

    def _plot_value_predictions_trend(self):
        """Plot the trend of value predictions during training."""
        plt.figure(figsize=(12, 6))

        # Plot value predictions over episodes
        episodes = range(1, len(self.value_predictions_history) + 1)
        plt.plot(
            episodes,
            self.value_predictions_history,
            "b-",
            linewidth=1,
            alpha=0.7,
            label="Value Predictions",
        )

        # Add moving average for smoother trend
        if len(self.value_predictions_history) > 10:
            window_size = min(50, len(self.value_predictions_history) // 10)
            moving_avg = np.convolve(
                self.value_predictions_history,
                np.ones(window_size) / window_size,
                mode="valid",
            )
            moving_avg_episodes = range(
                window_size, len(self.value_predictions_history) + 1
            )
            plt.plot(
                moving_avg_episodes,
                moving_avg,
                "r-",
                linewidth=2,
                label=f"Moving Average (window={window_size})",
            )

        plt.xlabel("Training Episodes")
        plt.ylabel("Value Prediction")
        plt.title(
            f"{self.args.agent.upper()} Training: Value Predictions Trend\n{self.args.agent} {self.args.response_model} {self.args.data_type} {self.args.benchmark}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        values_plot_file = os.path.join(self.model_dir, "value_predictions_trend.png")
        plt.savefig(values_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Saved value predictions trend plot to {values_plot_file}")

    def _plot_actor_critic_losses_trend(self, agent_name):
        """Plot the trend of actor-critic losses (PPO/A2C) during training."""

        # Get the appropriate loss lists based on agent type
        if agent_name == "ppo":
            policy_losses = self.ppo_policy_losses
            value_losses = self.ppo_value_losses
            entropy_losses = self.ppo_entropy_losses
        elif agent_name == "a2c":
            policy_losses = self.a2c_policy_losses
            value_losses = self.a2c_value_losses
            entropy_losses = self.a2c_entropy_losses
        else:
            return  # Not an actor-critic algorithm

        if not policy_losses:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        episodes = range(1, len(policy_losses) + 1)

        def _plot_with_moving_avg(ax, y, label, color="b-"):
            ax.plot(episodes, y, color, linewidth=1, alpha=0.7, label=label)
            if len(y) > 10:
                window_size = min(20, max(2, len(y) // 5))
                moving_avg = np.convolve(
                    y, np.ones(window_size) / window_size, mode="valid"
                )
                moving_avg_episodes = range(window_size, len(y) + 1)
                ax.plot(
                    moving_avg_episodes,
                    moving_avg,
                    "r-",
                    linewidth=2,
                    label=f"Moving Avg (w={window_size})",
                )
            ax.legend()
            ax.grid(True, alpha=0.3)

        _plot_with_moving_avg(ax1, policy_losses, "Policy Loss", "b-")
        ax1.set_title(f"{agent_name.upper()} {self.args.reward_mode} Policy Loss")
        ax1.set_xlabel("Updates")
        ax1.set_ylabel("Policy Loss")

        _plot_with_moving_avg(ax2, value_losses, "Value Loss", "g-")
        ax2.set_title(f"{agent_name.upper()} {self.args.reward_mode} Value Loss")
        ax2.set_xlabel("Updates")
        ax2.set_ylabel("Value Loss")

        _plot_with_moving_avg(ax3, entropy_losses, "Entropy Loss", "m-")
        ax3.set_title(f"{agent_name.upper()} {self.args.reward_mode} Entropy Loss")
        ax3.set_xlabel("Updates")
        ax3.set_ylabel("Entropy Loss")

        total_losses = [p + v - e for p, v, e in zip(policy_losses, value_losses, entropy_losses)]
        _plot_with_moving_avg(ax4, total_losses, "Total Loss", "c-")
        ax4.set_title(f"{agent_name.upper()} {self.args.reward_mode} Total Loss")
        ax4.set_xlabel("Updates")
        ax4.set_ylabel("Total Loss")

        plt.tight_layout()

        losses_plot_file = os.path.join(self.model_dir, f"{agent_name}_losses_trend.png")
        plt.savefig(losses_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(
            f"Saved {agent_name.upper()} {self.args.reward_mode} losses trend plot to {losses_plot_file}"
        )


class RewardHandlerRLTrainer(RLTrainer):
    """RL trainer with reward machine integration."""
    
    def __init__(self, agent, env, args, reward_handler, model_dir=None):
        super().__init__(agent, env, args, reward_handler, model_dir)
        
        # Reward machine state tracking
        self.rm_state_history = []
    
    def _update_algorithm_metrics(self, action_info, episode_max_q_values, episode_max_action_probs, episode_value_predictions):
        """
        Intelligently update algorithm-specific metrics based on agent type.
        
        Args:
            action_info: Action information from agent
            episode_max_q_values: List to store max Q values (for SARSA)
            episode_max_action_probs: List to store action probabilities (for A2C, SARSA, PPO)
            episode_value_predictions: List to store value predictions (for A2C, PPO)
        """
        if action_info.get("exploration", False):
            return  # Skip metrics for exploration actions
        
        # Update max Q values (for SARSA)
        if self.args.agent == "sarsa" and "q_values" in action_info:
            max_q_value = np.max(action_info["q_values"])
            episode_max_q_values.append(max_q_value)
        
        # Update action probabilities (for A2C, SARSA, PPO)
        if self.args.agent in ["a2c", "sarsa", "ppo"] and "log_prob" in action_info:
            confidence = np.exp(action_info["log_prob"])
            episode_max_action_probs.append(confidence)
        
        # Update value predictions (for A2C, PPO)
        if self.args.agent in ["a2c", "ppo"] and "value" in action_info:
            episode_value_predictions.append(action_info["value"])
    
    def _record_episode_metrics(self, episode_max_q_values, episode_max_action_probs, episode_value_predictions):
        """
        Record episode-level metrics based on algorithm type.
        
        Args:
            episode_max_q_values: List of max Q values for this episode
            episode_max_action_probs: List of action probabilities for this episode
            episode_value_predictions: List of value predictions for this episode
        """
        # Record max Q values (for SARSA)
        if self.args.agent == "sarsa":
            if episode_max_q_values:
                avg_max_q = np.mean(episode_max_q_values)
                self.max_q_values_history.append(avg_max_q)
            else:
                self.max_q_values_history.append(0.0)
        else:
            self.max_q_values_history.append(0.0)
        
        # Record action probabilities (for A2C, SARSA, PPO)
        if self.args.agent in ["a2c", "sarsa", "ppo"]:
            if episode_max_action_probs:
                avg_max_prob = np.mean(episode_max_action_probs)
                self.max_action_probs_history.append(avg_max_prob)
            else:
                self.max_action_probs_history.append(0.0)
        else:
            self.max_action_probs_history.append(0.0)
        
        # Record value predictions (for A2C, PPO)
        if self.args.agent in ["a2c", "ppo"]:
            if episode_value_predictions:
                avg_value = np.mean(episode_value_predictions)
                self.value_predictions_history.append(avg_value)
            else:
                self.value_predictions_history.append(0.0)
        else:
            self.value_predictions_history.append(0.0)
    
    def _train_episode(self, episode: int):
        # Reset reward machine state if needed
        if self.args.reward_mode == "reward_machine":
            if hasattr(self.reward_handler, "reward_machine"):
                self.reward_handler.reward_machine.reset()
        elif self.args.reward_mode == "pareto_buffer_rm":
            if hasattr(self.reward_handler.base_handler, "reward_machine"):
                self.reward_handler.base_handler.reward_machine.reset()

        state, info = self.env.reset()
        truncated = False
        total_reward = 0
        step_count = 0
        episode_loss = 0
        exploration_count = 0
        episode_max_q_values = []
        episode_max_action_probs = []
        episode_value_predictions = []

        # For on-policy algorithms: collect episode data for batch update
        if self._is_on_policy_algorithm(self.args.agent):
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_next_states = []
            episode_dones = []
            episode_old_log_probs = []
            episode_reward_dicts = []  # Store reward dictionaries for Pareto dominance check

        while not truncated and step_count < self.args.train_steps:
            logging.info("="*40)
            logging.info(f"Step {step_count+1} of episode {episode}")
            action_info = self.agent.select_action(state, env=self.env)
            
            action = action_info["action"]
            logging.info(f"action_info: {action_info}")
            if action_info.get("exploration", False):
                exploration_count += 1
            self._update_algorithm_metrics(action_info, episode_max_q_values, episode_max_action_probs, episode_value_predictions)
            next_state, reward_dict, truncated, info = self.env.step(action)

            # Compose info for reward machine
            next_state_info = {
                "failed_questions_ratio": info["failed_questions_ratio"],
                "rolling_accuracy": info["rolling_accuracy"],
                "avg_mastery": np.mean(list(info["mastery"].values())),
                "valid_failed_questions": info["valid_failed_questions"]
            }

            # Use reward handler
            if self.args.reward_mode == "reward_machine":
                scalar_reward = self.reward_handler.process_reward(
                    reward_dict=reward_dict,
                    next_state_info=next_state_info,
                )
                logging.info(
                    f"reward state after: {self.reward_handler.reward_machine.state}"
                )
            elif self.args.reward_mode == "pareto_buffer_rm":
                scalar_reward = self.reward_handler.process_reward(
                    reward_dict=reward_dict,
                    next_state_info=next_state_info,
                )
                logging.info(
                    f"reward state after: {self.reward_handler.base_handler.reward_machine.state}"
                )
            else:
                scalar_reward = self.reward_handler.process_reward(reward_dict = reward_dict)

            logging.info(f"Selected Questions: {info['selected_questions']}")
            logging.info(f"Questions Info: {info['questions_info']}")
            logging.info(f"Rolling Accuracy: {info['rolling_accuracy']}")
            logging.info(f"Masteries: {info['mastery']}")
            logging.info(f"Rewards: {reward_dict}")
            logging.info(f"reward scaler: {scalar_reward}")
    
            # For on-policy algorithms (PPO, A2C): collect episode data
            if self._is_on_policy_algorithm(self.args.agent):
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(scalar_reward)
                episode_next_states.append(next_state)
                episode_dones.append(truncated)
                episode_old_log_probs.append(action_info.get("log_prob", 0.0))
                episode_reward_dicts.append(reward_dict)  # Store reward dictionary for Pareto check
            else:
                # For other algorithms: update immediately
                old_log_prob = action_info.get("log_prob", None)
                loss = self._update_agent(
                    state, action, scalar_reward, next_state, truncated, old_log_prob
                )
                episode_loss += loss

            state = next_state
            total_reward += scalar_reward
            step_count += 1

        # For on-policy algorithms: process episode data
        if self._is_on_policy_algorithm(self.args.agent) and episode_states:
            episode_data = {
                'states': episode_states,
                'actions': episode_actions,
                'rewards': episode_rewards,
                'next_states': episode_next_states,
                'dones': episode_dones,
                'old_log_probs': episode_old_log_probs,
                'reward_dicts': episode_reward_dicts
            }
            loss_info = self._process_on_policy_update(self.args.agent, episode_data)
            episode_loss = loss_info.get('total_loss', 0.0)

        logging.info(f"*** Episode {episode} complete with scalar rewards: {total_reward} ***")
        logging.info(f"=" * 40)

        self.episode_rewards.append(total_reward)
        self.episode_losses.append(episode_loss / step_count if step_count > 0 else 0)
        exploration_rate = exploration_count / step_count if step_count > 0 else 0
        self.episode_exploration_rates.append(exploration_rate)
        self._record_episode_metrics(episode_max_q_values, episode_max_action_probs, episode_value_predictions)
            
        # Record reward machine state for this episode (if applicable)
        if self.args.reward_mode == "reward_machine" and hasattr(
            self.reward_handler.reward_machine, "state"
        ):
            self.rm_state_history.append(self.reward_handler.reward_machine.state)
        elif self.args.reward_mode == "pareto_buffer_rm" and hasattr(
            self.reward_handler.base_handler.reward_machine, "state"
        ):
            self.rm_state_history.append(
                self.reward_handler.base_handler.reward_machine.state
            )
        return step_count

    def train(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        logging.info(f"********* Starting Training *********")
        self.episode_steps = []  # Track steps per episode
        for episode in range(self.args.train_episodes):
            steps = self._train_episode(episode)
            if steps is not None:
                self.episode_steps.append(steps)
            else:
                self.episode_steps.append(self.args.train_steps)
            if (episode + 1) % 10 == 0:
                self._log_training_progress(episode + 1)
        
        # Handle remaining on-policy buffer data at the end of training
        if self._is_on_policy_algorithm(self.args.agent):
            buffer = self._get_agent_buffer(self.args.agent)
            if buffer['states']:
                logging.info(f"Processing remaining {self.args.agent.upper()} buffer data...")
                
                if self.args.agent == "ppo":
                    # PPO: Report steps collected
                    steps_collected = buffer['step_count']
                    episodes_collected = len(buffer['episode_rewards'])
                    logging.info(f"PPO collected {steps_collected} steps across {episodes_collected} episodes")
                else:
                    # A2C: Report episodes collected
                    episodes_collected = len(buffer['episode_rewards'])
                    logging.info(f"A2C collected {episodes_collected} episodes")
                
                # For Pareto buffer modes, filter to keep only Pareto optimal episodes
                if self.args.reward_mode in ["pareto_buffer_s", "pareto_buffer_rm"]:
                    filtered_buffer = self._filter_pareto_optimal_episodes(buffer)
                    if filtered_buffer['states']:  # Only update if we have data after filtering
                        loss_info = self._perform_agent_update(self.args.agent, filtered_buffer)
                        logging.info(f"Final {self.args.agent.upper()} Pareto buffer update completed with {len(filtered_buffer['episode_rewards'])} Pareto optimal episodes, {len(filtered_buffer['states'])} steps, loss: {loss_info}")
                    else:
                        logging.info("No Pareto optimal episodes found in final buffer")
                else:
                    # For standard modes, use all remaining data
                    loss_info = self._perform_agent_update(self.args.agent, buffer)
                    if self.args.agent == "ppo":
                        logging.info(f"Final PPO standard update completed with {buffer['step_count']} steps, {len(buffer['episode_rewards'])} episodes, loss: {loss_info}")
                    else:
                        logging.info(f"Final A2C standard update completed with {len(buffer['episode_rewards'])} episodes, {len(buffer['states'])} steps, loss: {loss_info}")
        
        self._save_training_results()
        # After training, plot reward machine state transitions if applicable
        if self.args.reward_mode == "reward_machine" and self.rm_state_history:
            self._plot_rm_state_timeline()
        elif self.args.reward_mode == "pareto_buffer_rm" and self.rm_state_history:
            self._plot_rm_state_timeline()
        # Plot steps per episode and scalar reward per episode
        self._plot_steps_per_episode()
        self._plot_scalar_reward_per_episode()
        return (
            self.episode_rewards,
            self.episode_losses,
            self.episode_exploration_rates,
            self.max_q_values_history,
        )

    def _plot_rm_state_timeline(self):
        """Plot the reward machine state transitions over episodes."""
        state_names = list(sorted(set(self.rm_state_history)))
        state_to_idx = {s: i for i, s in enumerate(state_names)}
        y = [state_to_idx[s] for s in self.rm_state_history]
        x = list(range(1, len(self.rm_state_history) + 1))
        plt.figure(figsize=(12, 4))
        plt.step(x, y, where="post", label="Reward Machine State")
        plt.yticks(list(state_to_idx.values()), state_names)
        plt.xlabel("Episode")
        plt.ylabel("Reward Machine State")
        plt.title("Reward Machine State Transitions During Training")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        output_file = os.path.join(self.model_dir, "reward_machine_state_timeline.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved reward machine state timeline to {output_file}")

    def _plot_steps_per_episode(self):
        plt.figure(figsize=(10, 5))
        plt.scatter(range(1, len(self.episode_steps) + 1), self.episode_steps, s=8)
        plt.xlabel("Episode")
        plt.ylabel("Total Steps")
        plt.title("Total Number of Steps per Episode")
        plt.grid(True, alpha=0.3)
        output_file = os.path.join(self.model_dir, "steps_per_episode.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved steps per episode plot to {output_file}")

    def _plot_scalar_reward_per_episode(self):
        plt.figure(figsize=(10, 5))
        plt.scatter(range(1, len(self.episode_rewards) + 1), self.episode_rewards, s=8)
        plt.xlabel("Episode")
        plt.ylabel("Scalar Reward")
        plt.title("Scalar Reward per Episode")
        plt.grid(True, alpha=0.3)
        output_file = os.path.join(self.model_dir, "scalar_reward_per_episode.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved scalar reward per episode plot to {output_file}")

