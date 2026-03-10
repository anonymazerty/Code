"""
Policy tools for orchestrator to call different policies.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from agents.llm_agents.simple_agent import SimpleLLMAgent
from agents.rl.ppo_policy import PPOPolicy
from agents.rl.sarsa_policy import SARSAPolicy
from agents.rl.a2c_policy import A2CPolicy

# Policy parameter mappings - defines which config parameters each policy accepts
POLICY_PARAM_MAPPINGS = {
    "ppo": {
        'hidden_dims', 'learning_rate', 'gamma',
        'gae_lambda', 'clip_ratio', 'value_coef', 'entropy_coef',
        'max_grad_norm', 'target_kl', 'mini_batch_size', 'n_epochs', 'normalize_advantages',
        'target_skill_bundle'
    },
    "sarsa": {
        'hidden_dims', 'learning_rate', 'gamma', 
        'epsilon', 'epsilon_decay', 'epsilon_min', 
        'target_skill_bundle', 'seed'
    },
    "a2c": {
        'hidden_dims', 'learning_rate', 'gamma',
        'gae_lambda', 'entropy_coef', 'value_coef', 'max_grad_norm', 
        'normalize_advantages', 'use_rms_prop', 'rms_prop_eps',
        'target_skill_bundle'
    },
}


class PolicyTool:
    """
    Wrapper class to load and call different policies with a unified interface.
    """
    
    def __init__(
        self,
        env: gym.Env,
        policy_type: str,
        policy_config: Dict[str, Any],
        model_dir: Optional[str] = None,
    ):
        """
        Initialize a policy tool.
        
        Args:
            policy_type: Type of policy ("ppo", "sarsa", "a2c", "llm")
            policy_config: Configuration dictionary for the policy
            model_path: Path to saved model (for RL policies)
        """
        self.env = env
        self.policy_type = policy_type
        self.policy_config = policy_config
        self.model_dir = model_dir
        self.policy = None
        
        self._load_policy()
    
    def _load_policy(self):
        """Load the specified policy."""
        if self.policy_type == "ppo":
            # Filter config for PPO policy
            ppo_config = self._filter_ppo_config(self.policy_config)
            self.policy = PPOPolicy(env=self.env, **ppo_config)
            if self.model_dir:
                self.policy.load(self.model_dir)
                
        elif self.policy_type == "sarsa":
            # Filter config for SARSA policy
            sarsa_config = self._filter_sarsa_config(self.policy_config)
            self.policy = SARSAPolicy(env=self.env, **sarsa_config)
            if self.model_dir:
                self.policy.load(self.model_dir)
                
        elif self.policy_type == "a2c":
            # Filter config for A2C policy
            a2c_config = self._filter_a2c_config(self.policy_config)
            self.policy = A2CPolicy(env=self.env, **a2c_config)
            if self.model_dir:
                self.policy.load(self.model_dir)
                
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def _filter_config(self, policy_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter configuration for a specific policy type."""
        if policy_type not in POLICY_PARAM_MAPPINGS:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        valid_params = POLICY_PARAM_MAPPINGS[policy_type]
        filtered_config = {k: v for k, v in config.items() if k in valid_params}
        self._log_filtered_params(policy_type.upper(), config, filtered_config, valid_params)
        return filtered_config
    
    def _filter_ppo_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter configuration for PPO policy."""
        return self._filter_config("ppo", config)
    
    def _filter_sarsa_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter configuration for SARSA policy."""
        return self._filter_config("sarsa", config)
    
    def _filter_a2c_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter configuration for A2C policy."""
        return self._filter_config("a2c", config)
    

    
    def _log_filtered_params(self, policy_type: str, original_config: Dict[str, Any], 
                            filtered_config: Dict[str, Any], valid_params: set):
        """Log which parameters were filtered out."""
        original_keys = set(original_config.keys())
        filtered_keys = set(filtered_config.keys())
        removed_keys = original_keys - filtered_keys
        
        if removed_keys:
            logging.info(f"{policy_type} policy: Filtered out {len(removed_keys)} parameters: {list(removed_keys)}")
        
        missing_keys = valid_params - filtered_keys
        if missing_keys:
            logging.warning(f"{policy_type} policy: Missing {len(missing_keys)} parameters: {list(missing_keys)}")
        
        logging.info(f"{policy_type} policy: Using {len(filtered_config)} parameters: {list(filtered_config.keys())}")
    
    def __call__(self, state: np.ndarray, **kwargs) -> Tuple[int, Dict[str, Any]]:
        """
        Call the policy with the given state.
        
        Args:
            state: Current state
            env: Environment instance
            **kwargs: Additional arguments
            
        Returns:
            action_info: Dict of action information
        """
        if self.policy_type in ["ppo", "sarsa", "a2c"]:
            # RL policies use select_action method
            action_info = self.policy.select_action(state, env=self.env, **kwargs)
            return action_info
        
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def get_policy_info(self) -> Dict[str, Any]:
        """Get information about the policy."""
        return {
            "policy_type": self.policy_type,
            "config": self.policy_config,
            "model_dir": self.model_dir
        }


class PolicyFactory:
    """
    Factory class to create policy tools.
    """
    
    @staticmethod
    def create_policy(
        env: gym.Env,
        policy_type: str,
        policy_config: Dict[str, Any],
        model_dir: Optional[str] = None,
    ) -> PolicyTool:
        """
        Create a policy tool.
        
        Args:
            policy_type: Type of policy ("ppo", "sarsa", "a2c")
            policy_config: Configuration dictionary for the policy
            model_path: Path to saved model (for RL policies)
            
        Returns:
            PolicyTool instance
        """
        return PolicyTool(env, policy_type, policy_config, model_dir)
    
    @staticmethod
    def get_available_policies() -> List[str]:
        """Get list of available policy types."""
        return ["ppo", "sarsa", "a2c"] 