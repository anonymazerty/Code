"""
Utility functions for configuration loading and management.
"""

import configparser
from typing import Dict, List, Any


def validate_agent_reward_combination(agent: str, reward_mode: str) -> bool:
    """
    Validate if the agent and reward_mode combination is supported.
    
    Supported combinations:
    - sarsa + scalarized/reward_machine
    - ppo + scalarized/pareto_buffer_rm
    - a2c + scalarized/pareto_buffer_rm
    
    Args:
        agent: Agent type (sarsa, a2c, ppo)
        reward_mode: Reward mode (none, scalarized, reward_machine, pareto_buffer_rm)
    
    Returns:
        True if combination is supported, False otherwise
    """
    supported_combinations = {
        "sarsa": ["scalarized", "reward_machine"],
        "ppo": ["scalarized", "pareto_buffer_rm"],
        "a2c": ["scalarized", "pareto_buffer_rm"]
    }
    
    if agent not in supported_combinations:
        return False
    
    return reward_mode in supported_combinations[agent]


def load_base_config(agent: str, config_path: str, reward_mode: str, objectives: List[str]) -> Dict[str, Any]:
    """
    Load configuration from config file at config_path based on agent, reward_mode, and objectives.
    
    Args:
        agent: Agent type (sarsa, a2c, ppo)
        config_path: Path to config file
        reward_mode: Reward mode (scalarized, reward_machine, pareto_buffer_rm)
        objectives: List of objectives
    
    Returns:
        Dictionary containing the configuration parameters
    """
    # Load base.ini
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Create configuration key
    # Sort objectives and join with underscore to match config file naming convention
    objectives_str = '_'.join(sorted(objectives))  # Sort for consistent key
    config_key = f"{agent.upper()}.{objectives_str}.{reward_mode}"
    
    # Check if configuration exists
    if config_key not in config:
        # Try to find any configuration for this agent and objectives
        available_configs = [s for s in config.sections() if s.startswith(f"{agent.upper()}.")]
        raise ValueError(f"Configuration '{config_key}' not found in base.ini. "
                       f"Available configurations for {agent}: {available_configs}")
    
    # Load COMMON parameters first (lowest priority)
    config_dict = {}
    if 'COMMON' in config:
        for key, value in config['COMMON'].items():
            if key == 'hidden_dims':
                config_dict[key] = [int(x.strip()) for x in value.split(',')]
            elif key in ['train_episodes', 'train_steps', 'n_epochs', 'mini_batch_size', 'ppo_n_steps', 'a2c_n_steps']:
                # Integer parameters
                config_dict[key] = int(value)
            elif key in ['normalize_advantages', 'use_rms_prop']:
                # Boolean parameters
                config_dict[key] = value.lower() == 'true'
            elif key in ['gamma', 'gae_lambda', 'value_coef', 'entropy_coef', 'max_grad_norm', 'target_kl', 'rms_prop_eps', 'clip_ratio']:
                # Float parameters
                config_dict[key] = float(value)
            else:
                # Try to parse as float first, then as int, then keep as string
                try:
                    config_dict[key] = float(value)
                except ValueError:
                    try:
                            config_dict[key] = int(value)
                    except ValueError:
                        config_dict[key] = value
    
    # Get base section for the agent
    base_section = f"{agent.upper()}_BASE"
    if base_section not in config:
        raise ValueError(f"Base section '{base_section}' not found in base.ini")
    
    # Load agent-specific base parameters (overrides COMMON)
    for key, value in config[base_section].items():
        if key == 'hidden_dims':
            config_dict[key] = [int(x.strip()) for x in value.split(',')]
        elif key in ['normalize_advantages', 'use_rms_prop']:
            config_dict[key] = value.lower() == 'true'
        elif key in ['n_epochs', 'mini_batch_size', 'ppo_n_steps', 'a2c_n_steps']:
            # Integer parameters
            config_dict[key] = int(value)
        elif key in ['gae_lambda', 'value_coef', 'entropy_coef', 'max_grad_norm', 'target_kl', 'rms_prop_eps']:
            # Float parameters
            config_dict[key] = float(value)
        else:
            try:
                config_dict[key] = float(value)
            except ValueError:
                try:
                    config_dict[key] = int(value)
                except ValueError:
                    config_dict[key] = value
    
    # Load specific configuration parameters
    for key, value in config[config_key].items():
        if key == 'learning_rate':
            config_dict[key] = float(value)
        elif key in ['ppo_n_steps', 'a2c_n_steps', 'n_epochs', 'mini_batch_size']:
            # Integer parameters
            config_dict[key] = int(value)
        elif key in ['clip_ratio', 'gae_lambda', 'value_coef', 'entropy_coef', 'max_grad_norm', 'target_kl', 'rms_prop_eps']:
            # Float parameters
            config_dict[key] = float(value)
        else:
            config_dict[key] = value
    
    return config_dict


def apply_base_config(args, base_config: Dict[str, Any]):
    """
    Apply base configuration to args.
    
    Args:
        args: Argument parser object
        base_config: Configuration dictionary from base.ini
    """
    for key, value in base_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            # Dynamically add the attribute if it doesn't exist
            setattr(args, key, value)
            print(f"Added parameter '{key}' = {value} from base.ini to args") 