"""
Utility functions for agent creation and management.
"""

from typing import List, Optional, Union

import torch

from agents.llm_agents.simple_agent import SimpleLLMAgent
from agents.rl.a2c_policy import A2CPolicy
from agents.rl.base_policy import BasePolicy
from agents.rl.ppo_policy import PPOPolicy
from agents.rl.sarsa_policy import SARSAPolicy


def create_agent(
    args,
    num_objectives: int,
    env=None,
) -> Union[BasePolicy, SimpleLLMAgent]:
    """
    Create an agent based on the specified type.

    Args:
        args: Command line arguments
        num_objectives: Number of objectives
        env: Gymnasium environment (required for RL policies)

    Returns:
        An instance of the specified agent type
    """
    if args.agent == "simple_llm":
        return SimpleLLMAgent(
            num_objectives=num_objectives,
            model_name=args.model_name,
            question_bank_path=args.benchmark_path,
            record_recommendations=True,
            target_skill_bundle=args.target_skill_bundle,
            prompt_type=args.prompt_type,
            objectives=args.objectives,
        )

    elif args.agent == "sarsa":
        if env is None:
            raise ValueError("Environment is required for SARSA agent creation")

        return SARSAPolicy(
            env=env,
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            target_skill_bundle=args.target_skill_bundle,
            seed=args.seed,
        )

    elif args.agent == "a2c":
        if env is None:
            raise ValueError("Environment is required for A2C agent creation")

        return A2CPolicy(
            env=env,
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            normalize_advantages=args.normalize_advantages,
            use_rms_prop=args.use_rms_prop,
            rms_prop_eps=args.rms_prop_eps,
            target_skill_bundle=args.target_skill_bundle,
        )

    elif args.agent == "ppo":
        if env is None:
            raise ValueError("Environment is required for PPO agent creation")

        return PPOPolicy(
            env=env,
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            mini_batch_size=args.mini_batch_size,
            n_epochs=args.n_epochs,
            normalize_advantages=args.normalize_advantages,
            target_skill_bundle=args.target_skill_bundle,
        )

    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
