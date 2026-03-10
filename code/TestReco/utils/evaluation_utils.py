"""
Utility functions for policy evaluation and result visualization.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from agents.llm_agents.simple_agent import SimpleLLMAgent
from agents.rl.base_policy import BasePolicy
from envs.education_env import EducationEnv
from generators.model import Claude_3_7_Sonnet_thinking, token_usage_callback

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values, handling numpy types in keys
        converted_dict = {}
        for key, value in obj.items():
            # Convert key to native Python type if it's a numpy type
            if isinstance(key, (np.integer, np.floating)):
                converted_key = int(key) if isinstance(key, np.integer) else float(key)
            else:
                converted_key = key
            converted_dict[converted_key] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        # Handle tuples (like in all_failed_questions values)
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    return obj

def evaluate_policy(
    env: EducationEnv,
    agent: Union[BasePolicy, SimpleLLMAgent],
    questions: List[Dict[str, Any]],
    args,
) -> Tuple[List[Dict[str, float]], List[Dict]]:
    """
    General evaluation function for all types of policies.

    Args:
        env: Educational environment
        agent: Policy to evaluate
        questions: List of questions
        args: Command line arguments
        model_dir: Optional model directory to save trajectories

    Returns:
        Tuple of (rewards_history, trajectories) where rewards_history is a list of dictionaries
    """
    # Initialize trajectories and rewards
    rewards_history = []  # List of dictionaries, each dict maps objective to reward
    trajectories = []

    # Try to load existing trajectories if filename is provided
    if hasattr(args, "trajectory_file") and args.trajectory_file:
        if os.path.exists(args.trajectory_file):
            with open(args.trajectory_file, "r") as f:
                trajectories = json.load(f)
                rewards_history = []
                for traj in trajectories:
                    episode_rewards = {}
                    for obj in env.objectives:
                        episode_rewards[obj] = sum(step["reward"]["reward_dict"][obj] for step in traj)
                    rewards_history.append(episode_rewards)
            logging.info(
                f"Loaded {len(trajectories)} existing trajectories from {args.trajectory_file}"
            )
        else:
            logging.warning(
                f"Specified trajectory file {args.trajectory_file} not found"
            )

    logging.info("=== Starting Policy Evaluation ===")

    # For each episode
    for episode in range(len(trajectories),  args.episodes):
        logging.info(f"Starting episode {episode}/{args.episodes}")

        # Reset environment
        state, info = env.reset()
        # Initialize episode rewards as dictionary
        episode_rewards = {obj: 0.0 for obj in env.objectives}
        trajectory = []

        # Continue episode
        for step in range(args.steps):
            logging.info("="*40)
            logging.info(f"Step {step+1} of episode {episode}")
            # Initialize variables for all agent types
            action_info = None
            latency = 0.0  

            # Select action based on agent type
            if isinstance(agent, SimpleLLMAgent):
                start_time = time.time()
                action_info = agent.select_action(env)
                end_time = time.time()
                latency = end_time - start_time
            elif args.agent in ["sarsa", "a2c", "ppo"]:
                start_time = time.time()
                action_info = agent.select_action(state, env=env)
                end_time = time.time()
                latency = end_time - start_time

            # Get action type from action_info
            action = action_info["action"]
            logging.info(f"action_info: {action_info}")

            next_state, reward_dict, truncated, info = env.step(action)

            # Record step information
            step_info = {
                "step": step,
                "state": {
                    "skill_difficulty_accuracy": info["skill_difficulty_accuracy"],
                    "skill_difficulty_counts": info["skill_difficulty_counts"],
                    "mastery": info["mastery"],
                    "failed_questions_ratio": info['failed_questions_ratio'],
                    "rolling_accuracy": info['rolling_accuracy'],
                    "all_failed_questions": info['all_failed_questions'],
                    "valid_failed_questions": info['valid_failed_questions'],
                    "cleared_questions": info['cleared_questions'],
                },
                "action": {
                    "action": action,
                    "selected_questions": info["selected_questions"],
                    "questions_info": info["questions_info"]
                },
                "reward": {
                    "reward_dict": reward_dict,
                    "total_reward": float(sum(reward_dict.values()))
                },
                "latency": latency,
            }

            trajectory.append(step_info)
            # Add rewards to episode totals
            for obj in env.objectives:
                episode_rewards[obj] += reward_dict[obj]

            logging.info(f"Selected Questions: {info['selected_questions']}")
            logging.info(f"Questions Info: {info['questions_info']}")
            logging.info(f"Rolling Accuracy: {info['rolling_accuracy']}")
            logging.info(f"Masteries: {info['mastery']}")
            logging.info(f"Rewards: {reward_dict}")
            logging.info(f"Inference Latency: {latency:.4f} seconds")

            if truncated:
                logging.info(f"Episode {episode} truncated at step {step}")
                break

            state = next_state

        # Record trajectory and rewards
        rewards_history.append(episode_rewards)
        trajectories.append(trajectory)

        logging.info(
            f"*** Episode {episode} complete with rewards: {episode_rewards} ***"
        )

            # Calculate and log average rewards
        avg_rewards = {}
        for obj in env.objectives:
            avg_rewards[obj] = np.mean([episode_rewards[obj] for episode_rewards in rewards_history])
        
        logging.info("=== Evaluation Summary ===")
        logging.info(f"Average Reward Per Objective:")
        for obj in env.objectives:
            logging.info(f"{obj}: {avg_rewards[obj]:.3f}")

        # Calculate weighted sum of rewards based on specified objectives
        weights = {obj: 1.0 / len(env.objectives) for obj in env.objectives}
        weighted_rewards = 0.0
        for obj in env.objectives:
            weighted_rewards += weights[obj] * avg_rewards[obj]

        logging.info(
            f"Weighted sum of rewards: {weighted_rewards:.3f}"
        )
        logging.info("="*40)

    # Log token usage statistics if using LLM agent
    if isinstance(agent, SimpleLLMAgent):
        if isinstance(agent.model, Claude_3_7_Sonnet_thinking):
            # For Claude model, get token usage directly from the model
            usage = agent.model.get_token_usage()
            total_prompt_tokens = usage["total_prompt_tokens"]
            total_completion_tokens = usage["total_completion_tokens"]
            total_requests = usage["request_count"]
        else:
            # For other models using LangChain
            total_prompt_tokens = token_usage_callback.total_prompt_tokens
            total_completion_tokens = token_usage_callback.total_completion_tokens
            total_requests = token_usage_callback.request_count

        total_tokens = total_prompt_tokens + total_completion_tokens

        logging.info("=== Token Usage Summary ===")
        logging.info(
            f"Total: {total_tokens} tokens ({total_prompt_tokens} prompt + {total_completion_tokens} completion)"
        )
        if total_requests > 0:
            logging.info(
                f"Average: {total_tokens/total_requests:.1f} tokens/request "
                f"({total_prompt_tokens/total_requests:.1f} prompt + {total_completion_tokens/total_requests:.1f} completion)"
            )
        logging.info(f"Total requests: {total_requests}")

    return rewards_history, trajectories
