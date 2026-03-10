"""
Main script for orchestrator evaluation in educational recommendation system.
"""

import argparse
import datetime
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time # Added for latency measurement

from envs.education_env import EducationEnv
from envs.response_models import BKTModel, IRTModel
from orchestrator.context_based_orchestrator import ContextBasedOrchestrator
from orchestrator.tool_call_orchestrator import ToolCallOrchestrator
from orchestrator.reflection_based_orchestrator import ReflectionBasedOrchestrator
from generators.factory import model_factory
from utils.benchmark_utils import load_benchmark_data
from agents.llm_agents.simple_agent import SimpleLLMAgent
from generators.model import Claude_3_7_Sonnet_thinking, token_usage_callback


class OrchestratorEvaluator:
    """Evaluator for orchestrator performance comparison."""
    
    def __init__(self, args, results_dir):
        self.args = args
        self.results_dir = results_dir
    
    def create_orchestrator(self, env):
        """Create the specified orchestrator."""
        # Create LLM
        llm = model_factory(self.args.model_name)
        
        # Create policy configurations, containing model_path, folder_path, config
        policy_configs = self._create_policy_configs()
        
        if self.args.orchestrator_type == "tool_call":
            orchestrator = ToolCallOrchestrator(
                env=env,
                llm=llm,
                policy_configs=policy_configs,
                verbose=self.args.verbose,
                objectives=self.args.objectives
            )
        elif self.args.orchestrator_type == "context_based":
            orchestrator = ContextBasedOrchestrator(
                env=env,
                llm=llm,
                policy_configs=policy_configs,
                verbose=self.args.verbose,
                objectives=self.args.objectives,
                rubric_path=self.args.rubric_path
            )
        elif self.args.orchestrator_type == "reflection_based":
            orchestrator = ReflectionBasedOrchestrator(
                env=env,
                llm=llm,
                policy_configs=policy_configs,
                verbose=self.args.verbose,
                objectives=self.args.objectives,
                max_rollouts=self.args.max_rollouts,
                rollout_steps=self.args.rollout_steps
            )
        # elif self.args.orchestrator_type == "intelligent":
        #     orchestrator = IntelligentOrchestrator(
        #         env=env,
        #         llm=llm,
        #         policy_configs=policy_configs,
        #         verbose=self.args.verbose,
        #         objectives=self.args.objectives,
        #         max_iterations=self.args.max_iterations if hasattr(self.args, 'max_iterations') else 3,
        #         decision_threshold=self.args.decision_threshold if hasattr(self.args, 'decision_threshold') else 0.8,
        #         enable_tool_calling=self.args.enable_tool_calling if hasattr(self.args, 'enable_tool_calling') else True
        #     )
        else:
            raise ValueError(f"Unknown orchestrator type: {self.args.orchestrator_type}")
        
        return orchestrator
    
    def _create_policy_configs(self):
        """Create policy configurations from trained policy folders."""
        # Create policy configs based on available model paths
        policy_configs = {}
        
        # Get model paths and load configs from folders
        model_paths = self._create_model_paths()
        
        for policy_name, model_info in model_paths.items():
            if model_info.get("folder_path") and os.path.exists(model_info.get("folder_path")):
                policy_configs[policy_name] = {
                    "folder_path": model_info.get("folder_path"),
                    "config": {}
                }

                # Load config from the folder
                config_file = os.path.join(model_info.get("folder_path"), "config.json")
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        policy_config = json.load(f)
                    policy_configs[policy_name]["config"] = policy_config
                else:
                    logging.warning(f"Config file not found in folder: {config_file}")
            else:
                logging.warning(f"Policy folder not found: {model_info.get('folder_path')}")
        return policy_configs
    
    def _create_model_paths(self):
        """Create model paths for trained policies."""
        model_paths = {}
        
        if hasattr(self.args, 'policy_folders') and self.args.policy_folders:
            for folder_path in self.args.policy_folders:
                if os.path.exists(folder_path):
                    # Check if it's a directory with subdirectories
                    if os.path.isdir(folder_path):
                        # Check if this directory itself contains a config.json (it's a policy folder)
                        config_file = os.path.join(folder_path, "config.json")
                        if os.path.exists(config_file):
                            # This is a policy folder
                            folder_name = os.path.basename(folder_path)
                            
                            # Try to determine policy type from folder name
                            if "sarsa" in folder_name.lower():
                                policy_name = "sarsa"
                            elif "ppo" in folder_name.lower():
                                policy_name = "ppo"
                            elif "a2c" in folder_name.lower():
                                policy_name = "a2c"
                            else:
                                policy_name = "unknown"
                            
                            # Create unique policy name
                            policy_key = f"{policy_name}_{len(model_paths)}"
                            
                            model_paths[policy_key] = {
                                "folder_path": folder_path,
                            }
                            logging.info(f"Found policy folder: {folder_path} (named: {policy_key})")
                        else:
                            # This might be a directory containing multiple policy folders
                            # Look for subdirectories that contain config.json
                            logging.info(f"Scanning directory for policy folders: {folder_path}")
                            found_any = False
                            for item in os.listdir(folder_path):
                                item_path = os.path.join(folder_path, item)
                                if os.path.isdir(item_path):
                                    sub_config_file = os.path.join(item_path, "config.json")
                                    if os.path.exists(sub_config_file):
                                        found_any = True
                                        folder_name = os.path.basename(item_path)
                                        
                                        # Try to determine policy type from folder name
                                        if "sarsa" in folder_name.lower():
                                            policy_name = "sarsa"
                                        elif "ppo" in folder_name.lower():
                                            policy_name = "ppo"
                                        elif "a2c" in folder_name.lower():
                                            policy_name = "a2c"
                                        else:
                                            policy_name = "unknown"
                                        
                                        # Create unique policy name
                                        policy_key = f"{policy_name}_{len(model_paths)}"
                                        
                                        model_paths[policy_key] = {
                                            "folder_path": item_path,
                                        }
                                        logging.info(f"Found policy folder: {item_path} (named: {policy_key})")
                            
                            if not found_any:
                                logging.warning(f"No policy folders found in directory: {folder_path}")
                    else:
                        logging.warning(f"Policy path is not a directory: {folder_path}")
                else:
                    logging.warning(f"Policy folder not found: {folder_path}")
        
        if not model_paths:
            logging.warning("No policy folders found. Please check your --policy_folders argument.")
        
        return model_paths
    
    def _get_step_token_usage(self, orchestrator, step_start_prompt_tokens=None, step_start_completion_tokens=None):
        """Get token usage for the current step."""
        # Check if this is a Claude model that has its own token tracking
        if isinstance(orchestrator.custom_llm, Claude_3_7_Sonnet_thinking):
            # For Claude model, get current token usage from the model
            usage = orchestrator.custom_llm.get_token_usage()
            current_prompt_tokens = usage["total_prompt_tokens"]
            current_completion_tokens = usage["total_completion_tokens"]
            
            # Calculate step tokens by subtracting step start values
            if step_start_prompt_tokens is not None and step_start_completion_tokens is not None:
                step_input_tokens = current_prompt_tokens - step_start_prompt_tokens
                step_output_tokens = current_completion_tokens - step_start_completion_tokens
            else:
                raise ValueError("Step start prompt and completion tokens are not set")
        else:
            # For other models, use token_usage_callback
            if step_start_prompt_tokens is not None and step_start_completion_tokens is not None:
                # Calculate tokens used in this step by subtracting step start values
                step_input_tokens = token_usage_callback.total_prompt_tokens - step_start_prompt_tokens
                step_output_tokens = token_usage_callback.total_completion_tokens - step_start_completion_tokens
            else:
                raise ValueError("Step start prompt and completion tokens are not set")
        
        return {
            "input_tokens": step_input_tokens,
            "output_tokens": step_output_tokens,
            "total_tokens": step_input_tokens + step_output_tokens
        }
    
    def run_evaluation(self, env, orchestrator, questions, difficulties):
        """Run evaluation for specified number of episodes."""
        logging.info(f"Starting evaluation with {self.args.episodes} episodes")
        
        # Initialize results storage
        results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_trajectories": [],
            "objective_rewards": {obj: [] for obj in env.objectives},
            "final_mastery": [],
            "policy_selections": [],
            "orchestrator_info": orchestrator.get_orchestrator_info()
        }

        logging.info("=== Starting Policy Evaluation ===")
        # logging.info(f"Number of questions: {len(questions)}")
        # logging.info(f"Objectives: {env.objectives}")
    
        for episode in range(self.args.episodes):
            logging.info(f"Starting episode {episode + 1}/{self.args.episodes}")
        
            # Reset environment
            state, info = env.reset()
            episode_rewards = {obj: 0.0 for obj in env.objectives}
            trajectory = []
            policy_selections = []
        
            for step in range(self.args.max_steps):
                # Initialize variables for all agent types
                action = None
                action_info = None
                latency = 0.0  # Initialize latency

                previous_state_info = orchestrator._extract_state_info()

                # Record token counters at the start of each step
                if isinstance(orchestrator.custom_llm, Claude_3_7_Sonnet_thinking):
                    # For Claude model, get token usage from the model
                    usage = orchestrator.custom_llm.get_token_usage()
                    step_start_prompt_tokens = usage["total_prompt_tokens"]
                    step_start_completion_tokens = usage["total_completion_tokens"]
                else:
                    # For other models, use token_usage_callback
                    step_start_prompt_tokens = token_usage_callback.total_prompt_tokens
                    step_start_completion_tokens = token_usage_callback.total_completion_tokens

                # Record start time for latency measurement
                start_time = time.time()

                # Select action using orchestrator
                action_info, orchestrator_info = orchestrator.select_action(state)
                
                # Record end time and calculate latency
                end_time = time.time()
                latency = end_time - start_time
                
                # Get token usage for this step (cumulative across all LLM calls in this step)
                step_token_usage = self._get_step_token_usage(orchestrator, step_start_prompt_tokens, step_start_completion_tokens)
                
                # Record policy selection if available
                if "selected_strategy" in orchestrator_info:
                    policy_selections.append(orchestrator_info["selected_strategy"])

                # Extract action type from action_info
                action = action_info["action"]
                
                # Take step in environment with action type
                next_state, reward_dict, truncated, info = env.step(action)
                
                # Save interaction with feedback to orchestrator memory (if context_based orchestrator)
                if hasattr(orchestrator, 'save_interaction_with_feedback'):
                    # Get state_info from orchestrator (we need to reconstruct it)
                    orchestrator.save_interaction_with_feedback(
                        state_info=previous_state_info,
                        selected_strategy=orchestrator_info["selected_strategy"],
                        action_info=action_info,
                        next_state=next_state,
                        reward_dict=reward_dict,
                        info=info
                    )
                
                # Record detailed step information (similar to evaluation_utils.py)
                step_info = {
                    "step": step,
                    "reward_dict": reward_dict,
                    "orchestrator_info": orchestrator_info,
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
                    "token": step_token_usage,
                }

                if "response" in action_info:
                    step_info["orchestrator_response"] = action_info["response"]
                if "selected_strategy" in orchestrator_info:
                    step_info["selected_strategy"] = orchestrator_info["selected_strategy"]

                trajectory.append(step_info)
                
                # Accumulate rewards
                for obj, reward in reward_dict.items():
                    if obj in episode_rewards:
                        episode_rewards[obj] += reward
            
                # Log step information
                logging.info(f"Step {step+1}: Strategy={orchestrator_info['selected_strategy']}, Action={action}, Selected Questions={info['selected_questions']}, Mastery={info['mastery']}, Reward={reward_dict}, Latency={latency:.3f}s")
            
                if truncated:
                    break
                
                state = next_state
        
            # Record episode results
            results["episode_rewards"].append(episode_rewards)
            results["episode_lengths"].append(step+1)
            results["episode_trajectories"].append(trajectory)
            results["policy_selections"].append(policy_selections)
            
            # Record final mastery from the last step
            if trajectory:
                final_mastery = trajectory[-1]['state']['mastery']
                results["final_mastery"].append(final_mastery)
            
            logging.info(
                f"*** Episode {episode} complete with rewards: {episode_rewards} ***"
            )
        
        return results
    
    def analyze_results(self, orchestrator, results, env):
        """Analyze and print detailed results."""
        if not results["episode_rewards"]:
            logging.warning("No episode rewards data available for analysis")
            return
            
        # Calculate average rewards for each objective
        avg_rewards = {}
        for obj in env.objectives:
            obj_rewards = [episode_rewards[obj] for episode_rewards in results["episode_rewards"]]
            avg_rewards[obj] = np.mean(obj_rewards)
        
        logging.info("=== Evaluation Summary ===")
        logging.info(f"Average rewards across all trajectories:")
        for obj, reward in avg_rewards.items():
            logging.info(f"{obj}: {reward:.3f}")
        
        # Calculate weighted sum of rewards based on specified objectives
        weighted_rewards = 0.0
        for obj in env.objectives:
            weight = 1 / (len(env.objectives))
            weighted_rewards += weight * avg_rewards[obj]

        logging.info(f"Weighted sum of rewards: {weighted_rewards:.3f} with weight per objective: {weight:.3f}")

        # Log token usage statistics for orchestrator
        if isinstance(orchestrator.custom_llm, Claude_3_7_Sonnet_thinking):
            # For Claude model, get token usage directly from the model
            usage = orchestrator.custom_llm.get_token_usage()
            total_prompt_tokens = usage["total_prompt_tokens"]
            total_completion_tokens = usage["total_completion_tokens"]
            total_requests = usage["request_count"]
        else:
            total_prompt_tokens = token_usage_callback.total_prompt_tokens
            total_completion_tokens = token_usage_callback.total_completion_tokens
            total_requests = token_usage_callback.request_count

        total_tokens = total_prompt_tokens + total_completion_tokens

        logging.info(f"Token Usage: {total_tokens} tokens ({total_requests} requests)")
        if total_requests > 0:
            logging.info(f"Average: {total_tokens/total_requests:.1f} tokens/request")
        else:
            logging.info("Average: N/A (no requests tracked)")

        # Trajectory analysis
        self._analyze_trajectories(results)
        
        # Policy selection analysis (if available)
        if any(results["policy_selections"]):
            self._analyze_policy_selections(results)
        
        # Mastery analysis
        self._analyze_mastery(results, env)
        
        # Save detailed results
        self._save_results(results, env)
    
    def _analyze_trajectories(self, results):
        """Analyze detailed trajectory information."""
        logging.info("\nTrajectory Analysis:")
        
        all_latencies = []
        all_correct_answers = 0
        total_questions = 0
        all_skills_tested = []
        all_input_tokens = []
        all_output_tokens = []
        all_total_tokens = []
        
        for episode_idx, trajectory in enumerate(results["episode_trajectories"]):
            episode_correct = 0
            episode_questions = 0
            
            for step_info in trajectory:
                # Collect latency data
                if "latency" in step_info:
                    all_latencies.append(step_info["latency"])
                
                # Collect token usage data
                if "token" in step_info:
                    token_data = step_info["token"]
                    if "input_tokens" in token_data:
                        all_input_tokens.append(token_data["input_tokens"])
                    if "output_tokens" in token_data:
                        all_output_tokens.append(token_data["output_tokens"])
                    if "total_tokens" in token_data:
                        all_total_tokens.append(token_data["total_tokens"])
                
                # Count correct answers from questions_info
                if "action" in step_info and "questions_info" in step_info["action"]:
                    questions_info = step_info["action"]["questions_info"]
                    for q_info in questions_info:
                        if isinstance(q_info, dict) and "correct" in q_info:
                            if q_info["correct"] is True:
                                episode_correct += 1
                                all_correct_answers += 1
                            episode_questions += 1
                            total_questions += 1
                
                # Collect skills tested from questions_info
                if "action" in step_info and "questions_info" in step_info["action"]:
                    questions_info = step_info["action"]["questions_info"]
                    for q_info in questions_info:
                        if isinstance(q_info, dict) and "skills" in q_info:
                            all_skills_tested.extend(q_info["skills"])
            
            # Episode-specific statistics
            if episode_questions > 0:
                episode_correct_rate = episode_correct / episode_questions
                logging.info(f"Episode {episode_idx + 1}: "
                           f"Correct Rate: {episode_correct_rate:.2%} "
                           f"({episode_correct}/{episode_questions})")
        
        # Overall statistics
        if total_questions > 0:
            overall_correct_rate = all_correct_answers / total_questions
            logging.info(f"\nOverall Correct Rate: {overall_correct_rate:.2%} "
                        f"({all_correct_answers}/{total_questions})")
        
        if all_latencies:
            avg_latency = np.mean(all_latencies)
            std_latency = np.std(all_latencies)
            logging.info(f"Average Latency: {avg_latency:.4f} ± {std_latency:.4f} seconds")
        else:
            logging.info("No latency data available for analysis")
        
        # Token usage analysis
        if all_input_tokens:
            avg_input_tokens = np.mean(all_input_tokens)
            total_input_tokens = np.sum(all_input_tokens)
            logging.info(f"Average Input Tokens per Step: {avg_input_tokens:.1f}")
            logging.info(f"Total Input Tokens: {total_input_tokens}")
        
        if all_output_tokens:
            avg_output_tokens = np.mean(all_output_tokens)
            total_output_tokens = np.sum(all_output_tokens)
            logging.info(f"Average Output Tokens per Step: {avg_output_tokens:.1f}")
            logging.info(f"Total Output Tokens: {total_output_tokens}")
        
        if all_total_tokens:
            avg_total_tokens = np.mean(all_total_tokens)
            total_tokens = np.sum(all_total_tokens)
            logging.info(f"Average Total Tokens per Step: {avg_total_tokens:.1f}")
            logging.info(f"Total Tokens Used: {total_tokens}")
        else:
            logging.info("No token usage data available for analysis")
        
        # Skills analysis
        if all_skills_tested:
            from collections import Counter
            skill_counts = Counter(all_skills_tested)
            logging.info(f"\nSkills Tested (frequency):")
            for skill, count in skill_counts.most_common():
                logging.info(f"  {skill}: {count} times")
    
    def _analyze_policy_selections(self, results):
        """Analyze policy selection patterns."""
        logging.info("\nPolicy Selection Analysis:")
        
        # Flatten all policy selections
        all_selections = []
        for episode_selections in results["policy_selections"]:
            all_selections.extend(episode_selections)
        
        if all_selections:
            from collections import Counter
            selection_counts = Counter(all_selections)
            total_selections = len(all_selections)
            
            logging.info("Policy Selection Frequency:")
            for policy, count in selection_counts.most_common():
                percentage = (count / total_selections) * 100
                logging.info(f"  {policy}: {count} ({percentage:.1f}%)")
    
    def _analyze_mastery(self, results, env):
        """Analyze mastery level improvements."""
        logging.info("\nMastery Analysis:")
        
        # Check if we have final mastery data
        if not results["final_mastery"]:
            logging.info("No final mastery data available for analysis")
            return
        
        # Calculate average final mastery for target skills
        target_skills = env.target_skill_bundle
        final_mastery_values = []
        
        for episode_mastery in results["final_mastery"]:
            if episode_mastery:  # Check if mastery data exists
                episode_avg = np.mean([
                    episode_mastery.get(skill, 0.0) 
                    for skill in target_skills
                ])
                final_mastery_values.append(episode_avg)
        
        if final_mastery_values:
            avg_final_mastery = np.mean(final_mastery_values)
            std_final_mastery = np.std(final_mastery_values)
            
            logging.info(f"Average Final Mastery: {avg_final_mastery:.3f} ± {std_final_mastery:.3f}")
            
            # Skill-specific mastery
            for skill in target_skills:
                skill_mastery = [
                    episode_mastery.get(skill, 0.0) 
                    for episode_mastery in results["final_mastery"]
                    if episode_mastery  # Only include episodes with mastery data
                ]
                if skill_mastery:
                    avg_skill_mastery = np.mean(skill_mastery)
                    logging.info(f"  {skill}: {avg_skill_mastery:.3f}")
        else:
            logging.info("No valid mastery data found for analysis")
    
    def _save_results(self, results, env):
        """Save detailed results to files."""
        # Save configuration
        self._save_config()
        
        # Save detailed trajectories (similar to train_evaluate_policy.py)
        from utils.evaluation_utils import convert_numpy_types
        trajectories_file = os.path.join(self.results_dir, "evaluation_trajectories.json")
        with open(trajectories_file, 'w') as f:
            json.dump(results["episode_trajectories"], f, indent=2, default=convert_numpy_types)
        
        # Save rewards history as JSON (since it's now a list of dictionaries)
        rewards_file = os.path.join(self.results_dir, "evaluation_rewards_history.json")
        with open(rewards_file, 'w') as f:
            json.dump(results["episode_rewards"], f, indent=2, default=convert_numpy_types)
        
        # Create and save visualization
        self._plot_evaluation_rewards(results["episode_rewards"], env.objectives)
        
        # Create and save mastery progression visualization
        if results["episode_trajectories"] and hasattr(env, 'target_skill_bundle'):
            self._plot_mastery_progression(
                trajectories=results["episode_trajectories"],
                target_skills=env.target_skill_bundle,
                max_steps=self.args.max_steps,
                early_stop_threshold=self.args.early_stop_threshold
            )
        
        # Save orchestrator profile statistics
        orchestrator_profile_file = os.path.join(self.results_dir, "orchestrator_profile.json")
        
        # Calculate orchestrator-level statistics
        orchestrator_profile = {}
        
        if results["episode_rewards"]:
            # Calculate scalarized rewards (simple average of all objectives)
            scalar_rewards = []
            for episode_rewards in results["episode_rewards"]:
                episode_scalar = np.mean(list(episode_rewards.values()))
                scalar_rewards.append(episode_scalar)
            
            avg_scalar_reward = float(np.mean(scalar_rewards))
            std_scalar_reward = float(np.std(scalar_rewards))
            
            # Calculate average reward for each objective
            objective_avg_rewards = {}
            for obj in env.objectives:
                obj_rewards = [episode_rewards[obj] for episode_rewards in results["episode_rewards"]]
                objective_avg_rewards[obj] = float(np.mean(obj_rewards))
            
            # Calculate average steps per episode
            avg_steps_per_episode = float(np.mean(results["episode_lengths"]))
            
            orchestrator_profile = {
                "avg_scalar_reward": avg_scalar_reward,
                "std_scalar_reward": std_scalar_reward,
                "objective_avg_rewards": objective_avg_rewards,
                "avg_steps_per_episode": avg_steps_per_episode,
                "objectives": env.objectives,
                "orchestrator_type": self.args.orchestrator_type,
                "model_name": self.args.model_name,
                "episodes": self.args.episodes
            }
        else:
            raise Exception("No episode rewards data available for analysis")
        
        # Save orchestrator profile to JSON file
        with open(orchestrator_profile_file, 'w') as f:
            json.dump(orchestrator_profile, f, indent=2)
        
        logging.info(f"Saved orchestrator profile to {orchestrator_profile_file}")
        
        # Log the statistics
        logging.info(f"\n=== Orchestrator Profile ===")
        logging.info(f"Orchestrator Type: {self.args.orchestrator_type}")
        logging.info(f"Model: {self.args.model_name}")
        logging.info(f"Episodes: {self.args.episodes}")
        logging.info(f"Average Scalar Reward: {orchestrator_profile.get('avg_scalar_reward', 0):.4f}")
        logging.info(f"Standard Deviation Scalar Reward: {orchestrator_profile.get('std_scalar_reward', 0):.4f}")
        logging.info(f"Average Steps per Episode: {orchestrator_profile.get('avg_steps_per_episode', 0):.2f}")
        logging.info(f"Objective Average Rewards:")
        for objective, avg_reward in orchestrator_profile.get('objective_avg_rewards', {}).items():
            logging.info(f"  {objective}: {avg_reward:.4f}")
    
    def _save_config(self):
        """Save configuration to results directory."""
        config = {
            "orchestrator_type": self.args.orchestrator_type,
            "model_name": self.args.model_name,
            "response_model": self.args.response_model,
            "data_type": self.args.data_type,
            "benchmark": self.args.benchmark,
            "target_skill_bundle": self.args.target_skill_bundle,
            "objectives": self.args.objectives,
            "max_steps": self.args.max_steps,
            "early_stop_threshold": self.args.early_stop_threshold,
            "episodes": self.args.episodes,
            "seed": self.args.seed,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        config_file = os.path.join(self.results_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _plot_evaluation_rewards(self, rewards_history, objectives):
        """Plot evaluation rewards and save to results directory."""
        num_episodes = len(rewards_history)
        num_objectives = len(objectives)

        plt.figure(figsize=(12, 8))

        for obj in objectives:
            obj_rewards = [episode_rewards[obj] for episode_rewards in rewards_history]
            plt.plot(range(1, num_episodes + 1), obj_rewards, 
                    label=obj, marker='o', markersize=3)

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"{self.args.orchestrator_type.upper()} Evaluation Rewards Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save to results directory
        output_file = os.path.join(self.results_dir, "evaluation_reward_performance.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_mastery_progression(self, trajectories, target_skills, max_steps, early_stop_threshold=0.8):
        """
        Plot average mastery progression over time for target skills.
        
        Args:
            trajectories: List of episode trajectories
            target_skills: List of target skills to track
            max_steps: Maximum number of steps per episode
            early_stop_threshold: Mastery threshold for early stopping
        """
        # Initialize mastery tracking
        step_mastery_data = {skill: [] for skill in target_skills}
        episode_lengths = []
        
        # Collect mastery data from all trajectories
        for trajectory in trajectories:
            episode_lengths.append(len(trajectory))
            
            for step_idx, step_info in enumerate(trajectory):
                if "state" in step_info and step_info["state"] and "mastery" in step_info["state"]:
                    mastery = step_info["state"]["mastery"]
                    
                    # Calculate average mastery for target skills at this step
                    for skill in target_skills:
                        if skill in mastery:
                            # Ensure we have enough data points for this step
                            while len(step_mastery_data[skill]) <= step_idx:
                                step_mastery_data[skill].append([])
                            step_mastery_data[skill][step_idx].append(mastery[skill])
        
        # Calculate average mastery for each step
        avg_mastery_by_step = {}
        for skill in target_skills:
            avg_mastery_by_step[skill] = []
            for step_data in step_mastery_data[skill]:
                if step_data:  # Only calculate average if we have data for this step
                    avg_mastery_by_step[skill].append(np.mean(step_data))
                else:
                    avg_mastery_by_step[skill].append(np.nan)
        
        plt.figure(figsize=(14, 8))
        
        # Define colors for different skills
        colors = sns.color_palette("husl", len(target_skills))
        
        # Plot mastery progression for each skill
        for i, skill in enumerate(target_skills):
            mastery_values = avg_mastery_by_step[skill]
            # Remove NaN values for plotting
            valid_indices = [j for j, val in enumerate(mastery_values) if not np.isnan(val)]
            if valid_indices:
                steps = [j + 1 for j in valid_indices]  # Step numbers start from 1
                values = [mastery_values[j] for j in valid_indices]
                
                # Plot mastery progression line with thicker line
                plt.plot(steps, values, 
                        color=colors[i], 
                        linestyle='-',  # Solid line
                        linewidth=3,  # Thicker line
                        label=f"{skill} ({self.args.orchestrator_type})",
                        alpha=0.8)
        
        # Add mastery threshold horizontal line at 0.8
        plt.axhline(y=early_stop_threshold, color='red', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Mastery Threshold ({early_stop_threshold})')
        
        # Add star for average episode length 
        if episode_lengths:
            avg_episode_length = np.mean(episode_lengths)
            # Find the closest step with data for the star
            max_step_with_data = max([len(data) for data in step_mastery_data.values() if data])
            if max_step_with_data > 0:
                star_step = min(int(avg_episode_length), max_step_with_data)
                # Calculate average mastery at this step across all skills
                avg_mastery_at_star = []
                for skill in target_skills:
                    if star_step <= len(avg_mastery_by_step[skill]) and not np.isnan(avg_mastery_by_step[skill][star_step - 1]):
                        avg_mastery_at_star.append(avg_mastery_by_step[skill][star_step - 1])
                
                if avg_mastery_at_star:
                    star_mastery = np.mean(avg_mastery_at_star)
                    plt.scatter(avg_episode_length, star_mastery, 
                              marker='*', 
                              s=200,  
                              color='black',
                              edgecolors='black',
                              linewidth=1,
                              zorder=5,
                              label=f'Average Episode Length ({avg_episode_length:.1f} steps)')
        
        # Customize the plot
        plt.xlabel('Step Number (1-30)', fontsize=16, fontweight='bold')
        plt.ylabel('Average Mastery Level', fontsize=16, fontweight='bold')
        plt.title(f'Mastery Progression Over Time - {self.args.orchestrator_type.upper()}', fontsize=18, fontweight='bold')
        plt.legend(loc='best', fontsize=12, frameon=True)
        plt.grid(True, alpha=0.3)
        plt.xlim(0.5, max_steps + 0.5)
        plt.ylim(0, 1.05)
        
        # Make tick labels larger and bold
        plt.tick_params(axis='both', which='major', labelsize=14)
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save to results directory
        output_file = os.path.join(self.results_dir, "evaluation_mastery_progression.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Mastery progression plot saved to: {output_file}")
        
        plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Orchestrator Evaluation")
    
    # Environment settings
    parser.add_argument("--data_type", type=str, default="QA", 
                       choices=["QA"], help="Data type")
    parser.add_argument("--benchmark", type=str, default="math_bench", 
                       choices=["medmcqa", "math_bench"], help="Benchmark dataset to use")
    parser.add_argument("--benchmark_path", type=str, default="data/benchmark",
                       help="Path to benchmark data directory")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of questions to load from benchmark (for testing)")
    parser.add_argument("--target_skill_bundle", nargs="+", 
                       default=["Linear Algebra"],
                       help="List of target skills to focus on during learning")
    parser.add_argument("--objectives", nargs="+", 
                       default=["aptitude", "gap", "performance"],
                       help="List of objectives to optimize")
    parser.add_argument("--response_model", type=str, default="irt",
                       choices=["irt", "bkt"], help="Response model to use")
    parser.add_argument("--early_stop_threshold", type=float, default=0.8,
                       help="Early stopping threshold")
    
    # Orchestrator settings
    parser.add_argument("--orchestrator_type", type=str, default="context_based",
                       choices=["context_based", "tool_call", "reflection_based"], 
                       help="Type of orchestrator")
    parser.add_argument("--model_name", type=str, default="claude-3.7-sonnet-thinking", # llama-3-8b
                       help="LLM model name")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    
    # Evaluation settings
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=30,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--results_dir", type=str, default="trash_results",
                       help="Directory to save results")
    
    # Policy folders
    parser.add_argument("--policy_folders", nargs="+", type=str, default=["Policy_Set_Results_IRT_005v4/1O_ppo_scalarized_aptitude_irt_2025-09-17_11-28-43_model", "Policy_Set_Results_IRT_005v4/1O_ppo_scalarized_gap_irt_2025-09-17_11-28-43_model", "Policy_Set_Results_IRT_005v4/1O_ppo_scalarized_performance_irt_2025-09-17_11-28-38_model"],
                       help="List of paths to folders containing trained policy models")
    parser.add_argument("--rubric_path", type=str, default="configs/rubric.json",
                       help="Path to rubric file for policy classification")
    
    # Reflection-based orchestrator specific settings
    parser.add_argument("--max_rollouts", type=int, default=3,
                       help="Maximum number of rollouts before forcing a decision (reflection_based only)")
    parser.add_argument("--rollout_steps", type=int, default=3,
                       help="Number of steps in each rollout simulation (reflection_based only)")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"3O_{args.orchestrator_type}_{args.model_name}_{args.response_model}_{timestamp}"
    results_dir = os.path.join(args.results_dir, dir_name)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(results_dir, "evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
            # Removed StreamHandler to prevent terminal output
        ]
    )
    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Arguments: {args}")
    
    # Create evaluator
    evaluator = OrchestratorEvaluator(args, results_dir)
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load benchmark data
    questions, question_skill_map, unique_skills = load_benchmark_data(args)
    
    # Rescale difficulties and save/load from CSV
    from train_evaluate_policy import rescale_difficulties
    difficulties = rescale_difficulties(questions, args)
    
    # Create response model
    if args.response_model == "irt":
        response_model = IRTModel(len(unique_skills), seed=args.seed)
        # Set question parameters for IRT model using rescaled difficulties
        question_difficulties = {i: difficulties[i]["scaled_difficulty"] for i in range(len(questions))}
        response_model.set_question_params_batch(question_difficulties)
    elif args.response_model == "bkt":
        response_model = BKTModel(len(unique_skills), seed=args.seed)
    else:
        response_model = None
    
    # Create environment
    env = EducationEnv(
        skills=args.target_skill_bundle,
        num_questions=len(questions),
        objectives=args.objectives,
        response_model=response_model,
        target_skill_bundle=args.target_skill_bundle,
        max_steps=args.max_steps,
        early_stop_threshold=args.early_stop_threshold,
        seed=args.seed
    )

    # Set difficulty levels from question bank
    env.set_difficulty_levels(difficulties)

    # Set question skills mapping
    for question_idx, question_info in question_skill_map.items():
        env.set_question_skills_difficulty(question_idx, question_info["skills"])

    # Create orchestrator
    orchestrator = evaluator.create_orchestrator(env)
    logging.info(f"Orchestrator created: {args.orchestrator_type}")
    
    # Run evaluation
    results = evaluator.run_evaluation(env, orchestrator, questions, difficulties)
    
    # Analyze and save results
    evaluator.analyze_results(orchestrator, results, env)
    
    logging.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 