import argparse
import datetime
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from envs.education_env import EducationEnv
from envs.response_models import BKTModel, IRTModel
from reward_handlers.pareto_buffer_handler import ParetoBufferHandler
from reward_handlers.reward_machine_handler import (
    EduRewardMachine,
    RewardMachineHandler,
)
from reward_handlers.scalarized_handler import ScalarizedRewardHandler
from utils.agent_utils import create_agent
from utils.benchmark_utils import load_benchmark_data
from utils.evaluation_utils import (
    convert_numpy_types,
    evaluate_policy,
)
from utils.rl_trainer import RewardHandlerRLTrainer
from utils.utils import load_base_config, apply_base_config, validate_agent_reward_combination


def rescale_difficulties(questions: List[Dict[str, Any]], args) -> Dict[int, float]:
    """
    Rescale difficulty levels by binning them uniformly across 0-1 range.
    
    Args:
        questions: List of question dictionaries
        args: Command-line arguments containing benchmark and data_type
        
    Returns:
        Dictionary mapping question indices to rescaled difficulty values
    """
    # Create output directory
    output_dir = f"benchmarks/{args.benchmark.upper()}/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define CSV file path
    csv_path = os.path.join(output_dir, f"{args.benchmark}_{args.data_type}_scaled_difficulty.csv")
    
    # Check if CSV file already exists
    if os.path.exists(csv_path):
        print(f"Loading rescaled difficulties from {csv_path}")
        df = pd.read_csv(csv_path)
        difficulties = {}
        for _, row in df.iterrows():
            difficulties[int(row['question_id'])] = {"scaled_difficulty": float(row['scaled_difficulty']), "original_difficulty": float(row['original_difficulty'])}
        return difficulties
    
    print(f"Rescaling difficulties and saving to {csv_path}")
    
    # Extract original difficulties
    original_difficulties = []
    for i, question in enumerate(questions):
        original_difficulties.append(float(question["difficulty"]))
    
    # Get unique difficulty levels
    unique_difficulties = sorted(list(set(original_difficulties)))
    n_difficulties = len(unique_difficulties)
    
    print(f"Found {n_difficulties} unique difficulty levels: {unique_difficulties}")
    
    # Handle edge case: if only one difficulty level, assign 0.5
    if n_difficulties == 1:
        logging.error("Only one difficulty level found, assigning 0.5 to all questions")
        raise ValueError("Only one difficulty level found, assigning 0.5 to all questions")
    
    # Create bins for rescaling
    # Divide 0-1 into n_difficulties bins
    bin_edges = np.linspace(0, 1, n_difficulties + 1)
    
    # Create mapping from original difficulty to bin index
    difficulty_to_bin = {}
    for i, orig_diff in enumerate(unique_difficulties):
        difficulty_to_bin[orig_diff] = i
    
    # Create rescaled difficulties dictionary
    difficulties = {}
    for i, question in enumerate(questions):
        orig_diff = float(question["difficulty"])

        # Use dataset question id if present; fallback to list index
        qid_raw = question.get("id", i)
        try:
            qid = int(qid_raw)
        except Exception:
            qid = int(i)

        # Find which bin this difficulty belongs to
        bin_idx = difficulty_to_bin[orig_diff]
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]

        # Generate rescaled difficulty within this bin
        rescaled_diff = np.random.uniform(bin_start, bin_end)
        rescaled_diff = round(round(rescaled_diff / 0.05) * 0.05, 2)

        difficulties[qid] = {"scaled_difficulty": rescaled_diff, "original_difficulty": orig_diff}
        print(f"Question {qid}: difficulty {orig_diff} -> {rescaled_diff} (bin [{bin_start:.3f}, {bin_end:.3f}])")
# Save to CSV
    data = []
    for i, question in enumerate(questions):
        qid_raw = question.get("id", i)
        try:
            qid = int(qid_raw)
        except Exception:
            qid = int(i)
        data.append({
            'question_id': qid,
            'original_difficulty': difficulties[qid]["original_difficulty"],
            'scaled_difficulty': difficulties[qid]["scaled_difficulty"],
            # 'text': question.get("text", "")[:100] + "..." if len(question.get("text", "")) > 100 else question.get("text", ""),
            # 'topic': question.get("topic", "")
        })
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Saved rescaled difficulties to {csv_path}")
    
    return difficulties


class ModelManager:
    """manage model directory and save logic for all agents"""
    
    def __init__(self, args):
        self.args = args
        self.model_dir = self._generate_model_dir()
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _generate_model_dir(self):
        """generate model directory path"""
        objective_str = "_".join(self.args.objectives)
        if self.args.agent == "simple_llm":
            base_name = f"{len(self.args.objectives)}O_{self.args.agent}_{self.args.prompt_type}_{self.args.model_name}_{objective_str}_{self.args.response_model}_{self.args.timestamp}"
        else:
            base_name = f"{len(self.args.objectives)}O_{self.args.agent}_{self.args.reward_mode}_{objective_str}_{self.args.response_model}_{self.args.timestamp}"
        return f"{self.args.results_dir}/{base_name}_model"
    
    def save_config(self, agent_config=None):
        """save config to model directory"""
        config = {
            "agent_type": self.args.agent,
            "response_model": self.args.response_model,
            "data_type": self.args.data_type,
            "benchmark": self.args.benchmark,
            "target_skill_bundle": self.args.target_skill_bundle,
            "objectives": self.args.objectives,
            "timestamp": self.args.timestamp,
            "steps": self.args.steps,
            "episodes": self.args.episodes,
            "seed": self.args.seed
        }
        
        # add agent-specific config
        if self.args.agent == "simple_llm":
            config.update({
                "model_name": self.args.model_name,
                "prompt_type": self.args.prompt_type,
                "num_objectives": len(self.args.objectives) + 1,
                "weights": [1 / (len(self.args.objectives) + 1)] * (len(self.args.objectives) + 1),
                "question_bank_path": self.args.benchmark_path,
                "target_skill_bundle": self.args.target_skill_bundle,
                "record_recommendations": True,
                "objectives": self.args.objectives,
            })
        else:
            config.update({
                "reward_mode": self.args.reward_mode,
                "learning_rate": self.args.learning_rate,
                "gamma": self.args.gamma,
                "hidden_dims": self.args.hidden_dims,
            })
            
            # add agent-specific hyperparameters
            if self.args.agent == "sarsa":
                config.update({
                    "epsilon": self.args.epsilon,
                    "epsilon_decay": self.args.epsilon_decay,
                    "epsilon_min": self.args.epsilon_min,
                })
            elif self.args.agent == "a2c":
                config.update({
                    "reward_mode": self.args.reward_mode,
                    "learning_rate": self.args.learning_rate,
                    "gamma": self.args.gamma,
                    "hidden_dims": self.args.hidden_dims,
                    "a2c_n_steps": self.args.a2c_n_steps,
                    "entropy_coef": self.args.entropy_coef,
                    "value_coef": self.args.value_coef,
                    "max_grad_norm": self.args.max_grad_norm,
                    "use_rms_prop": self.args.use_rms_prop,
                    "rms_prop_eps": self.args.rms_prop_eps,
                })
            elif self.args.agent == "ppo":
                config.update({
                    "ppo_n_steps": self.args.ppo_n_steps,
                    "n_epochs": self.args.n_epochs,
                    "gae_lambda": self.args.gae_lambda,
                    "clip_ratio": self.args.clip_ratio,
                    "target_kl": self.args.target_kl,
                    "mini_batch_size": self.args.mini_batch_size,
                    "normalize_advantages": self.args.normalize_advantages,
                })
        
        # For RL agents, try to merge with existing config.json if it exists
        config_file = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_file) and self.args.agent in ["sarsa", "a2c", "ppo"]:
            try:
                with open(config_file, 'r') as f:
                    existing_config = json.load(f)
                # Merge existing config with new config (new config takes precedence)
                existing_config.update(config)
                config = existing_config
                logging.info(f"Merged with existing config from {config_file}")
            except Exception as e:
                logging.warning(f"Failed to merge with existing config: {e}")
        
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Saved configuration to {config_file}")
    
    def save_evaluation_results(self, rewards_history, trajectories, env_objectives):
        """save evaluation results to model directory"""
        # save evaluation rewards history
        rewards_file = os.path.join(self.model_dir, "evaluation_rewards_history.npy")
        np.save(rewards_file, rewards_history)
        logging.info(f"Saved evaluation rewards history to {rewards_file}")
        
        # save evaluation trajectories (trajectories during evaluation)
        trajectories_file = os.path.join(self.model_dir, "evaluation_trajectories.json")
        with open(trajectories_file, 'w') as f:
            json.dump(trajectories, f, indent=2, default=convert_numpy_types)
        logging.info(f"Saved evaluation trajectories to {trajectories_file}")
        
        # Calculate and save policy-level profile statistics
        self._save_policy_level_profile(rewards_history, trajectories, env_objectives)
        
        # plot and save evaluation rewards plot
        self._plot_evaluation_rewards(rewards_history, env_objectives)

        # plot and save mastery progression plot
        self._plot_mastery_progression(trajectories, self.args.target_skill_bundle, self.args.train_steps, self.args.mastery_threshold)
    
    def _save_policy_level_profile(self, rewards_history, trajectories, env_objectives):
        """Calculate and save policy-level profile statistics"""
        # Handle dictionary format rewards_history
        
        # Calculate scalarized rewards (simple average of all objectives)
        scalar_rewards = []
        for episode_rewards in rewards_history:
            episode_scalar = np.mean(list(episode_rewards.values()))
            scalar_rewards.append(episode_scalar)
        
        avg_scalar_reward = np.mean(scalar_rewards)
        std_scalar_reward = np.std(scalar_rewards)
        
        # Calculate average reward for each objective
        objective_avg_rewards = {}
        for objective in env_objectives:
            objective_values = [episode_rewards[objective] for episode_rewards in rewards_history]
            objective_avg_rewards[objective] = float(np.mean(objective_values))
        
        # Calculate average steps per episode
        steps_per_episode = [len(trajectory) for trajectory in trajectories]
        avg_steps_per_episode = np.mean(steps_per_episode)
        
        # Create policy profile dictionary with only statistics
        policy_profile = {
            "avg_scalar_reward": float(avg_scalar_reward),
            "std_scalar_reward": float(std_scalar_reward),
            "objective_avg_rewards": objective_avg_rewards,
            "avg_steps_per_episode": float(avg_steps_per_episode),
            "objectives": env_objectives
        }
        
        # Save policy profile to JSON file
        profile_file = os.path.join(self.model_dir, "policy_level_profile.json")
        with open(profile_file, 'w') as f:
            json.dump(policy_profile, f, indent=2)
        logging.info(f"Saved policy level profile to {profile_file}")
        
        # Log the statistics
        logging.info(f"\n=== Policy Level Profile ===")
        logging.info(f"Average Scalar Reward: {avg_scalar_reward:.4f}")
        logging.info(f"Standard Deviation Scalar Reward: {std_scalar_reward:.4f}")
        logging.info(f"Average Steps per Episode: {avg_steps_per_episode:.2f}")
        logging.info(f"Objective Average Rewards:")
        for objective, avg_reward in objective_avg_rewards.items():
            logging.info(f"  {objective}: {avg_reward:.4f}")
    
    def _plot_evaluation_rewards(self, rewards_history, objectives):
        """plot evaluation rewards plot and save to model directory"""
        # Handle dictionary format rewards_history
        
        num_episodes = len(rewards_history)
        num_objectives = len(objectives)

        plt.figure(figsize=(12, 8))

        for objective in objectives:
            objective_values = [episode_rewards[objective] for episode_rewards in rewards_history]
            plt.plot(range(1, num_episodes + 1), objective_values, label=objective)

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"{self.args.agent.upper()} Evaluation Rewards Performance")
        plt.legend()
        plt.grid(True)

        # save to model directory
        output_file = os.path.join(self.model_dir, "evaluation_reward_performance.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved evaluation reward plot to {output_file}")
    
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
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot mastery progression for each skill
        colors = plt.cm.Set3(np.linspace(0, 1, len(target_skills)))
        for i, skill in enumerate(target_skills):
            mastery_values = avg_mastery_by_step[skill]
            # Remove NaN values for plotting
            valid_indices = [j for j, val in enumerate(mastery_values) if not np.isnan(val)]
            if valid_indices:
                steps = [j + 1 for j in valid_indices]  # Step numbers start from 1
                values = [mastery_values[j] for j in valid_indices]
                plt.plot(steps, values, label=f"{skill} ({self.args.agent})", 
                        color=colors[i], marker='o', markersize=4, linewidth=2)
        
        # Add horizontal dashed line for mastery threshold
        plt.axhline(y=early_stop_threshold, color='red', linestyle='--', 
                    alpha=0.7, linewidth=2, label=f'Mastery Threshold ({early_stop_threshold})')
        
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
                    plt.plot(star_step, star_mastery, 'k*', markersize=15, 
                            label=f'Average Episode Length ({avg_episode_length:.1f} steps)')
        
        # Customize the plot
        plt.xlabel("Step Number", fontsize=12)
        plt.ylabel("Average Mastery Level", fontsize=12)
        plt.title(f"Mastery Progression Over Time - {self.args.agent.upper()}", fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max_steps)
        plt.ylim(-0.05, 1.05)
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        
        # Save to model directory
        output_file = os.path.join(self.model_dir, "evaluation_mastery_progression.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved mastery progression plot to {output_file}")
    
    def get_model_dir(self):
        """get model directory path"""
        return self.model_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Objective Sequential Decision Making"
    )

    # Core parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--config_path", type=str, default="configs/base.ini", help="Path to config file")
    
    # Agent parameters
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["simple_llm", "sarsa", "a2c", "ppo"],
        help="Agent type: Simple LLM, SARSA, A2C, PPO",
    )
    parser.add_argument(
        "--target_skill_bundle",
        nargs="+",
        default=["Fundamental Mathematics"],
        help="List of target skills to focus on during learning",
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default=None,
        help="Path to existing trajectory file to continue from",
    )

    # Reward and objectives
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="scalarized",
        choices=["scalarized", "reward_machine", "pareto_buffer_rm"],
        help="Reward processing mode for RL agents (sarsa: scalarized/reward_machine, ppo/a2c: scalarized/pareto_buffer_rm)",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=[
            "aptitude",
            "gap",
            "performance",
        ],
        help="List of objectives to optimize",
    )

    # Training parameters
    parser.add_argument(
        "--train_episodes",
        type=int,
        default=10000,
        help="Number of training episodes for RL agents",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=50,
        help="Maximum steps per training episode",
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes", type=int, default=30, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Steps per evaluation episode"
    )

    # LLM agent parameters (only needed for simple_llm)
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama-3-8b",  # llama-3-8b, mistral-24b-instruct
        help="LLM model name (for LLM agent)",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="few_shot_cot",
        choices=["zero_shot", "few_shot", "few_shot_cot"],
        help="Type of prompt template to use for LLM agent",
    )

    # Benchmark parameters
    parser.add_argument(
        "--benchmark",
        type=str,
        default="math_bench",
        choices=["medmcqa", "math_bench"],
        help="Benchmark dataset to use",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="QA",
        choices=["QA"],
        help="Data type to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to load from benchmark (for testing)",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default="data/benchmark",
        help="Path to benchmark data directory",
    )

    # Response model parameters
    parser.add_argument(
        "--response_model",
        type=str,
        default="irt",
        choices=["irt", "bkt"],
        help="Response model to use (irt uses GDIRT model)",
    )
    parser.add_argument(
        "--mastery_threshold",
        type=float,
        default=0.8,
        help="Threshold for transitioning from performance/gap to aptitude stage (default: 0.8) for reward machine",
    )

    # RL Training parameters
    parser.add_argument(
        "--only_evaluate",
        action="store_true",
        # default=True,
        help="Only evaluate the policy, do not train it from scratch.",
    )
    parser.add_argument(
        "--RL_trained_policy",
        type=str,
        # default="Policy_Set_Results_IRT_005v4/3O_ppo_pareto_buffer_rm_gap_performance_aptitude_irt_2025-09-17_17-52-57_model",
        help="Path to the pre-trained RL policy file to load.",
    )

    return parser.parse_args()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load base configuration for RL agents
    if args.agent in ["sarsa", "a2c", "ppo"]:
        # Validate agent and reward_mode combination
        if not validate_agent_reward_combination(args.agent, args.reward_mode):
            supported_combinations = {
                "sarsa": ["scalarized", "reward_machine"],
                "ppo": ["scalarized", "pareto_buffer_rm"],
                "a2c": ["scalarized", "pareto_buffer_rm"]
            }
            print(f"Unsupported combination: {args.agent} + {args.reward_mode}")
            print(f"Supported combinations for {args.agent}: {supported_combinations.get(args.agent, [])}")
            exit(1)
        
        base_config = load_base_config(args.agent, args.config_path, args.reward_mode, args.objectives)
        apply_base_config(args, base_config)
        print(f"Loaded base configuration from {args.config_path}: {base_config}")        

    # Create ModelManager early to get model directory for logging
    model_manager = ModelManager(args)
    
    # Configure logging to save to model directory
    log_file = os.path.join(model_manager.get_model_dir(), "1_training.log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info(f"Arguments: {args}")
    logging.info(f"Model directory: {model_manager.get_model_dir()}")
    logging.info(f"Log file path: {log_file}")
    
    # Verify log file is created
    if os.path.exists(log_file):
        print(f"Log file created successfully: {log_file}")
    else:
        print(f"Log file not created: {log_file}")

    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load benchmark data
    questions, question_skill_map, unique_skills = load_benchmark_data(args)

    # Rescale difficulties and save/load from CSV
    difficulties = rescale_difficulties(questions, args)

    if args.response_model == "irt":
        response_model = IRTModel(len(unique_skills), seed=args.seed)
        # Set question parameters for IRT model using rescaled difficulties
        # Inspect difficulties structure
        print("Type of difficulties:", type(difficulties))
        print("Sample difficulty keys:", list(difficulties.keys())[:5])

        # Align IRT item indices with the order we will register questions in the env
        question_ids_in_order = []
        for q in questions:
            qid_raw = q.get("id")
            if qid_raw is None:
                raise ValueError("Each question must have an 'id' field when using ID-based difficulty mapping.")
            question_ids_in_order.append(int(qid_raw))

        # Map dense IRT item index -> scaled_difficulty (env will assign the same dense indices in this order)
        question_difficulties = {i: float(difficulties[qid]["scaled_difficulty"]) for i, qid in enumerate(question_ids_in_order)}
        response_model.set_question_params_batch(question_difficulties)
    elif args.response_model == "bkt":
        response_model = BKTModel(len(unique_skills), seed=args.seed)

    # Create the environment
    env = EducationEnv(
        skills=args.target_skill_bundle,
        num_questions=len(questions),
        objectives=args.objectives,
        response_model=response_model,
        target_skill_bundle=args.target_skill_bundle,
        max_steps=args.train_steps,
        seed=args.seed,
        early_stop_threshold=args.mastery_threshold,
        question_recommender=None,  # Can be set to actual QuestionRecommender if needed
    )

    # Set difficulty levels from question bank
    env.set_difficulty_levels(difficulties)

    # Set question skills mapping in environment (use dataset question IDs, not 0..N-1 indices)
    # question_skill_map is often keyed by list index; we translate it to real question ids from `questions`.
    for i, q in enumerate(questions):
        qid_raw = q.get("id", i)
        try:
            qid = int(qid_raw)
        except Exception:
            qid = int(i)

        if i in question_skill_map:
            skills = question_skill_map[i]["skills"]
        elif str(i) in question_skill_map:
            skills = question_skill_map[str(i)]["skills"]
        elif qid in question_skill_map:
            skills = question_skill_map[qid]["skills"]
        elif str(qid) in question_skill_map:
            skills = question_skill_map[str(qid)]["skills"]
        else:
            raise KeyError(f"Could not find skills for question index {i} (qid={qid}) in question_skill_map")

        env.set_question_skills_difficulty(qid, skills)

    # Set up reward handler for RL agents
    reward_handler = None
    if args.agent in ["sarsa", "a2c", "ppo"]:
        if args.reward_mode in ["pareto_buffer_s", "pareto_buffer_rm"]:
            # Set buffer size based on agent's episodes per update
            if args.agent == "ppo":
                buffer_size = args.ppo_n_steps 
            elif args.agent == "a2c":
                buffer_size = args.a2c_n_steps
            else:
                raise ValueError(f"Invalid agent: {args.agent} for pareto buffer")
            
        if args.reward_mode == "scalarized":
            weights = [1.0 / len(args.objectives)] * len(args.objectives)
            reward_handler = ScalarizedRewardHandler(weights, args.objectives)
        elif args.reward_mode == "reward_machine":
            # Instantiate the reward machine
            edu_reward_machine = EduRewardMachine(
                objectives=args.objectives,
            )
            reward_handler = RewardMachineHandler(edu_reward_machine)
        elif args.reward_mode == "pareto_buffer_s":
            weights = [1.0 / len(args.objectives)] * len(args.objectives)
            scalarized_handler = ScalarizedRewardHandler(weights, args.objectives)
            
            reward_handler = ParetoBufferHandler(
                buffer_size=buffer_size, base_handler=scalarized_handler
            )
        elif args.reward_mode == "pareto_buffer_rm":
            # Pareto buffer with reward machine
            edu_reward_machine = EduRewardMachine(
                objectives=args.objectives,
            )
            rm_handler = RewardMachineHandler(edu_reward_machine)
            
            
            reward_handler = ParetoBufferHandler(
                buffer_size=buffer_size, base_handler=rm_handler
            )

    # RL agent creation and training
    if args.agent == "simple_llm":
        agent = create_agent(
            args,
            len(args.objectives),
        )
    elif args.agent in ["sarsa", "a2c", "ppo"]:
        agent = create_agent(args, len(args.objectives), env=env)
        
        # Check if we should load pre-trained policy or train new one
        if args.only_evaluate and hasattr(args, 'RL_trained_policy') and args.RL_trained_policy:
            logging.info(f"Loading pre-trained {args.agent} policy from {args.RL_trained_policy}")
            # Load the pre-trained policy
            agent.load(args.RL_trained_policy)
            logging.info(f"Successfully loaded pre-trained {args.agent} policy")
        else:
            # Train new policy
            logging.info(f"Training new {args.agent} policy")
            trainer = RewardHandlerRLTrainer(agent, env, args, reward_handler, model_dir=model_manager.get_model_dir())
            trainer.train()
    
    else:
        logging.info(f"Agent {args.agent} not found")
        exit(1)
    
    if not args.only_evaluate:
        model_manager.save_config()

    # Evaluate policy for all episodes and steps
    logging.info(f"********* Starting Evaluation *********")
    rewards_history, trajectories = evaluate_policy(
        env, agent, questions, args
    )

    # Analyze trajectories
    logging.info("\n=== Trajectory Analysis ===")
    for episode_idx, trajectory in enumerate(trajectories):
        # Calculate statistics for this episode
        total_questions = len(trajectory) * 5  # Each step now processes 5 questions
        total_actions = len(trajectory)
        
        # Calculate average mastery for target skills
        target_skill_mastery = []
        for step in trajectory:
            if step["state"]["mastery"]:
                target_mastery = [
                    step["state"]["mastery"][skill]
                    for skill in args.target_skill_bundle
                    if skill in step["state"]["mastery"]
                ]
                if target_mastery:
                    target_skill_mastery.append(np.mean(target_mastery))

        avg_target_mastery = (
            np.mean(target_skill_mastery) if target_skill_mastery else 0
        )

        logging.info(f"\nEpisode {episode_idx + 1}:")
        logging.info(f"  Total Actions: {total_actions}")
        logging.info(f"  Total Questions Processed: {total_questions}")
        logging.info(f"  Average Target Skill Mastery: {avg_target_mastery:.2f}")

    # Save evaluation results using ModelManager
    model_manager.save_evaluation_results(rewards_history, trajectories, env.objectives)

    # Final log message and verification
    logging.info("Training and evaluation completed successfully!")
    logging.info(f"All results saved to: {model_manager.get_model_dir()}")


if __name__ == "__main__":
    main()
