import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.response_models import (
    BaseResponseModel,
    BKTModel,
    IRTModel,
)
from utils.question_recommender import QuestionRecommender


class EducationEnv(gym.Env):
    """
    Educational environment for multi-objective sequential decision making.

    MDP Definition:
    - State: [mastery for each skill, each skill x difficulty level's global accuracy, remaining failed question ratio]
    - Action: 3 action types (0: Failed Questions, 1: Easy Questions, 2: High-Aptitude Questions)
    - Transition: Each action selects self.questions_per_step questions, responses determined by simulate_response and state updates
    - Reward: Calculated by reward evaluator based on multiple objectives (performance, gap, aptitude)
    """

    def __init__(
        self,
        skills: List[str] = None,
        num_questions: int = 20,
        max_history_length: int = 5,
        objectives: List[str] = ["performance", "gap", "aptitude"],
        early_stop_threshold: float = 0.8,
        response_model: BaseResponseModel = None,
        target_skill_bundle: Optional[List[str]] = None,
        max_steps: int = 30,
        seed: Optional[int] = None,
        question_recommender: QuestionRecommender = None,
        ncc_window: int = 2,
    ):
        super().__init__()

        # Environment parameters
        self.skills = skills
        self.num_skills = len(self.skills)
        self.num_questions = num_questions
        self.max_history_length = max_history_length
        self.objectives = objectives
        self.num_objectives = len(objectives)
        self.early_stop_threshold = early_stop_threshold
        self.response_model = response_model
        self.target_skill_bundle = (
            target_skill_bundle if target_skill_bundle else self.skills
        )
        self.max_steps = max_steps
        self.question_recommender = question_recommender
        self.ncc_window = ncc_window  # Window size for NCC condition

        # Global list of all failed questions (question_id -> (skills, original_difficulty, scaled_difficulty))
        self.all_failed_questions = dict()
        
        # Global list of all cleared questions (question_id -> (skills, original_difficulty, scaled_difficulty))
        self.cleared_questions = dict()
        
        # NCC tracking for each skill: {skill_idx: {difficulty: [correctness_history]}}
        self.ncc_tracking = {}
        
        # Experience tracking: store seen materials with (question_id, difficulty, correct)
        self.seen_materials = []  # List of (question_id, difficulty, correct)
        
        # Track failed tests as a skill x difficulty matrix
        # Difficulty levels will be set dynamically from question bank
        self.difficulty_levels = 0  # Will be set later
        self.failed_questions_ratio = None  # Will be initialized after setting difficulty_levels
        self.skill_difficulty_accuracy = None  # Will be initialized after setting difficulty_levels
        self.skill_difficulty_counts = None  # Will be initialized after setting difficulty_levels

        # Action space: 3 action types (0: Failed, 1: Easy, 2: High-Aptitude)
        self.action_space = spaces.Discrete(3)
        
        # Number of questions to select per action
        self.questions_per_step = 5

        # Observation space components
        mastery_dim = self.num_skills  # [0,1] for each skill
        skill_difficulty_accuracy_dim = self.num_skills * self.difficulty_levels  # skill x difficulty accuracy matrix
        failed_questions_ratio_dim = self.num_skills * self.difficulty_levels  # skill x difficulty failed ratio matrix

        # Total observation dimension: #skills + 2*(#skills × #difficulty_levels)
        obs_dim = (
            mastery_dim
            + skill_difficulty_accuracy_dim
            + failed_questions_ratio_dim
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=0,  # All values are non-negative
            high=1,  # All values are bounded by 1
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Question-skill mapping
        self.skill_to_idx = {skill: i for i, skill in enumerate(self.skills)}
        self.question_skills_difficulty_map = {}  # Will store skills and difficulty for each question
        
        # Pre-computed question lists for different strategies
        self.aptitude_cache = {}  # Cache for aptitude calculations
        self.experience_cache = {}  # Cache for experience calculations
        self.gap_cache = {}  # Cache for gap calculations

        # Local random generator for environment
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize state
        # self.reset()

    def warmup_fixed_questions(self):
        """
        For single-skill case, present self.questions_per_step questions with difficulties below current mastery
        to initialize the environment state. All answers are set to correct.
        No mastery or failed question updates are performed during warmup.
        """
        if len(self.skills) != 1:
            raise ValueError("Warmup only supported for single-skill case")
        skill = self.skills[0]
        skill_idx = self.skill_to_idx[skill]
        current_mastery = self.mastery[skill_idx]

        # Select warmup questions: ceil(questions_per_step/2) below mastery, 
        # floor(questions_per_step/2) above mastery but below min(mastery + 0.2, 1)
        warmup_qs = []
        
        # Calculate thresholds
        upper_threshold = min(current_mastery + 0.2, 1.0)
        
        # Get questions below mastery
        below_mastery_qs = [qid for qid, info in self.question_skills_difficulty_map.items() 
                           if skill in info["skills"] and info["scaled_difficulty"] < current_mastery]
        
        # Get questions above mastery but below upper threshold
        above_mastery_qs = [qid for qid, info in self.question_skills_difficulty_map.items() 
                           if skill in info["skills"] and 
                           current_mastery <= info["scaled_difficulty"] < upper_threshold]
        
        # Calculate number of questions to select from each category
        num_below = math.ceil(self.questions_per_step / 2)
        num_above = math.floor(self.questions_per_step / 2)
        
        # Select from below mastery questions
        if below_mastery_qs and num_below > 0:
            num_below_select = min(num_below, len(below_mastery_qs))
            selected_below = self.rng.choice(below_mastery_qs, size=num_below_select, replace=False)
            warmup_qs.extend(selected_below)
        
        # Select from above mastery questions
        if above_mastery_qs and num_above > 0:
            num_above_select = min(num_above, len(above_mastery_qs))
            selected_above = self.rng.choice(above_mastery_qs, size=num_above_select, replace=False)
            warmup_qs.extend(selected_above)
        
        # If we still need more questions, fill with any available questions
        remaining_needed = self.questions_per_step - len(warmup_qs)
        if remaining_needed > 0:
            all_qs = [qid for qid, info in self.question_skills_difficulty_map.items() 
                     if skill in info["skills"] and qid not in warmup_qs]
            if all_qs:
                num_additional = min(remaining_needed, len(all_qs))
                selected_additional = self.rng.choice(all_qs, size=num_additional, replace=False)
                warmup_qs.extend(selected_additional)
        
        # If still no questions, raise error
        if not warmup_qs:
            raise ValueError("No questions available for warmup")
        
        # Process warmup questions - below mastery = correct, above mastery = incorrect
        for i, qid in enumerate(warmup_qs):
            info = self.question_skills_difficulty_map[qid]
            skills_tested = np.array([skill_idx])
            
            # Determine correctness based on whether question is below or above mastery
            if i < num_below:
                # Questions from below_mastery_qs are correct
                correct = True
            else:
                # Questions from above_mastery_qs are incorrect
                correct = False
            
            # Record seen material for experience tracking
            self.seen_materials.append((qid, info["scaled_difficulty"], correct))
            
            # Update skill-difficulty accuracy
            self._update_skill_difficulty_accuracy(qid, correct, skills_tested, info["original_difficulty"])
            
            # Update failed_question for incorrect answers
            if not correct:
                self._update_failed_question(qid, skills_tested, correct, info["original_difficulty"], info["scaled_difficulty"])
        
        # No training needed for IRT model - abilities are updated during warmup
            
        logging.info(f"warmup questions: {warmup_qs} ({num_below} correct below mastery, {len(warmup_qs) - num_below} incorrect above mastery)")
        
        # Update failed_questions_ratio after warmup
        self._update_failed_questions_ratio()
        
        # Update aptitude, experience, and gap cache after warmup
        self._update_aptitude_cache()
        self._update_experience_cache()
        self._update_gap_cache()

    def reset(self):
        """Reset the environment to initial state."""
        # Initialize state components
        self.mastery = 0.4 * np.ones(self.num_skills)
        self.current_step = 0

        self.all_failed_questions = dict()
        self.cleared_questions = dict()
        
        # Initialize NCC tracking for each skill
        self.ncc_tracking = {i: {} for i in range(self.num_skills)}
        
        # Initialize experience tracking
        self.seen_materials = [] 
        
        # Initialize caches
        self.aptitude_cache = {}
        self.experience_cache = {}
        self.gap_cache = {}
        
        # Only initialize matrices if difficulty_levels has been set
        if hasattr(self, 'difficulty_levels') and self.difficulty_levels > 0:
            self.failed_questions_ratio = np.zeros((self.num_skills, self.difficulty_levels), dtype=np.float32)
            self.skill_difficulty_accuracy = np.zeros((self.num_skills, self.difficulty_levels), dtype=np.float32)
            self.skill_difficulty_counts = np.zeros((self.num_skills, self.difficulty_levels), dtype=np.uint8)  # Use uint8 for counts
        else:
            raise ValueError("Difficulty levels must be set before reset. Call set_difficulty_levels() first.")
        
        logging.info(f"INITIAL MASTERIES: {self.mastery}")

        # Reset response model state
        if self.response_model is not None:
            self.response_model.reset(self.mastery)

        if not self.question_skills_difficulty_map:
            raise ValueError("Question skill mapping should be initialized")

        # warmup
        self.warmup_fixed_questions()
        logging.info(f"WARMUP COMPLETED!")
        logging.info(f"mastery: {self.mastery}")
        
        # # Sync failed_questions_ratio matrix with failed_question dict
        # self._update_failed_questions_ratio()

        # Return observation and info as required by Gymnasium (cache tolist() results)
        failed_questions_ratio_list = self.failed_questions_ratio.tolist()
        skill_difficulty_accuracy_list = self.skill_difficulty_accuracy.tolist()
        skill_difficulty_counts_list = self.skill_difficulty_counts.tolist()
        
        info = {
            "current_step": self.current_step,
            "mastery": {
                self.skills[i]: round(float(self.mastery[i]), 2)
                for i in range(self.num_skills)
            },
            "all_failed_questions": self.all_failed_questions,
            "valid_failed_questions": self._get_valid_failed_questions(),
            "cleared_questions": self.cleared_questions,
            "failed_questions_ratio": failed_questions_ratio_list,
            "skill_difficulty_accuracy": skill_difficulty_accuracy_list,
            "skill_difficulty_counts": skill_difficulty_counts_list,
        }

        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        """Construct the observation array from current state components."""
        return np.concatenate(
            [
                self.mastery,  # Mastery for each skill
                self.skill_difficulty_accuracy.flatten(),  # Flatten the skill x difficulty accuracy matrix
                self.failed_questions_ratio.flatten(),  # Flatten the skill x difficulty failed ratio matrix
            ]
        ).astype(np.float32)

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """
        Take a step in the environment by selecting an action type that recommends self.questions_per_step questions.

        Args:
            action: Action type (0: Failed Questions, 1: Easy Questions, 2: High-Aptitude Questions)

        Returns:
            observation: (mastery_dict, skill_difficulty_accuracy, failed_questions_ratio)
            reward: Vector of rewards for each objective
            truncated: Whether the episode was truncated (e.g., max steps reached)
            info: Additional information
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Select questions based on action type
        selected_question_ids = self._select_questions_by_action(action)
        
        # Process all selected questions
        all_question_info = []
        
        # Process each selected question
        for question_id in selected_question_ids:
            question_info = self.question_skills_difficulty_map[question_id]
            recommended_skills = question_info["skills"]
            original_difficulty = question_info["original_difficulty"]
            scaled_difficulty = question_info["scaled_difficulty"]
            
            # Convert skill names to indices
            skills_tested = []
            for skill_name in recommended_skills:
                if skill_name in self.skill_to_idx:
                    skills_tested.append(self.skill_to_idx[skill_name])
            skills_tested = np.array(skills_tested)
            
            # Simulate student's response using response model
            correct, prob = self._simulate_response(question_id, skills_tested)
            
            # Record seen material for experience tracking
            self.seen_materials.append((question_id, scaled_difficulty, correct))
            
            # Store information for batch processing
            all_question_info.append({
                "question_id": question_id,
                "skills": recommended_skills,
                "original_difficulty": original_difficulty,
                "scaled_difficulty": scaled_difficulty,
                "correct": correct,
                "skills_tested": skills_tested
            })
            # Update state components (except failed_question, which needs to be updated after reward calculation)
            self._update_skill_difficulty_accuracy(question_id, correct, skills_tested, original_difficulty)
        
        # Calculate rewards based on all questions in the batch
        rewards_dict = self._calculate_batch_rewards(all_question_info)
        
        # Update mastery with all question info
        self._update_mastery(all_question_info)
        if isinstance(self.response_model, IRTModel):
            self.response_model.reset(self.mastery)
        
        # Calculate rolling accuracy for this step (percentage of correct answers)
        correct_count = sum(1 for q_info in all_question_info if q_info["correct"])
        if len(all_question_info) == 0:
            rolling_accuracy = np.nan
        else:
            rolling_accuracy = correct_count / len(all_question_info)
        
        # Now update failed_question state after reward calculation
        for question_info in all_question_info:
            question_id = question_info["question_id"]
            skills_tested = question_info["skills_tested"]  # Use the already computed skills_tested
            correct = question_info["correct"]
            original_difficulty = question_info["original_difficulty"]
            scaled_difficulty = question_info["scaled_difficulty"]
            self._update_failed_question(question_id, skills_tested, correct, original_difficulty, scaled_difficulty)

        logging.info(f">>> all failed questions: {len(self.all_failed_questions)}, {self.all_failed_questions}")
        logging.info(f">>> valid failed questions: {len(self._get_valid_failed_questions())}, {self._get_valid_failed_questions()}")
        logging.info(f">>> cleared questions: {len(self.cleared_questions)}, {self.cleared_questions}")

        self._update_aptitude_cache()
        self._update_experience_cache()
        self._update_gap_cache()


        # Update step count and check if truncated
        avg_mastery = np.mean(self.mastery)
        self.current_step += 1
        if self.current_step >= self.max_steps or avg_mastery >= 1:
            truncated = True
        else:
            truncated = False

        # Prepare info dict (cache tolist() results to avoid repeated calls)
        failed_questions_ratio_list = self.failed_questions_ratio.tolist()
        skill_difficulty_accuracy_list = self.skill_difficulty_accuracy.tolist()
        skill_difficulty_counts_list = self.skill_difficulty_counts.tolist()
        
        info = {
            "action": action,
            "selected_questions": selected_question_ids,
            "questions_info": all_question_info,
            "rolling_accuracy": rolling_accuracy,
            "mastery": {
                self.skills[i]: round(self.mastery[i], 2)
                for i in range(self.num_skills)
            },
            "all_failed_questions": self.all_failed_questions,
            "valid_failed_questions": self._get_valid_failed_questions(),
            "cleared_questions": self.cleared_questions,
            "failed_questions_ratio": failed_questions_ratio_list,
            "skill_difficulty_accuracy": skill_difficulty_accuracy_list,
            "skill_difficulty_counts": skill_difficulty_counts_list,
        }

        return self._get_obs(), rewards_dict, truncated, info

    def predict_step(
        self,
        action: int,
        rollout_steps: int = 1,
        policy=None,
    ) -> Dict[str, Any]:
        """
        predict step is used for policy evaluation, it does not update the environment state
        
        Args:
            action: Action type (0: Failed Questions, 1: Easy Questions, 2: High-Aptitude Questions)
            rollout_steps: Number of steps to rollout (default: 1)
            policy: Policy to use for subsequent steps (if None, uses same action for all steps)
            
        Returns:
            dict: containing rollout_steps_info, final_state, final_info, original_mastery
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        assert rollout_steps > 0, f"rollout_steps must be positive, got {rollout_steps}"
        
        # Keep original state for restoring after prediction
        original_state = {
            "mastery": self.mastery.copy(),
            "all_failed_questions": self.all_failed_questions.copy(),
            "cleared_questions": self.cleared_questions.copy(),
            "failed_questions_ratio": self.failed_questions_ratio.copy(),
            "skill_difficulty_accuracy": self.skill_difficulty_accuracy.copy(),
            "skill_difficulty_counts": self.skill_difficulty_counts.copy(),
            "seen_materials": self.seen_materials.copy(),
            "current_step": self.current_step,
            "aptitude_cache": self.aptitude_cache.copy(),
            "experience_cache": self.experience_cache.copy(),
            "gap_cache": self.gap_cache.copy(),
            "ncc_tracking": {k: v.copy() for k, v in self.ncc_tracking.items()}  # Deep copy ncc_tracking
        }
        
        try:
            # Store rollout information
            rollout_steps_info = []
            # cumulative_reward = 0.0
            current_action = action
            
            # Perform multi-step rollout
            for step in range(rollout_steps):
                # Select questions based on current action type
                selected_question_ids = self._select_questions_by_action(current_action)
                
                # Process all selected questions
                all_question_info = []
                
                # Process each selected question
                for question_id in selected_question_ids:
                    question_info = self.question_skills_difficulty_map[question_id]
                    recommended_skills = question_info["skills"]
                    original_difficulty = question_info["original_difficulty"]
                    scaled_difficulty = question_info["scaled_difficulty"]
                    
                    # Convert skill names to indices
                    skills_tested = []
                    for skill_name in recommended_skills:
                        if skill_name in self.skill_to_idx:
                            skills_tested.append(self.skill_to_idx[skill_name])
                    skills_tested = np.array(skills_tested)
                    
                    # Simulate student's response using response model
                    correct, prob = self._simulate_response(question_id, skills_tested)
                    
                    # Record seen material for experience tracking 
                    self.seen_materials.append((question_id, scaled_difficulty, correct))
                    
                    # Store information for batch processing
                    all_question_info.append({
                        "question_id": question_id,
                        "skills": recommended_skills,
                        "original_difficulty": original_difficulty,
                        "scaled_difficulty": scaled_difficulty,
                        "correct": correct,
                        "skills_tested": skills_tested
                    })
                    
                    # Update state components 
                    self._update_skill_difficulty_accuracy(question_id, correct, skills_tested, original_difficulty)
                
                # Calculate rewards based on all questions in the batch
                rewards_dict = self._calculate_batch_rewards(all_question_info)
                
                # Update mastery with all question info 
                self._update_mastery(all_question_info)
                if isinstance(self.response_model, IRTModel):
                    self.response_model.reset(self.mastery)
                
                # Calculate rolling accuracy for this step (percentage of correct answers)
                correct_count = sum(1 for q_info in all_question_info if q_info["correct"])
                if len(all_question_info) == 0:
                    rolling_accuracy = np.nan
                else:
                    rolling_accuracy = correct_count / len(all_question_info)
                
                # Now update failed_question state after reward calculation
                for question_info in all_question_info:
                    question_id = question_info["question_id"]
                    skills_tested = question_info["skills_tested"]
                    correct = question_info["correct"]
                    original_difficulty = question_info["original_difficulty"]
                    scaled_difficulty = question_info["scaled_difficulty"]
                    self._update_failed_question(question_id, skills_tested, correct, original_difficulty, scaled_difficulty)
                
                self._update_aptitude_cache()
                self._update_experience_cache()
                self._update_gap_cache()
                
                # Calculate step reward
                # step_reward = sum(rewards_dict.values())
                # cumulative_reward += step_reward
                
                # Prepare step info dict (cache tolist() results)
                failed_questions_ratio_list = self.failed_questions_ratio.tolist()
                skill_difficulty_accuracy_list = self.skill_difficulty_accuracy.tolist()
                skill_difficulty_counts_list = self.skill_difficulty_counts.tolist()
                
                step_info = {
                    "step": step + 1,
                    "action": current_action,
                    "selected_questions": selected_question_ids,
                    "questions_info": all_question_info,
                    # "rolling_accuracy": rolling_accuracy,
                    "rewards_dict": rewards_dict,  # Match step function naming
                    # "step_reward": step_reward,  # Keep for predict_step specific info
                    "mastery": {
                        self.skills[i]: round(self.mastery[i], 2)
                        for i in range(self.num_skills)
                    },
                    "valid_failed_questions": self._get_valid_failed_questions(),
                    "cleared_questions": self.cleared_questions.copy(),  # Copy to avoid reference issues
                    "all_failed_questions": self.all_failed_questions.copy(),
                    "failed_questions_ratio": failed_questions_ratio_list,
                    "skill_difficulty_accuracy": skill_difficulty_accuracy_list,
                    "skill_difficulty_counts": skill_difficulty_counts_list,
                }
                
                rollout_steps_info.append(step_info)
                
                # For next step, use policy if provided, otherwise keep the same action
                if policy is None:
                    raise ValueError("policy must be provided for subsequent steps")
                elif step < rollout_steps - 1:  # Don't call policy on last step
                    # Get next state for policy decision
                    next_state = self._get_obs()
                    # Call policy to get next action
                    action_info = policy(next_state, available_actions=None)
                    current_action = action_info["action"]
            
            # Get final state after all rollout steps
            final_state = self._get_obs()
            
            # Prepare final info dict
            # final_failed_questions_ratio_list = self.failed_questions_ratio.tolist()
            # final_skill_difficulty_accuracy_list = self.skill_difficulty_accuracy.tolist()
            # final_skill_difficulty_counts_list = self.skill_difficulty_counts.tolist()
            
            final_info = {
                # "rollout_steps": rollout_steps,
                # "cumulative_reward": cumulative_reward,
                # "average_reward": cumulative_reward / rollout_steps,
                "final_mastery": {
                    self.skills[i]: round(self.mastery[i], 2)
                    for i in range(self.num_skills)
                },
                # "final_all_failed_questions": self.all_failed_questions,
                # "final_failed_questions_ratio": final_failed_questions_ratio_list,
                # "final_skill_difficulty_accuracy": final_skill_difficulty_accuracy_list,
                # "final_skill_difficulty_counts": final_skill_difficulty_counts_list,
            }

            original_info = {
                "mastery": original_state["mastery"],
                "all_failed_questions": original_state["all_failed_questions"],
                "cleared_questions": original_state["cleared_questions"],
            }
            
            result = {
                "rollout_steps_info": rollout_steps_info,
                # "cumulative_reward": cumulative_reward,
                "final_state": final_state,
                "final_info": final_info,
                "original_info": original_info
            }
            
            return result
            
        finally:
            # Restore original state
            self.mastery = original_state["mastery"]
            self.all_failed_questions = original_state["all_failed_questions"]
            self.cleared_questions = original_state["cleared_questions"]
            self.failed_questions_ratio = original_state["failed_questions_ratio"]
            self.skill_difficulty_accuracy = original_state["skill_difficulty_accuracy"]
            self.skill_difficulty_counts = original_state["skill_difficulty_counts"]
            self.seen_materials = original_state["seen_materials"]
            self.current_step = original_state["current_step"]
            self.aptitude_cache = original_state["aptitude_cache"]
            self.experience_cache = original_state["experience_cache"]
            self.gap_cache = original_state["gap_cache"]
            self.ncc_tracking = original_state["ncc_tracking"]
            
            # Reset response model to original state
            if self.response_model is not None:
                self.response_model.reset(self.mastery)

    def _update_skill_difficulty_accuracy(
        self, action: int, correct: bool, skills_tested: list, original_difficulty: float
    ):
        """Update skill-difficulty accuracy matrices."""
        # Update skill-difficulty accuracy matrix for each skill
        for skill_idx in skills_tested:
            # Map difficulty value to index using difficulty_to_idx
            if original_difficulty in self.difficulty_to_idx:
                diff_idx = self.difficulty_to_idx[original_difficulty]
                if 0 <= diff_idx < self.difficulty_levels:
                    # Update count and accuracy
                    self.skill_difficulty_counts[skill_idx, diff_idx] += 1
                    current_accuracy = self.skill_difficulty_accuracy[skill_idx, diff_idx]
                    current_count = self.skill_difficulty_counts[skill_idx, diff_idx]
                    
                    # Update accuracy using incremental mean
                    if current_count == 1:
                        self.skill_difficulty_accuracy[skill_idx, diff_idx] = float(correct)
                    else:
                        # Incremental update: new_avg = old_avg + (new_value - old_avg) / new_count
                        self.skill_difficulty_accuracy[skill_idx, diff_idx] = current_accuracy + (float(correct) - current_accuracy) / current_count



    def _simulate_response(
        self, question_idx: int, skills_tested: np.ndarray
    ) -> Tuple[bool, float]:
        """Simulate student's response using response model.

        Returns:
            Tuple[bool, float]: (correct, probability)
        """
        if self.response_model is None:
            # Default behavior if no response model is provided
            return self.rng.random() > 0.5, 0.5

        if isinstance(self.response_model, (BKTModel, IRTModel)):
            # For BKT and IRT models
            if isinstance(self.response_model, IRTModel):
                correct, prob = self.response_model.predict_response(
                    question_idx, skills_tested
                )
            else:  # BKTModel
                correct, prob = self.response_model.predict_response(skills_tested)

            return correct, prob

        else:
            raise ValueError(
                f"Unknown response model type: {type(self.response_model)}"
            )



    def _calculate_batch_rewards(self, question_info_list: List[Dict]) -> Dict[str, float]:
        """
        Calculate rewards for a batch of questions based on the new reward structure.
        
        Args:
            question_info_list: List of dictionaries containing question information
            
        Returns:
            Dict[str, float]: Dictionary mapping objective names to their reward values
        """
        rewards_dict = {}
        
        if not question_info_list:
            return {"performance": 0.0, "gap": 0.0, "aptitude": 0.0}
        
        
        # Calculate rewards for each question
        all_perf_rewards = []
        all_apt_rewards = []
        all_gap_rewards = []

        for question_info in question_info_list:
            question_idx = question_info["question_id"]
            correct = question_info["correct"]
            skills = question_info["skills"]
            original_difficulty = question_info["original_difficulty"]
            scaled_difficulty = question_info["scaled_difficulty"]
            
            # Convert skill names to indices
            skills_tested = []
            for skill_name in skills:
                if skill_name in self.skill_to_idx:
                    skills_tested.append(self.skill_to_idx[skill_name])
            
            if not skills_tested:
                continue
                
            # Performance reward: correct = 1, incorrect = 0
            r_perf = 1.0 if correct else 0.0
            all_perf_rewards.append(r_perf)
            
            # Gap reward: failed question that are cleared first time + 1
            if correct and question_idx not in self.cleared_questions and question_idx in self.all_failed_questions:
                r_gap = 1.0
            else:
                r_gap = 0.0
            all_gap_rewards.append(r_gap)
            
            # Aptitude reward: difficulty - mastery (both in 0-1 range)
            r_apt = 0.0
            avg_mastery = np.mean([self.mastery[idx] for idx in skills_tested])
            if avg_mastery == 1.0 and scaled_difficulty == 1.0:
                r_apt = 1.0
            elif avg_mastery == 1.0 and scaled_difficulty < 1.0:
                r_apt = 0.0
            else:
                r_apt = (scaled_difficulty - avg_mastery) / (1-avg_mastery)
            all_apt_rewards.append(r_apt)
        
        # Calculate average rewards across all questions
        avg_perf = np.mean(all_perf_rewards)
        avg_gap = np.mean(all_gap_rewards)
        avg_apt = np.mean(all_apt_rewards)
        
        # Ensure rewards are in [0, 1] range (except gap which can be negative)
        avg_perf = np.clip(avg_perf, 0, 1)
        avg_apt = np.clip(avg_apt, 0, 1)
        avg_gap = np.clip(avg_gap, 0, 1)
        
        # Store rewards in dictionary by objective name
        rewards_dict["performance"] = round(avg_perf, 4)
        rewards_dict["gap"] = round(avg_gap, 4)
        rewards_dict["aptitude"] = round(avg_apt, 4)
        return rewards_dict

    def set_difficulty_levels(self, difficulties: dict):
        """
        Set difficulty levels and initialize related matrices based on question bank difficulties.
        Uses the already rescaled difficulty values and original difficulty values directly.
        
        Args:
            difficulties: dict, key is question_idx, value is rescaled difficulty level and original difficulty level for this question.
        """
        if not difficulties:
            raise ValueError("Difficulties dictionary cannot be empty")
    
        # Use the already rescaled difficulty values and original difficulty values directly
        self.difficulties = difficulties.copy()
        
        # Get unique difficulty levels from the rescaled values
        unique_difficulties = sorted(list(set([difficulties[idx]["original_difficulty"] for idx in difficulties])))
        self.difficulty_levels = len(unique_difficulties)
        logging.info(f"Difficulty levels: {unique_difficulties}")
        
        # Create difficulty mapping from rescaled values to indices
        self.difficulty_to_idx = {diff: idx for idx, diff in enumerate(unique_difficulties)}
        
        # Initialize matrices with the correct dimensions
        self.failed_questions_ratio = np.zeros((self.num_skills, self.difficulty_levels), dtype=np.float32)
        self.skill_difficulty_accuracy = np.zeros((self.num_skills, self.difficulty_levels), dtype=np.float32)
        self.skill_difficulty_counts = np.zeros((self.num_skills, self.difficulty_levels), dtype=np.uint8)  # Use uint8 for counts
        
        # Update observation space dimensions
        mastery_dim = self.num_skills
        skill_difficulty_accuracy_dim = self.num_skills * self.difficulty_levels
        failed_questions_ratio_dim = self.num_skills * self.difficulty_levels
        
        obs_dim = mastery_dim + skill_difficulty_accuracy_dim + failed_questions_ratio_dim
        
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        min_scaled = 1.0 / len(unique_difficulties)
        max_scaled = 1.0
        print(f"Difficulty scaling: {len(unique_difficulties)} unique difficulties mapped to [1/{len(unique_difficulties)}, 1.0] range")

    def _recommend_failed_questions_old(self) -> List[int]:
        """
        Action 0: Recommend Gap Questions
        Select questions that are more difficult than current mastery and have smallest gap (weakest points)
        """
        # Check if gap cache is None (not initialized)
        if self.gap_cache is None:
            raise ValueError("Gap cache not initialized")
        
        candidate_questions = [(qid, self.gap_cache[qid]) for qid in self.gap_cache]
        
        if not candidate_questions:
            # If no questions in cache, fall back to random
            return self._recommend_random_questions()
        
        # Sort by gap (ascending - smaller gap means weaker points)
        candidate_questions.sort(key=lambda x: x[1])
        
        # Find questions with minimum gap (weakest points)
        min_gap = candidate_questions[0][1]
        min_gap_questions = [q for q in candidate_questions if q[1] == min_gap]
        
        if len(min_gap_questions) >= self.questions_per_step:
            # If we have enough questions with min gap, select randomly from them
            selected = self.rng.choice(len(min_gap_questions), 
                                    size=self.questions_per_step, 
                                    replace=False)
            selected_qids = [min_gap_questions[i][0] for i in selected]
        else:
            # Otherwise, select top questions by gap (weakest points first)
            selected_qids = [q[0] for q in candidate_questions[:self.questions_per_step]]
        
        return selected_qids

    def _recommend_failed_questions(self) -> List[int]:
        """
        Action 0: Recommend Gap Questions
        Select from valid failed questions randomly
        """
        # Get valid failed questions
        valid_failed_questions = self._get_valid_failed_questions()
        
        # if not valid_failed_questions:
        #     # If no valid failed questions, fall back to random
        #     return self._recommend_random_questions()
        
        # Randomly select from valid failed questions
        num_to_select = min(self.questions_per_step, len(valid_failed_questions))
        selected_qids = self.rng.choice(valid_failed_questions, size=num_to_select, replace=False).tolist()
        
        logging.info(f"Selected {len(selected_qids)} gap questions from {len(valid_failed_questions)} valid failed questions")
        
        return selected_qids

    def _get_valid_failed_questions(self) -> List[int]:
        """
        Get valid failed questions based on current mastery level.
        Valid questions are those with difficulty <= avg_mastery + 0.2 and not in cleared_questions.
        
        Returns:
            List of valid failed question IDs
        """
        avg_mastery = float(np.mean(self.mastery))  # Convert to Python float
        difficulty_threshold = min(avg_mastery + 0.2, 1.0)
        
        valid_failed_questions = []
        for qid, (skills, original_difficulty, scaled_difficulty) in self.all_failed_questions.items():
            # Check if question is not cleared and within difficulty threshold
            if qid not in self.cleared_questions and scaled_difficulty <= difficulty_threshold:
                valid_failed_questions.append(int(qid))  # Ensure qid is Python int
        
        return valid_failed_questions

    def _recommend_easy_questions(self) -> List[int]:
        """
        Action 1: Recommend Performance Questions
        Select questions that are more difficult than current mastery and most experienced/familiar
        """
        # Check if experience cache is None (not initialized)
        if self.experience_cache is None:
            raise ValueError("Experience cache not initialized")
        
        candidate_questions = [(qid, self.experience_cache[qid]) for qid in self.experience_cache]
        
        if not candidate_questions:
            # If no questions in cache, fall back to random
            return self._recommend_random_questions()
        
        # Sort by experience (ascending - lower experience means more familiar)
        candidate_questions.sort(key=lambda x: x[1])
        
        # Find questions with minimum experience (most familiar)
        min_exp = candidate_questions[0][1]
        min_exp_questions = [q for q in candidate_questions if q[1] == min_exp]
        
        if len(min_exp_questions) >= self.questions_per_step:
            # If we have enough questions with min experience, select randomly from them
            selected = self.rng.choice(len(min_exp_questions), 
                                    size=self.questions_per_step, 
                                    replace=False)
            selected_qids = [min_exp_questions[i][0] for i in selected]
        else:
            # Otherwise, select top questions by experience (most familiar first)
            selected_qids = [q[0] for q in candidate_questions[:self.questions_per_step]]
        
        return selected_qids

    def _recommend_high_aptitude_questions_old(self) -> List[int]:
        """
        Action 2: Recommend High-Aptitude Questions
        Select questions that are more difficult than current mastery and have highest aptitude
        """
        
        # Check if aptitude cache is None (not initialized)
        if self.aptitude_cache is None:
            raise ValueError("Aptitude cache not initialized")
        
        candidate_questions = [(qid, self.aptitude_cache[qid]) for qid in self.aptitude_cache]
        
        if not candidate_questions:
            # If no questions are harder than current mastery, fall back to random
            return self._recommend_random_questions()
        
        # Sort by aptitude (descending)
        candidate_questions.sort(key=lambda x: (-x[1]))
        
        # Find questions with maximum aptitude
        max_apt = candidate_questions[0][1]
        max_apt_questions = [q for q in candidate_questions if q[1] == max_apt]
        
        if len(max_apt_questions) >= self.questions_per_step:
            # If we have enough questions with max aptitude, select randomly from them
            selected = self.rng.choice(len(max_apt_questions), 
                                    size=self.questions_per_step, 
                                    replace=False)
            selected_qids = [max_apt_questions[i][0] for i in selected]
        else:
            # Otherwise, select top questions by aptitude
            selected_qids = [q[0] for q in candidate_questions[:self.questions_per_step]]
        
        return selected_qids

    def _recommend_high_aptitude_questions(self) -> List[int]:
        """
        Action 2: Recommend High-Aptitude Questions
        Select questions with difficulty >= min{avg_mastery + 0.2, 1} randomly
        """
        # Calculate average mastery across all skills
        avg_mastery = np.mean(self.mastery)
        difficulty_threshold = min(avg_mastery + 0.2, 1.0)
        
        # Find all questions with difficulty >= threshold
        candidate_questions = []
        for qid in self.question_skills_difficulty_map:
            scaled_difficulty = self.question_skills_difficulty_map[qid]["scaled_difficulty"]
            if scaled_difficulty >= difficulty_threshold:
                candidate_questions.append(qid)
        
        # if not candidate_questions:
        #     # If no questions meet the threshold, fall back to random
        #     return self._recommend_random_questions()
        
        # Randomly select from candidate questions
        num_to_select = min(self.questions_per_step, len(candidate_questions))
        selected_qids = self.rng.choice(candidate_questions, size=num_to_select, replace=False).tolist()
        
        logging.info(f"Selected {len(selected_qids)} high aptitude questions with difficulty >= {difficulty_threshold:.3f}")

        # if not enough, select randomly
        # remaining_needed = self.questions_per_step - len(selected_qids)
        # if remaining_needed > 0:
        #     all_qs = [qid for qid in self.question_skills_difficulty_map if qid not in selected_qids]
        #     if all_qs:
        #         num_additional = min(remaining_needed, len(all_qs))
        #         selected_additional = self.rng.choice(all_qs, size=num_additional, replace=False)
        #         selected_qids.extend(selected_additional)
        # logging.info(f"others from random questions")
        
        return selected_qids

    def _recommend_random_questions(self) -> List[int]:
        """
        Action 3: Random Questions
        Select self.questions_per_step random questions that are more difficult than current mastery
        """
        # Filter questions that are more difficult than current mastery for each skill
        candidate_questions = []
        
        for qid in self.question_skills_difficulty_map:
            info = self.question_skills_difficulty_map[qid]
            scaled_difficulty = info["scaled_difficulty"]
            skills = info["skills"]
            
            # Check if this question is more difficult than mastery for any tested skill
            is_harder = False
            for skill_name in skills:
                skill_idx = self.skill_to_idx[skill_name]
                if scaled_difficulty > self.mastery[skill_idx] or (self.mastery[skill_idx] == 1.0 and scaled_difficulty == 1.0):
                    is_harder = True
                    break
            
            if is_harder:
                candidate_questions.append(qid)
        
        if not candidate_questions:
            # If no questions are harder than current mastery, fall back to all questions
            candidate_questions = list(self.question_skills_difficulty_map.keys())
        
        # Select random questions from candidates
        num_to_select = min(self.questions_per_step, len(candidate_questions))
        return self.rng.choice(candidate_questions, size=num_to_select, replace=False).tolist()

    def _select_questions_by_action(self, action: int) -> List[int]:
        """
        Select questions based on the action type.
        
        Args:
            action: Action type (0: Failed, 1: Easy, 2: High-Aptitude)
            
        Returns:
            List of question IDs to present
        """
        if action == 0:
            return self._recommend_failed_questions()
        elif action == 1:
            return self._recommend_easy_questions()
        elif action == 2:
            return self._recommend_high_aptitude_questions()
        else:
            raise ValueError(f"Invalid action: {action}")

    def set_question_skills_difficulty(
        self, question_idx: int, 
        skills: List[str],
        original_difficulty: float = None,
        scaled_difficulty: float = None,
    ):
        """
        NEW LOGIC (no indexing):
        - question_idx is actually the dataset question ID (same as QA.json 'id' and CSV question_id).
        - self.difficulties must be a dict keyed by that same question ID.
        """
        qid = int(question_idx)

        if not hasattr(self, 'difficulties') or self.difficulties is None:
            raise ValueError("Difficulty scaling not initialized")

        if qid not in self.difficulties:
            raise KeyError(
                f"Question id {qid} not found in difficulties. "
                f"Make sure your CSV has this id and you loaded it into set_difficulty_levels()."
            )

        scaled_difficulty = self.difficulties[qid]["scaled_difficulty"]
        original_difficulty = self.difficulties[qid]["original_difficulty"]

        self.question_skills_difficulty_map[qid] = {
            "skills": skills,
            "scaled_difficulty": scaled_difficulty,
            "original_difficulty": original_difficulty
        }

        # Update pre-computed lists
        self.aptitude_cache = {}

    def _update_aptitude_cache(self):
        """Update aptitude cache for questions that are more difficult than current mastery."""
        self.aptitude_cache = {}
        
        for qid in self.question_skills_difficulty_map:
            info = self.question_skills_difficulty_map[qid]
            scaled_difficulty = info["scaled_difficulty"]
            skills = info["skills"]
            
            # Check if this question is more difficult than mastery for any tested skill
            is_harder = False
            aptitudes = []
            
            for skill_name in skills:
                skill_idx = self.skill_to_idx[skill_name]
                if scaled_difficulty > self.mastery[skill_idx] or (self.mastery[skill_idx] == 1.0 and scaled_difficulty == 1.0):
                    is_harder = True
                    # Calculate aptitude: diff = scaled_difficulty - mastery
                    diff = round(scaled_difficulty - self.mastery[skill_idx], 2)
                    aptitude = max(0, diff)  
                    aptitudes.append(aptitude)
            
            if is_harder and aptitudes:
                # Use average aptitude across skills
                avg_aptitude = np.mean(aptitudes)
                self.aptitude_cache[qid] = avg_aptitude

    def _update_experience_cache(self):
        """Update experience cache for questions that are more difficult than current mastery."""
        self.experience_cache = {}
        
        for qid in self.question_skills_difficulty_map:
            info = self.question_skills_difficulty_map[qid]
            scaled_difficulty = info["scaled_difficulty"]
            skills = info["skills"]
            
            # Check if this question is more difficult than mastery for any tested skill
            is_harder = False
            experiences = []
            
            for skill_name in skills:
                skill_idx = self.skill_to_idx[skill_name]
                if scaled_difficulty > self.mastery[skill_idx] or (self.mastery[skill_idx] == 1.0 and scaled_difficulty == 1.0):
                    is_harder = True
                    # Calculate experience for this skill-scaled_difficulty combination
                    experience = self._calculate_experience(scaled_difficulty)
                    experiences.append(experience)
            
            if is_harder and experiences:
                # Use average experience across skills
                avg_experience = np.mean(experiences)
                self.experience_cache[qid] = avg_experience

    def _update_gap_cache(self):
        """Update gap cache for questions that are more difficult than current mastery."""
        self.gap_cache = {}
        
        for qid in self.question_skills_difficulty_map:
            info = self.question_skills_difficulty_map[qid]
            scaled_difficulty = info["scaled_difficulty"]
            skills = info["skills"]
            
            # Check if this question is more difficult than mastery for any tested skill
            is_harder = False
            gaps = []
            
            for skill_name in skills:
                skill_idx = self.skill_to_idx[skill_name]
                if scaled_difficulty > self.mastery[skill_idx] or (self.mastery[skill_idx] == 1.0 and scaled_difficulty == 1.0):
                    is_harder = True
                    # Calculate gap for this skill-scaled_difficulty combination
                    gap = self._calculate_gap(scaled_difficulty)
                    gaps.append(gap)
            
            if is_harder and gaps:
                # Use average gap across skills
                avg_gap = np.mean(gaps)
                self.gap_cache[qid] = avg_gap



    def _update_mastery(self, all_question_info: List[dict]) -> None:
        """Update mastery levels based on all questions using NCC condition.

        Args:
            all_question_info: List of question info dictionaries containing:
                - question_id: Question identifier
                - skills: List of skill names tested
                - scaled_difficulty: Scaled difficulty level (0-1 range)
                - original_difficulty: Original difficulty level
                - correct: Whether the response was correct
                - skills_tested: Array of skill indices being tested
        """
        # Group materials by skill
        skill_materials = {}
        for question_info in all_question_info:
            skills_tested = question_info["skills_tested"]
            scaled_difficulty = question_info["scaled_difficulty"]
            correct = question_info["correct"]
            
            for skill_idx in skills_tested:
                if skill_idx not in skill_materials:
                    skill_materials[skill_idx] = []
                skill_materials[skill_idx].append((scaled_difficulty, correct))
        
        # Update mastery for each skill with all its materials
        for skill_idx, materials in skill_materials.items():
            self._update_skill_mastery(skill_idx, materials)

    def _calculate_experience(self, scaled_difficulty: float, sim_threshold: float = 0.1) -> float:
        """
        Calculate experience for a given scaled_difficulty based on seen materials.
        
        Args:
            scaled_difficulty: scaled_difficulty level of the question
            sim_threshold: Similarity threshold for considering materials as similar
            
        Returns:
            Experience value (lower is more experienced/familiar)
        """
        # Get correct seen materials
        correct_seen = [prev_mat for prev_mat in self.seen_materials if prev_mat[2] == 1]
        
        if len(correct_seen) == 0:
            return 0.0
        
        # Calculate similarity scores (absolute difference from target difficulty)
        similarities = [round(abs(prev_mat[1] - scaled_difficulty), 2) for prev_mat in correct_seen]
        
        # Return average similarity (lower means more similar/experienced)
        return sum(similarities) / len(similarities)

    def _calculate_gap(self, scaled_difficulty: float, window_l: int = 3) -> float:
        """
        Calculate gap for a given scaled_difficulty based on recent incorrect seen materials.
        
        Args:
            scaled_difficulty: scaled_difficulty level of the question
            window_l: Window size for recent incorrect materials
            
        Returns:
            Gap value (lower means smaller gap/weaker point)
        """
        # Get recent incorrect seen materials (last window_l items, reversed)
        incorrect_seen = [prev_mat for prev_mat in self.seen_materials if prev_mat[2] == 0]
        incorrect_seen = incorrect_seen[::-1][:window_l]  # Reverse and take last window_l
        
        if len(incorrect_seen) == 0:
            return 0.0
        
        # Calculate similarity scores (absolute difference from target difficulty)
        similarities = [abs(prev_mat[1] - scaled_difficulty) for prev_mat in incorrect_seen]
        
        # Return average similarity (lower means smaller gap)
        return sum(similarities) / len(similarities)
    
    def _calculate_gap_v2(self, scaled_difficulty: float, window_l: int = 3) -> float:
        """
        Calculate gap for a given scaled_difficulty based on recent incorrect seen materials.
        
        Args:
            scaled_difficulty: scaled_difficulty level of the question
            window_l: Window size for recent incorrect materials
            
        Returns:
            Gap value (lower means smaller gap/weaker point)
        """
        # Get recent incorrect seen materials (last window_l items, reversed)
        incorrect_seen = [prev_mat for prev_mat in self.seen_materials if prev_mat[2] == 0]
        incorrect_seen = incorrect_seen[::-1][:window_l]  # Reverse and take last window_l
        
        if len(incorrect_seen) == 0:
            return np.nan
        
        # Calculate similarity scores (absolute difference from target difficulty)
        similarities = [abs(prev_mat[1] - scaled_difficulty) for prev_mat in incorrect_seen]
        
        # Return average similarity (lower means smaller gap)
        return sum(similarities) / len(similarities)

    def _update_skill_mastery(self, skill_idx: int, materials: List[Tuple[float, bool]]) -> None:
        """Update mastery for a single skill using NCC condition with all materials.
        
        Args:
            skill_idx: Index of the skill to update
            materials: List of (scaled_difficulty, correct) tuples for all questions
        """
        # Get current mastery for this skill
        current_mastery = self.mastery[skill_idx]
        max_skill = current_mastery
        enter = True
        smallest_false_diff = None
        
        # Process all materials in order
        for scaled_difficulty, correct in materials:
            # Update NCC tracking for this scaled_difficulty
            if scaled_difficulty not in self.ncc_tracking[skill_idx]:
                self.ncc_tracking[skill_idx][scaled_difficulty] = []
            
            # Add current response to history
            if len(self.ncc_tracking[skill_idx][scaled_difficulty]) == self.ncc_window:
                self.ncc_tracking[skill_idx][scaled_difficulty].pop(0)
            self.ncc_tracking[skill_idx][scaled_difficulty].append(correct)
            
            # Process the response
            if correct and enter:
                max_skill = scaled_difficulty
            elif enter:
                smallest_false_diff = scaled_difficulty
                enter = False
        
        # If we encountered a wrong answer, clear higher difficulty histories
        if not enter and smallest_false_diff is not None:
            for diff in list(self.ncc_tracking[skill_idx].keys()):
                if diff >= smallest_false_diff:
                    self.ncc_tracking[skill_idx][diff] = []
        
        # Check if we can advance to new mastery level
        if max_skill != current_mastery:
            # Check NCC condition: need window consecutive correct answers at this difficulty
            if (len(self.ncc_tracking[skill_idx][max_skill]) < self.ncc_window or 
                sum(self.ncc_tracking[skill_idx][max_skill]) < self.ncc_window):
                max_skill = current_mastery        
        
        # Update mastery
        self.mastery[skill_idx] = max_skill
        
        # Ensure mastery stays in [0,1] range
        self.mastery[skill_idx] = np.clip(self.mastery[skill_idx], 0, 1)

    def _update_failed_question(
        self,
        question_idx: int,
        skills_tested: np.ndarray,
        correct: bool,
        original_difficulty: float,
        scaled_difficulty: float,
    ):
        """
        Update all_failed_questions dict, cleared_questions dict and failed_questions_ratio matrix based on the response.
        If answered incorrectly, add to all_failed_questions.
        If answered correctly, add to cleared_questions (all_failed_questions is never removed).
        """
        if len(skills_tested) == 0:
            return

        # Get skill names for the tested skills
        tested_skill_names = [self.skills[idx] for idx in skills_tested]

        if not correct:
            # Add to global failed questions list
            logging.info(f"ADD to all_failed_questions: {question_idx}, {tested_skill_names}, {scaled_difficulty}")
            self.all_failed_questions[int(question_idx)] = (tested_skill_names, float(original_difficulty), float(scaled_difficulty))
        elif correct and question_idx in self.all_failed_questions:
            # Add to cleared questions
            logging.info(f"ADD to cleared_questions: {question_idx}, {tested_skill_names}, {scaled_difficulty}")
            self.cleared_questions[int(question_idx)] = (tested_skill_names, float(original_difficulty), float(scaled_difficulty))
            # Note: all_failed_questions is never removed - it keeps all historical failed questions
        
        # Update failed_questions_ratio matrix with ratios
        self._update_failed_questions_ratio()

    def _update_failed_questions_ratio(self):
        """
        Update failed_questions_ratio matrix with ratios (proportion of total valid failed questions).
        """
        # Reset the matrix
        self.failed_questions_ratio = np.zeros((self.num_skills, self.difficulty_levels), dtype=np.float32)
        
        # Get valid failed questions
        valid_failed_questions = self._get_valid_failed_questions()
        total_valid_failed = len(valid_failed_questions)
        
        if total_valid_failed == 0:
            return  # No valid failed questions, keep matrix as zeros
        
        # Calculate ratios for each (skill, original_difficulty) combination from valid failed questions
        for question_idx in valid_failed_questions:
            if question_idx in self.all_failed_questions:
                skills, original_difficulty, scaled_difficulty = self.all_failed_questions[question_idx]
            for skill_name in skills:
                skill_idx = self.skill_to_idx[skill_name]
                # Map difficulty value to index using difficulty_to_idx
                diff_idx = self.difficulty_to_idx[original_difficulty]
                if 0 <= diff_idx < self.difficulty_levels:
                    # Add 1/total_valid_failed for each valid failed question in this combination
                    self.failed_questions_ratio[skill_idx, diff_idx] += 1.0 / total_valid_failed

    