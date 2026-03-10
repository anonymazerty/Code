"""
Reward machine handler and example finite state reward machine for staged multi-objective RL.
"""

import logging
from typing import Any
import numpy as np

class EduRewardMachine:
    """
    Dynamic finite state reward machine for staged multi-objective RL:
    - u_0 (performance stage): Improve accuracy when no valid failed questions (use r_perf)
    - u_1 (gap stage): Clear valid failed questions (use r_gap) 
    - u_2 (aptitude stage): Challenge with high-aptitude questions when mastery >= 0.8 (use r_apt)
    
    State transitions are determined dynamically based on environment state:
    - Performance → Gap: when valid failed questions exist
    - Gap → Performance: when no valid failed questions
    - Performance → Aptitude: when mastery >= 0.8 and no valid failed questions
    - Gap → Aptitude: when mastery >= 0.8 and no valid failed questions
    - Aptitude → Performance: when mastery < 0.8
    - Aptitude → Gap: when valid failed questions appear
    """

    def __init__(
        self,
        objectives=None,
        aptitude_state_threshold=0.6,
    ):
        self.objectives = objectives
        self.aptitude_state_threshold = aptitude_state_threshold
        
        # Check if we have required objectives
        self.has_gap = "gap" in self.objectives
        self.has_performance = "performance" in self.objectives
        self.has_aptitude = "aptitude" in self.objectives
        
        if len(self.objectives) < 2:
            raise ValueError(f"Need at least 2 objectives for reward machine")
        
        # Initialize state - always start from performance stage
        self.state = "u_0"
        # if self.has_gap:
        #     self.state = "u_1"  # Start from gap stage
        # else:
        #     self.state = "u_0"  # Start from performance stage

    def get_reward(self, reward_dict, next_state_info):
        """
        Get reward based on current stage and state information.
        
        Args:
            reward_dict: Dictionary mapping objective names to reward values
            next_state_info: Dictionary containing state information
        """
        
        # Get state information
        has_valid_failed_questions = np.sum(next_state_info["failed_questions_ratio"]) > 0
        # valid_failed_questions = next_state_info["valid_failed_questions"]
        avg_mastery = next_state_info["avg_mastery"]
        
        # ---------- u_0 (performance stage) ----------
        if self.state == "u_0":
            scalar_reward = 0.5 * reward_dict["performance"] + 0.5 * reward_dict["gap"]
            
            # Check transitions
            if avg_mastery >= self.aptitude_state_threshold:
                # Transition to aptitude stage if mastery >= threshold
                self.state = "u_1"
                logging.info(f"Transitioning from performance stage to aptitude stage (mastery: {avg_mastery:.2f})")

            # if has_valid_failed_questions and self.has_gap:
            #     # Transition to gap stage if there are valid failed questions
            #     self.state = "u_1"
            #     logging.info(f"Transitioning from performance stage to gap stage (valid failed questions detected)")
            # elif not has_valid_failed_questions and avg_mastery >= self.aptitude_state_threshold and self.has_aptitude:
            #     # Transition to aptitude stage if no valid failed questions and mastery >= threshold
            #     self.state = "u_2"
            #     logging.info(f"Transitioning from performance stage to aptitude stage (mastery: {avg_mastery:.2f})")
            
            return scalar_reward

        # ---------- u_1 (gap stage) ----------
        elif self.state == "u_1":
            scalar_reward = 0.5 * reward_dict["gap"] + 0.5 * reward_dict["aptitude"]
            
            # Check transitions
            if avg_mastery < self.aptitude_state_threshold:
                # Transition to aptitude stage if mastery < threshold
                self.state = "u_0"
                logging.info(f"Transitioning from aptitude stage to performance stage (mastery: {avg_mastery:.2f})")
            # if not has_valid_failed_questions and self.has_performance:
            #     # Back to performance stage if no valid failed questions
            #     self.state = "u_0"
            #     logging.info(f"Transitioning from gap stage to performance stage (no valid failed questions)")
            # elif not has_valid_failed_questions and avg_mastery >= self.aptitude_state_threshold and self.has_aptitude:
            #     # To aptitude stage if no valid failed questions and mastery >= threshold
            #     self.state = "u_2"
            #     logging.info(f"Transitioning from gap stage to aptitude stage (mastery: {avg_mastery:.2f})")
            
            return scalar_reward

        # ---------- u_2 (aptitude stage) ----------
        # elif self.state == "u_2":
        #     scalar_reward = reward_dict["aptitude"]
            
        #     # Check transitions
        #     if avg_mastery < self.aptitude_state_threshold and self.has_performance:
        #         # Back to performance stage if mastery drops below threshold
        #         self.state = "u_0"
        #         logging.info(f"Transitioning from aptitude stage to performance stage (mastery: {avg_mastery:.2f})")
        #     elif has_valid_failed_questions and self.has_gap:
        #         # Back to gap stage if valid failed questions appear
        #         self.state = "u_1"
        #         logging.info(f"Transitioning from aptitude stage to gap stage (valid failed questions detected)")
            
        #     return scalar_reward

    def reset(self):
        """Reset to initial performance stage."""
        self.state = "u_0"
        # self.state = "u_1" if self.has_gap else "u_0" # Always start from gap stage
        logging.info(f"Reward machine reset to {self.state} stage")
    
    def get_current_state(self):
        """Get current state name."""
        return self.state
    
    def get_current_state_description(self):
        """Get human-readable description of current state."""
        state_descriptions = {
            "u_0": "Performance Stage - Improving accuracy when no valid failed questions",
            "u_1": "Gap Stage - Clearing valid failed questions", 
            "u_2": "Aptitude Stage - Challenging with high-difficulty questions when mastery >= 0.8"
        }
        return state_descriptions.get(self.state, f"Unknown state: {self.state}")
    
    def get_state_transition_summary(self):
        """Get summary of possible state transitions."""
        return {
            "u_0": {
                "description": "Performance Stage",
                "reward": "performance",
                "transitions": [
                    "→ u_1 (Gap): when valid failed questions exist",
                    "→ u_2 (Aptitude): when mastery >= 0.8 AND no valid failed questions"
                ]
            },
            "u_1": {
                "description": "Gap Stage", 
                "reward": "gap",
                "transitions": [
                    "→ u_0 (Performance): when no valid failed questions",
                    "→ u_2 (Aptitude): when mastery >= 0.8 AND no valid failed questions"
                ]
            },
            "u_2": {
                "description": "Aptitude Stage",
                "reward": "aptitude", 
                "transitions": [
                    "→ u_0 (Performance): when mastery < 0.8",
                    "→ u_1 (Gap): when valid failed questions appear"
                ]
            }
        }


class RewardMachineHandler:
    """
    Handler for using a reward machine (finite state automaton) to compute rewards.
    The reward machine should provide get_reward(state, action, next_state_info) and reset().
    """

    def __init__(self, reward_machine: Any):
        self.reward_machine = reward_machine

    def process_reward(self, reward_dict, next_state_info, **kwargs) -> float:
        return self.reward_machine.get_reward(reward_dict, next_state_info)

    def reset(self):
        self.reward_machine.reset()
