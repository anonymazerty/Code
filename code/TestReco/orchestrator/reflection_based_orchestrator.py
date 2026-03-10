"""
Reflection-based Orchestrator for educational recommendation system.

This orchestrator builds on Tool Call Orchestrator (TO) by adding reflection component (c_reflct).
It can simulate multi-step rollouts with policies to evaluate their long-term performance,
or make direct decisions about which policy to use for question recommendation.
"""

import logging
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from langchain_core.runnables import Runnable

from tools.policies import PolicyFactory
from orchestrator.reflection_based_prompts import ReflectionBasedPromptTemplates
from orchestrator.langchain_wrapper import LangChainWrapper
from orchestrator.shared_components import BaseOrchestrator


class ReflectionBasedOrchestrator(BaseOrchestrator):
    """
    Reflection-based Orchestrator that can simulate multi-step rollouts to evaluate policies.
    
    The orchestrator can:
    1. Simulate calling a policy for multiple steps to get rollout trajectory
    2. Evaluate policies based on cumulative rewards over the rollout
    3. Make a decision about which policy to use for question recommendation
    4. Support multiple rounds of rollout simulation before making the final decision
    """
    
    def __init__(
        self,
        env,
        llm,
        policy_configs: Dict[str, Dict[str, Any]],
        verbose: bool = True,
        objectives: List[str] = None,
        max_rollouts: int = 3,
        rollout_steps: int = 3,
    ):
        """
        Initialize the reflection-based orchestrator.
        
        Args:
            env: Environment instance
            llm: Language model instance
            policy_configs: Dictionary mapping policy names to their configs
            verbose: Whether to enable verbose logging
            objectives: List of optimization objectives
            max_rollouts: Maximum number of rollouts before forcing a decision
            rollout_steps: Number of steps in each rollout simulation
        """
        # Initialize base orchestrator
        super().__init__(env, llm, policy_configs, verbose, objectives)
        
        # Reflection-specific initialization
        self.max_rollouts = max_rollouts
        self.rollout_steps = rollout_steps
        
        # Create Runnable chain with appropriate prompt
        self.chain = self._create_chain()
        
        # Track rollout history for current step
        self.current_step_rollouts = []
        
    
    def _create_chain(self) -> Runnable:
        """Create Runnable chain with system prompt and reflection-based prompt template."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate
        
        # Check if this is Claude (custom LLM with Claude-like response format)
        self.is_claude = hasattr(self.custom_llm, 'model_name') and 'claude' in self.custom_llm.model_name.lower()
        
        # Create chat prompt template with system and human messages
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", ReflectionBasedPromptTemplates.SYSTEM_PROMPT.template),
            ("human", ReflectionBasedPromptTemplates.FEW_SHOT_POLICY_SELECTION.template)
        ])
        
        # Create Runnable chain using prompt | llm pattern
        return chat_prompt | self.llm
    
    def select_action(
        self, 
        state: np.ndarray, 
        available_actions: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Select an action using reflection-based orchestrator.
        
        Args:
            state: Current state
            available_actions: List of available actions (optional)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (action_info, orchestrator_info)
        """
        # Reset rollout history for new step
        self.current_step_rollouts = []
        
        # Extract state information
        state_info = self._extract_state_info()
        
        # Get policy meta information
        policy_meta_info = self._get_policy_meta_info()
        
        # Main decision loop
        for rollout_round in range(self.max_rollouts + 1):
            if self.verbose:
                logging.info(f"Rollout round {rollout_round + 1}/{self.max_rollouts + 1}")
            
            # Prepare input for LLM chain
            chain_input = self._prepare_chain_input(
                state_info, 
                policy_meta_info, 
                self.current_step_rollouts
            )
            
            # Log the complete prompt for debugging
            if self.verbose:
                self._log_prompt_details(chain_input)
            
            # Call the LLM to decide whether to simulate rollout or make decision
            response = self.chain.invoke(chain_input, config={"max_tokens": 4096})
            
            # Log LLM response
            if self.verbose:
                self._log_llm_response(response)
            
            # Parse the response
            parsed_response = self._parse_orchestrator_response(response)
            
            if not parsed_response["is_final_decision"]:
                # Simulate rollout with the specified policy
                rollout_feedback = self._simulate_policy_rollout(
                    parsed_response["policy"], 
                    state, 
                    available_actions
                )
                
                # Record rollout
                self.current_step_rollouts.append({
                    "round": rollout_round + 1,
                    "policy_called": parsed_response["policy"],
                    "rollout_feedback": rollout_feedback,
                    # "reasoning": parsed_response.get("reasoning", "")
                })
                
                if self.verbose:
                    logging.info(f"Rollout {rollout_round + 1}: Simulated {parsed_response['policy']}")
                    # logging.info(f"Cumulative reward: {rollout_feedback['cumulative_reward']:.3f}")
                
                # Continue to next round if we haven't reached max rollouts
                if rollout_round < self.max_rollouts:
                    continue
                else:
                    # Force decision after max rollouts
                    if self.verbose:
                        logging.info("Reached max rollouts, forcing decision")
                    break
                    
            elif parsed_response["is_final_decision"]:
                # Make final decision about which policy to use
                final_policy = parsed_response["policy"]
                if self.verbose:
                    logging.info(f"Making decision: Selected policy {final_policy}")
                
                # Call the selected policy to get action
                action_info = self._call_selected_policy(final_policy, state, available_actions)
                
                orchestrator_info = {
                    "orchestrator_type": "reflection_based",
                    "selected_strategy": final_policy,
                    "rollouts_made": self.current_step_rollouts,
                    "total_rollouts": len(self.current_step_rollouts),
                    "final_reasoning": parsed_response["reasoning"]
                }
                
                return action_info, orchestrator_info
            else:
                raise ValueError(f"Invalid response format")
        
        # If we reach here, we need to force a decision
        if self.verbose:
            logging.info("Forcing decision after exhausting rollouts")
        
        # Use a random policy
        best_policy = random.choice(list(self.policy_tools.keys()))
        action_info = self._call_selected_policy(best_policy, state, available_actions)
        
        orchestrator_info = {
            "orchestrator_type": "reflection_based",
            "selected_strategy": best_policy,
            "rollouts_made": self.current_step_rollouts,
            "total_rollouts": len(self.current_step_rollouts),
            "final_reasoning": "Decision forced after max rollouts",
            "forced_decision": True
        }
        
        return action_info, orchestrator_info
    
    def _simulate_policy_rollout(
        self, 
        policy_name: str, 
        state: np.ndarray, 
        available_actions: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Simulate a multi-step rollout with a policy to get cumulative feedback.
        
        Args:
            policy_name: Name of the policy to simulate
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Dictionary containing cumulative rollout feedback
        """
        # Get the policy tool
        policy_tool = self.policy_tools[policy_name]
        
        # Store original environment state
        # original_state = self._save_env_state()
        
        # Get action from policy
        action_info = policy_tool(state, available_actions=available_actions)
        action = action_info["action"]
    
        # Use predict_step to get multi-step feedback without updating environment
        feedback = self.env.predict_step(action=action, rollout_steps=self.rollout_steps, policy=policy_tool)
        
        # Extract rollout information from feedback
        rollout_steps_info = feedback["rollout_steps_info"]
        # cumulative_reward = feedback["cumulative_reward"]
        final_info = feedback["final_info"]
        original_info = feedback["original_info"]
        
        # Format step information for LLM
        formatted_steps = []
        for step_info in rollout_steps_info:
            formatted_step = {
                "step": step_info["step"],
                "action": step_info["action"],
                "action_description": self._get_action_description(step_info["action"]),
                "questions_info": step_info["questions_info"],
                "all_failed_questions": step_info["all_failed_questions"],
                "valid_failed_questions": step_info["valid_failed_questions"],
                "cleared_questions": step_info["cleared_questions"],
                # "rolling_accuracy": step_info["rolling_accuracy"],
                "rewards_dict": step_info["rewards_dict"],
                # "step_reward": step_info["step_reward"],
                "mastery": step_info["mastery"],
            }
            formatted_steps.append(formatted_step)
        
        # Format rollout feedback for LLM
        rollout_feedback = {
            # "policy_name": policy_name,
            "rollout_steps": formatted_steps,
            # "cumulative_reward": cumulative_reward,
            # "average_reward": final_info["average_reward"],
            "final_mastery": final_info["final_mastery"],
            "original_info": original_info,
            # "final_accuracy": formatted_steps[-1]["rolling_accuracy"] if formatted_steps else 0.0,
        }
        
        return rollout_feedback
    
    def _get_action_description(self, action: int) -> str:
        """Get human-readable description of action."""
        action_descriptions = {
            0: "recommend failed questions",
            1: "recommend easy questions", 
            2: "recommend high-aptitude questions",
        }
        return action_descriptions.get(action, f"action {action}")
    
    def _save_env_state(self) -> Dict[str, Any]:
        """Save current environment state for restoration."""
        return {
            "mastery": self.env.mastery.copy(),
            "all_failed_questions": self.env.all_failed_questions.copy(),
            "cleared_questions": self.env.cleared_questions.copy(),
            "failed_questions_ratio": self.env.failed_questions_ratio.copy(),
            "skill_difficulty_accuracy": self.env.skill_difficulty_accuracy.copy(),
            "skill_difficulty_counts": self.env.skill_difficulty_counts.copy(),
            "seen_materials": self.env.seen_materials.copy(),
            "current_step": self.env.current_step,
            "aptitude_cache": self.env.aptitude_cache.copy(),
            "experience_cache": self.env.experience_cache.copy(),
            "gap_cache": self.env.gap_cache.copy(),
            "ncc_tracking": {k: v.copy() for k, v in self.env.ncc_tracking.items()},  # Deep copy ncc_tracking
        }
    
    def _restore_env_state(self, state: Dict[str, Any]):
        """Restore environment state from saved state."""
        self.env.mastery = state["mastery"]
        self.env.all_failed_questions = state["all_failed_questions"]
        self.env.cleared_questions = state["cleared_questions"]
        self.env.failed_questions_ratio = state["failed_questions_ratio"]
        self.env.skill_difficulty_accuracy = state["skill_difficulty_accuracy"]
        self.env.skill_difficulty_counts = state["skill_difficulty_counts"]
        self.env.seen_materials = state["seen_materials"]
        self.env.current_step = state["current_step"]
        self.env.aptitude_cache = state["aptitude_cache"]
        self.env.experience_cache = state["experience_cache"]
        self.env.gap_cache = state["gap_cache"]
        self.env.ncc_tracking = state["ncc_tracking"]
        
        # Reset response model to match restored state
        if self.env.response_model is not None:
            self.env.response_model.reset(self.env.mastery)
   
    def _prepare_chain_input(
        self, 
        state_info: Dict[str, Any], 
        policy_meta_info: Dict[str, Dict[str, Any]],
        rollouts_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare input for the LLM chain."""
        # Get objectives description and format instructions
        objectives_description = ReflectionBasedPromptTemplates.get_objectives_description(self.objectives)
        format_instructions = ReflectionBasedPromptTemplates.get_format_instructions("policy_selection")
        
        # Format rollouts history
        rollouts_text = self._format_rollouts_history(rollouts_history)
        
        # Get RO-specific tool signatures
        RO_tool_signatures = self._get_RO_tool_signatures()
        
        # Use base method and add reflection-specific variables
        return self._prepare_chain_input_base(
            state_info=state_info,
            policy_meta_info=policy_meta_info,
            objectives_description=objectives_description,
            format_instructions=format_instructions,
            RO_tool_signatures=RO_tool_signatures,
            rollouts_history=rollouts_text,
            max_rollouts=self.max_rollouts,
            rollout_steps=self.rollout_steps
        )
    
    def _format_rollouts_history(self, rollouts_history: List[Dict[str, Any]]) -> str:
        """Format rollouts history according to configs/reflection_tool_inout.json output schema."""
        if not rollouts_history:
            return "You have not simulated any rollouts yet."
        
        rollouts_lines = []
        for rollout in rollouts_history:
            rollout_feedback = rollout["rollout_feedback"]
            policy_name = rollout['policy_called']
            
            rollouts_lines.append(f"Rollout {rollout['round']}: Simulated {policy_name} for {self.rollout_steps} steps")
            
            # Recommended questions: List of tuples from all steps
            actions = [step_info['action_description'] for step_info in rollout_feedback['rollout_steps']]
            rollouts_lines.append(f"  - Actions: {actions}")
            
            all_recommended_questions = []
            q_idx_set = set()
            
            for step_info in rollout_feedback['rollout_steps']:
                selected_questions = step_info['questions_info']

                # Get current step mastery for difficulty calculation
                avg_mastery = np.mean(list(step_info['mastery'].values()))
                
                # Format recommended questions as tuples with difficulty based on mastery
                for i in range(len(selected_questions)):
                    q_idx = selected_questions[i]['question_id']
                    if q_idx in q_idx_set:
                        continue
                    q_idx_set.add(q_idx)
                    
                    difficulty = selected_questions[i]['scaled_difficulty']
                    
                    # Use same logic as _format_tool_calls_history
                    if avg_mastery == 1.0 and difficulty == 1.0:
                        difficulty_str = "Easy"
                    elif difficulty < avg_mastery + 0.1:
                        difficulty_str = "Easy"
                    elif difficulty < avg_mastery + 0.2:
                        difficulty_str = "Medium"
                    elif difficulty >= avg_mastery + 0.2:
                        difficulty_str = "Hard"
                    
                    all_recommended_questions.append(f"('Q{q_idx}', '{difficulty_str}')")
            
            if all_recommended_questions:
                rollouts_lines.append(f"  - Recommended {len(all_recommended_questions)} questions: [{', '.join(all_recommended_questions)}]")
            else:
                rollouts_lines.append(f"  - No recommended questions")
            
            # # report failed and cleared questions
            # original_failed_questions = rollout_feedback['original_info']['all_failed_questions']
            # original_cleared_questions = rollout_feedback['original_info']['cleared_questions']
            # current_failed_questions = step_info['all_failed_questions']
            # current_cleared_questions = step_info['cleared_questions']
            
            # # newly added failed questions
            # newly_added_failed_questions = [q_idx for q_idx in current_failed_questions if q_idx not in original_failed_questions]
            # # newly cleared questions
            # newly_cleared_questions = [q_idx for q_idx in current_cleared_questions if q_idx not in original_cleared_questions]
            
            # rollouts_lines.append(f"  - Number of failed recommended questions: {len(newly_added_failed_questions)}")
            # rollouts_lines.append(f"  - Number of failed questions that are cleared: {len(newly_cleared_questions)}")
            
            # Student feedback section
            rollouts_lines.append("  - Student feedback:")
            
            # Calculate total rewards across all steps
            total_performance_reward = 0.0
            total_gap_reward = 0.0
            total_aptitude_reward = 0.0
            
            for step_info in rollout_feedback['rollout_steps']:
                rewards_dict = step_info['rewards_dict']
                total_performance_reward += rewards_dict['performance']
                total_gap_reward += rewards_dict['gap']
                total_aptitude_reward += rewards_dict['aptitude']
            
            # Total performance reward: Float. Sum of all performance rewards
            rollouts_lines.append(f"    * Total performance reward: {total_performance_reward:.2f}")
            
            # Total gap reward: Float. Sum of all gap rewards
            rollouts_lines.append(f"    * Total gap reward: {total_gap_reward:.2f}")
            
            # Total aptitude reward: Float. Sum of all aptitude rewards
            rollouts_lines.append(f"    * Total aptitude reward: {total_aptitude_reward:.2f}")
            
            # Final mastery level: Float. Normalized to [0, 1]
            final_mastery = np.mean(list(rollout_feedback['final_mastery'].values()))
            mastery_change = "unchanged"
            original_mastery = rollout_feedback['original_info']['mastery']
            if final_mastery > original_mastery:
                mastery_change = "improved"
            elif final_mastery < original_mastery:
                mastery_change = "declined"
            
            # rollouts_lines.append(f"    * Final mastery level: {final_mastery:.2f}")
            rollouts_lines.append(f"    * Mastery change: {mastery_change}")
            
            rollouts_lines.append("")
        
        return "\n".join(rollouts_lines)
    
    def _parse_orchestrator_response(self, response) -> Dict[str, Any]:
        """Parse the orchestrator response to determine action type."""
        # Define validator function for reflection response
        def validate_reflection_response(parsed_json):
            policy = parsed_json.get("policy")
            if policy not in self.policy_tools.keys():
                logging.warning(f"Extracted policy not in available policies: {policy}")
                return False
            return True
        
        # Handle different response types
        if isinstance(response, dict):
            # Claude thinking model returns {"content": "...", "reasoning": "..."}
            if "content" in response and "reasoning" in response:
                # Extract JSON from content field
                content = response["content"]
                reasoning = response["reasoning"]
                
                # Parse JSON from content
                parsed_json = self._extract_json_from_response(
                    content, 
                    required_fields=["is_final_decision", "policy"],
                    validator_func=validate_reflection_response
                )
                
                if parsed_json:
                    result = {
                        "reasoning": reasoning,  # Use Claude's built-in reasoning
                        "is_final_decision": parsed_json["is_final_decision"],
                        "policy": parsed_json["policy"]
                    }
                    
                    logging.info(f"Successfully extracted JSON: {result}")
                    logging.info(f"Selected policy: {result['policy']}")
                    logging.info(f"Claude reasoning: {reasoning}")
                    
                    return result
            else:
                # Other dict format, try to extract JSON from the dict itself
                parsed_json = self._extract_json_from_response(
                    str(response), 
                    required_fields=["is_final_decision", "policy"],
                    validator_func=validate_reflection_response
                )
        else:
            # String response from other models
            parsed_json = self._extract_json_from_response(
                response, 
                required_fields=["is_final_decision", "policy"],
                validator_func=validate_reflection_response
            )
        
        if parsed_json:
            result = {
                "reasoning": parsed_json.get("reasoning", ""),
                "is_final_decision": parsed_json["is_final_decision"],
                "policy": parsed_json["policy"]
            }
            
            logging.info(f"Successfully extracted JSON: {result}")
            logging.info(f"Selected policy: {result['policy']}")
            logging.info(f"Reasoning: {result.get('reasoning', '')}")
            
            return result
        
        return None
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Get fallback response when parsing fails."""
        # Use first available policy and force a decision
        fallback_policy = list(self.policy_tools.keys())[0]
        return {
            "reasoning": "Fallback: Using first available policy due to parsing error",
            "is_final_decision": True,  # Force decision
            "policy": fallback_policy
        }
    
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator."""
        info = super().get_orchestrator_info()
        info["max_rollouts"] = self.max_rollouts
        info["rollout_steps"] = self.rollout_steps
        return info
    
    def _log_prompt_details(self, chain_input: Dict[str, Any]):
        """Log detailed information about the prompt being sent to the LLM."""
        if not self.verbose:
            return
            
        logging.info("=" * 80)
        logging.info("REFLECTION-BASED ORCHESTRATOR PROMPT DETAILS")
        logging.info("=" * 80)
        
        try:
            # Format and log system prompt
            system_prompt = ReflectionBasedPromptTemplates.SYSTEM_PROMPT.format(**chain_input)
            logging.info("SYSTEM PROMPT:")
            logging.info("-" * 40)
            logging.info(system_prompt)
            
            # Format and log user prompt
            user_prompt = ReflectionBasedPromptTemplates.FEW_SHOT_POLICY_SELECTION.format(**chain_input)
            logging.info("USER PROMPT:")
            logging.info("-" * 40)
            logging.info(user_prompt)
            
        except Exception as e:
            logging.error(f"Error formatting prompts for logging: {e}")
        
        logging.info("=" * 80)
        logging.info("END PROMPT DETAILS")
        logging.info("=" * 80)
