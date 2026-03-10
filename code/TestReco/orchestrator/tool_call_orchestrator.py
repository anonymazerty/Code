"""
Tool Call Orchestrator for educational recommendation system.

This orchestrator can choose to either call a policy as a tool to get feedback,
or make a direct decision about which policy to use for question recommendation.
It supports multiple rounds of tool calling before making the final decision.
"""

import logging
import json
import os
import random
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from langchain_core.runnables import Runnable
from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferWindowMemory

from tools.policies import PolicyFactory
from orchestrator.tool_call_prompts import ToolCallPromptTemplates
from orchestrator.langchain_wrapper import LangChainWrapper
from orchestrator.shared_components import BaseOrchestrator


class ToolCallOrchestrator(BaseOrchestrator):
    """
    Tool Call Orchestrator that can either call policies as tools or make direct decisions.
    
    The orchestrator can:
    1. Call a policy as a tool to get feedback (reward, correctness)
    2. Make a decision about which policy to use for question recommendation
    3. Support multiple rounds of tool calling before making the final decision
    """
    
    def __init__(
        self,
        env,
        llm,
        policy_configs: Dict[str, Dict[str, Any]],
        verbose: bool = True,
        objectives: List[str] = None,
        max_tool_calls: int = 3,
    ):
        """
        Initialize the tool call orchestrator.
        
        Args:
            env: Environment instance
            llm: Language model instance
            policy_configs: Dictionary mapping policy names to their configs
            verbose: Whether to enable verbose logging
            objectives: List of optimization objectives
            max_tool_calls: Maximum number of tool calls before forcing a decision
        """
        # Initialize base orchestrator
        super().__init__(env, llm, policy_configs, verbose, objectives)
        
        # Tool call specific initialization
        self.max_tool_calls = max_tool_calls
        
        # Create Runnable chain with appropriate prompt
        self.chain = self._create_chain()
        
        # Track tool call history for current step
        self.current_step_tool_calls = []
        
    
    def _create_chain(self) -> Runnable:
        """Create Runnable chain with system prompt and tool call prompt template."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate
        
        # Check if this is Claude (custom LLM with Claude-like response format)
        self.is_claude = hasattr(self.custom_llm, 'model_name') and 'claude' in self.custom_llm.model_name.lower()
        
        # Create chat prompt template with system and human messages
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", ToolCallPromptTemplates.SYSTEM_PROMPT.template),
            ("human", ToolCallPromptTemplates.FEW_SHOT_POLICY_SELECTION.template)
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
        Select an action using tool call orchestrator.
        
        Args:
            state: Current state
            available_actions: List of available actions (optional)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (action_info, orchestrator_info)
        """
        # Reset tool call history for new step
        self.current_step_tool_calls = []
        
        # Extract state information
        state_info = self._extract_state_info()
        
        # Get policy meta information
        policy_meta_info = self._get_policy_meta_info()
        
        # Main decision loop
        for tool_call_round in range(0, self.max_tool_calls + 1):
            if self.verbose:
                logging.info(f"Tool call round {tool_call_round}/{self.max_tool_calls + 1}")
            
            # Prepare input for LLM chain
            chain_input = self._prepare_chain_input(
                state_info, 
                policy_meta_info, 
                self.current_step_tool_calls
            )
            
            # Log the complete prompt for debugging
            if self.verbose:
                self._log_prompt_details(chain_input)
            
            # Call the LLM to decide whether to call tool or make decision
            response = self.chain.invoke(chain_input, config={"max_tokens": 4096})
            
            # Log LLM response
            if self.verbose:
                self._log_llm_response(response)
            
            # Parse the response
            parsed_response = self._parse_orchestrator_response(response)
            
            if not parsed_response["is_final_decision"]:
                # Continue to next round if we haven't reached max tool calls
                if tool_call_round >= self.max_tool_calls:
                    # Force decision after max tool calls
                    if self.verbose:
                        logging.info("Reached max tool calls, forcing decision")
                    break

                # Call the specified policy as a tool
                tool_feedback = self._call_policy_as_tool(
                    parsed_response["policy"], 
                    state, 
                    available_actions
                )
                
                # Record tool call
                self.current_step_tool_calls.append({
                    "round": tool_call_round + 1,
                    "policy_called": parsed_response["policy"],
                    "feedback": tool_feedback,
                    # "reasoning": parsed_response.get("reasoning", "")
                })
                
                if self.verbose:
                    logging.info(f"Tool call {tool_call_round + 1}: Called {parsed_response['policy']}")
                    logging.info(f"Feedback: {tool_feedback}")
                
                    
            elif parsed_response["is_final_decision"]:
                # Make final decision about which policy to use
                final_policy = parsed_response["policy"]
                if self.verbose:
                    logging.info(f"Making decision: Selected policy {final_policy}")
                
                # Call the selected policy to get action
                action_info = self._call_selected_policy(final_policy, state, available_actions)
                
                orchestrator_info = {
                    "orchestrator_type": "tool_call",
                    "selected_strategy": final_policy,
                    "tool_calls_made": self.current_step_tool_calls,
                    "total_tool_calls": len(self.current_step_tool_calls),
                    "final_reasoning": parsed_response.get("reasoning", "")
                }
                
                return action_info, orchestrator_info
            else:
                raise ValueError(f"Invalid response format")
        
        # If we reach here, we need to force a decision
        if self.verbose:
            logging.info("Forcing random decision after exhausting tool calls")
        
        # Use a random policy
        best_policy = random.choice(list(self.policy_tools.keys()))
        action_info = self._call_selected_policy(best_policy, state, available_actions)
        
        orchestrator_info = {
            "orchestrator_type": "tool_call",
            "selected_strategy": best_policy,
            "tool_calls_made": self.current_step_tool_calls,
            "total_tool_calls": len(self.current_step_tool_calls),
            "final_reasoning": "Decision forced after max tool calls",
            "forced_decision": True
        }
        
        return action_info, orchestrator_info
    
    def _call_policy_as_tool(
        self, 
        policy_name: str, 
        state: np.ndarray, 
        available_actions: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Call a policy as a tool to get feedback without updating environment state.
        
        Args:
            policy_name: Name of the policy to call
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Dictionary containing feedback information
        """
        # Get the policy tool
        policy_tool = self.policy_tools[policy_name]
        
        # Get action from policy
        action_info = policy_tool(state, available_actions=available_actions)
        
        # Use predict_step to get feedback without updating environment
        action = action_info["action"]
        
        # Get feedback using predict_step (single step, default rollout_steps=1)
        feedback = self.env.predict_step(action=action, rollout_steps=1, policy=policy_tool)
        
        # Extract information from the first (and only) step
        step_info = feedback["rollout_steps_info"][0]
        questions_info = step_info["questions_info"]
        
        # Map action number to action type description
        action_descriptions = {
            0: "recommend failed questions",
            1: "recommend easy questions", 
            2: "recommend high-aptitude questions",
        }
        action_desc = action_descriptions.get(action, f"action {action}")
        
        # Format feedback for LLM
        tool_feedback = {
            "policy_name": policy_name,
            "action": action,
            "action_description": action_desc,
            "questions_info": questions_info,
            "all_failed_questions": step_info["all_failed_questions"],
            "valid_failed_questions": step_info["valid_failed_questions"],
            "cleared_questions": step_info["cleared_questions"],
            "rewards_dict": step_info["rewards_dict"],
            "mastery": step_info["mastery"],
            "original_info": feedback["original_info"],
        }
        
        return tool_feedback
    
    
    def _prepare_chain_input(
        self, 
        state_info: Dict[str, Any], 
        policy_meta_info: Dict[str, Any],
        tool_calls_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare input for the LLM chain."""
        # Get objectives description and format instructions
        objectives_description = ToolCallPromptTemplates.get_objectives_description(self.objectives)
        format_instructions = ToolCallPromptTemplates.get_format_instructions("policy_selection")
        
        # Use base method - tool_signatures will be automatically loaded from JSON files
        base_input = self._prepare_chain_input_base(
            state_info=state_info,
            policy_meta_info=policy_meta_info,
            objectives_description=objectives_description,
            format_instructions=format_instructions,
            max_tool_calls=self.max_tool_calls
        )
        
        # Add tool call history formatting
        base_input["tool_call_history"] = self._format_tool_calls_history(tool_calls_history)
        
        return base_input
    
    def _format_tool_calls_history(self, tool_calls_history: List[Dict[str, Any]]) -> str:
        """Format tool calls history according to configs/tool_inout.json output schema."""
        if not tool_calls_history:
            return "You have not called any tools yet."
        
        tool_calls_lines = []
        for tool_call in tool_calls_history:
            feedback = tool_call["feedback"]
            tool_calls_lines.append(f"Round {tool_call['round']}: Called {tool_call['policy_called']}")
        
            # Format according to output schema from configs/tool_inout.json
            # Action: String. One of ['Failed Questions', 'Easy Questions', 'High-Aptitude Questions']
            action_descriptions = {
                0: "Failed Questions",
                1: "Easy Questions", 
                2: "High-Aptitude Questions",
            }


            action_name = action_descriptions[feedback['action']]
            tool_calls_lines.append(f"  - Action: {action_name}")

            avg_mastery = np.mean(list(feedback['mastery'].values()))
            
            # Recommended questions: List of 5 tuples. Each is a question identifier
            questions_info = feedback['questions_info']
            if questions_info:
                # Format as tuples with difficulty if available
                question_tuples = []
                for i in range(len(questions_info)):
                    difficulty = questions_info[i]['scaled_difficulty']
                    if avg_mastery == 1.0 and difficulty == 1.0:
                        difficulty = "Easy"
                    elif difficulty < avg_mastery + 0.1:
                        difficulty = "Easy"
                    elif difficulty < avg_mastery + 0.2:
                        difficulty = "Medium"
                    elif difficulty >= avg_mastery + 0.2:
                        difficulty = "Hard"
                    
                    question_tuples.append(f"('Q{questions_info[i]['question_id']}', '{difficulty}')")
                tool_calls_lines.append(f"  - Recommended questions: {question_tuples}")
            
            # report failed and cleared questions
            # original_failed_questions = feedback['original_info']['all_failed_questions']
            # original_cleared_questions = feedback['original_info']['cleared_questions']
            # current_failed_questions = feedback['all_failed_questions']
            # current_cleared_questions = feedback['cleared_questions']
            
            # newly added failed questions
            # newly_added_failed_questions = [q_idx for q_idx in current_failed_questions if q_idx not in original_failed_questions]
            # newly cleared questions
            # newly_cleared_questions = [q_idx for q_idx in current_cleared_questions if q_idx not in original_cleared_questions]
            
            # tool_calls_lines.append(f"  - Number of failed recommended questions: {len(newly_added_failed_questions)}")
            # tool_calls_lines.append(f"  - Number of failed questions that are cleared: {len(newly_cleared_questions)}")
            
            # Student feedback section
            tool_calls_lines.append("  - Student feedback:")
            
            # Performance reward: Float. Normalized to [0, 1]
            rewards_dict = feedback['rewards_dict']
            performance_reward = rewards_dict['performance']
            tool_calls_lines.append(f"    * Performance reward: {performance_reward:.2f}")
            
            # Gap reward: Float. Normalized to [0, 1]
            gap_reward = rewards_dict['gap']
            tool_calls_lines.append(f"    * Gap reward: {gap_reward:.2f}")
            
            # Aptitude reward: Float. Normalized to [0, 1]
            aptitude_reward = rewards_dict['aptitude']
            tool_calls_lines.append(f"    * Aptitude reward: {aptitude_reward:.2f}")
            
            # Mastery change: Dictionary of skill names to float deltas
            original_mastery = feedback['original_info']['mastery']
            mastery_changes = avg_mastery - original_mastery
            if mastery_changes > 0:
                mastery_change = "improved"
            elif mastery_changes < 0:
                mastery_change = "declined"
            else:
                mastery_change = "unchanged"

            # tool_calls_lines.append(f"    * Mastery level: {avg_mastery:.2f}")
            tool_calls_lines.append(f"    * Mastery change: {mastery_change}")
            
            tool_calls_lines.append("")
        
        return "\n".join(tool_calls_lines)
    
    def _parse_orchestrator_response(self, response) -> Dict[str, Any]:
        """Parse the orchestrator response to determine action type."""
        # Handle different response types
        if isinstance(response, dict):
            # Claude thinking model returns {"content": "...", "reasoning": "..."}
            if "content" in response and "reasoning" in response:
                content = response["content"]
                reasoning = response["reasoning"]
                
                # Ensure content is a string
                if not content or not isinstance(content, str) or content.strip() == "":
                    logging.warning("Empty content from Claude, using fallback")
                    return self._get_fallback_response()
                
                json_result = self._extract_and_parse_json(content)
                if json_result:
                    # Use Claude's built-in reasoning
                    json_result["reasoning"] = reasoning
                    logging.info(f"Using Claude reasoning: {reasoning}")
                    return json_result
            elif 'content' in response:
                response = response['content']
            else:
                response = str(response)
        
        # Ensure response is a string
        if not response or not isinstance(response, str) or response.strip() == "":
            logging.warning("Empty response from LLM, using fallback")
            return self._get_fallback_response()
        
        json_result = self._extract_and_parse_json(response)
        if json_result:
            return json_result
        
        # If JSON extraction fails, use fallback
        logging.warning("JSON extraction failed, using fallback response")
        return self._get_fallback_response()
    
    def _normalize_policy_name(self, name: str) -> str:
        if not name:
            return ""
        n = str(name).strip().lower()
        n = n.replace("-", "_").replace(" ", "_")

        # "policy1" -> "policy_1"
        if n.startswith("policy") and "_" not in n:
            suffix = n[len("policy"):]
            if suffix.isdigit():
                n = f"policy_{suffix}"

        return n

    def _extract_and_parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        def validate_tool_call_response(parsed_json):
            raw = parsed_json.get("policy", "")
            policy = self._normalize_policy_name(raw)

            if policy not in self.policy_tools:
                logging.warning(
                    f"Extracted policy not in available policies: raw={raw} normalized={policy}. "
                    f"Available={list(self.policy_tools.keys())}"
                )
                return False

            parsed_json["policy"] = policy
            return True

        parsed_json = self._extract_json_from_response(
            response,
            required_fields=["is_final_decision", "policy"],
            validator_func=validate_tool_call_response,
        )

        if not parsed_json:
            return None

        return {
            "reasoning": parsed_json.get("reasoning", ""),
            "is_final_decision": bool(parsed_json["is_final_decision"]),
            "policy": parsed_json["policy"],
        }

    def _get_fallback_response(self) -> Dict[str, Any]:
        # Don’t bias toward policy_0
        fallback_policy = random.choice(list(self.policy_tools.keys()))
        return {
            "reasoning": "Fallback: random policy due to parsing error",
            "is_final_decision": True,
            "policy": fallback_policy,
        }
    
    def _call_selected_policy(
        self, 
        selected_policy: str, 
        state: np.ndarray, 
        available_actions: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Call the selected policy to get action."""
        # Get the selected policy tool
        policy_tool = self.policy_tools[selected_policy]
        # Call the policy to get action
        action_info = policy_tool(state, available_actions=available_actions)
        
        return action_info
    
    
    def _log_prompt_details(self, chain_input: Dict[str, Any]):
        """Log detailed information about the prompt being sent to the LLM."""
        if not self.verbose:
            return
            
        logging.info("=" * 80)
        logging.info("TOOL CALL ORCHESTRATOR PROMPT DETAILS")
        logging.info("=" * 80)
        
        try:
            # Format and log system prompt
            system_prompt = ToolCallPromptTemplates.SYSTEM_PROMPT.format(**chain_input)
            logging.info("SYSTEM PROMPT:")
            logging.info("-" * 40)
            logging.info(system_prompt)
            
            # Format and log user prompt
            user_prompt = ToolCallPromptTemplates.FEW_SHOT_POLICY_SELECTION.format(**chain_input)
            logging.info("USER PROMPT:")
            logging.info("-" * 40)
            logging.info(user_prompt)
            
        except Exception as e:
            logging.error(f"Error formatting prompts for logging: {e}")
        
        logging.info("=" * 80)
        logging.info("END PROMPT DETAILS")
        logging.info("=" * 80)
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator."""
        info = super().get_orchestrator_info()
        info["max_tool_calls"] = self.max_tool_calls
        return info
    
    
