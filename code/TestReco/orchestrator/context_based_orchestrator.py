"""
Context-based orchestrator using LLM reasoning to select strategies based on meta information.

This orchestrator provides meta information about each policy (success rate, gap recovery ability, 
Q-value distribution, representative trajectories) to the LLM, which then selects the most 
appropriate strategy based on the current student state and policy performance.
"""

import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.runnables import Runnable
from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
import os
import json

from tools.policies import PolicyFactory
from orchestrator.context_based_prompts import ContextBasedPromptTemplates
from orchestrator.langchain_wrapper import LangChainWrapper
from orchestrator.shared_components import SharedComponents, BaseOrchestrator


class ContextBasedOrchestrator(BaseOrchestrator):
    """
    Context-based orchestrator that uses LLM reasoning to select strategies based on meta information.
    
    The LLM receives:
    - Current student state
    - Meta information about each policy (success rate, gap recovery, Q-value distribution, etc.)
    
    The LLM selects a strategy name, then that strategy generates the action.
    """
    
    def __init__(
        self,
        env,
        llm,
        policy_configs: Dict[str, Dict[str, Any]],
        verbose: bool = True,
        objectives: List[str] = None,
        rubric_path: str = "configs/rubric.json",
    ):
        """
        Initialize the context-based orchestrator.
        
        Args:
            env: Environment instance
            llm: Language model instance
            policy_configs: Dictionary mapping policy names to their configs
            verbose: Whether to enable verbose logging
            objectives: List of optimization objectives
            rubric_path: Path to rubric file for policy classification
        """
        # Initialize base orchestrator
        super().__init__(env, llm, policy_configs, verbose, objectives)
        
        # Context-based specific initialization
        self.rubric_path = rubric_path
        self.rubric = self._load_rubric()
        
        # Create Runnable chain with appropriate prompt
        self.chain = self._create_chain()
    
    def _load_rubric(self) -> Dict[str, Any]:
        """Load rubric from file."""
        if not os.path.exists(self.rubric_path):
            raise FileNotFoundError(f"Rubric file not found at {self.rubric_path}")
            
        with open(self.rubric_path, 'r') as f:
            rubric = json.load(f)
        logging.info(f"Loaded rubric from {self.rubric_path}")
        return rubric
    
    
    def _create_chain(self) -> Runnable:
        """Create Runnable chain with system prompt and policy selection prompt template."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate
        
        # Check if this is Claude (custom LLM with Claude-like response format)
        self.is_claude = hasattr(self.custom_llm, 'model_name') and 'claude' in self.custom_llm.model_name.lower()
        
        # Create chat prompt template with system and human messages
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", ContextBasedPromptTemplates.SYSTEM_PROMPT.template),
            ("human", ContextBasedPromptTemplates.FEW_SHOT_POLICY_SELECTION.template)
        ])
        
        # Create Runnable chain using prompt | llm pattern
        return chat_prompt | self.llm
    
    def _get_policy_meta_info(self) -> Dict[str, Any]:
        """Get meta information about each policy with rubric information."""
        # Get base policy meta info from parent class
        base_meta_info = super()._get_policy_meta_info()
        
        # Add rubric information for context-based orchestrator
        base_meta_info["rubric"] = self.rubric
        
        return base_meta_info
    
    def select_action(
        self, 
        state: np.ndarray, 
        available_actions: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action using context-based strategy selection.
        
        Args:
            state: Current state
            available_actions: List of available actions (optional)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (action, info_dict)
        """
        # Extract state information
        state_info = self._extract_state_info()
        
        # Get policy meta information
        policy_meta_info = self._get_policy_meta_info()
        
        # Prepare input for LLM chain
        chain_input = self._prepare_chain_input(state_info, policy_meta_info)
        # logging.info(f"STATE INFO: {state_info}")
        
        # Log the complete prompt for debugging
        self._log_prompt_details(chain_input)
        
        # Call the LLM to select a strategy
        # logging.info("Calling LLM to select strategy...")
        response = self.chain.invoke(chain_input, config={"max_tokens": 4096})
        
        # Log LLM response
        self._log_llm_response(response)
        
        # Parse the selected strategy
        selected_strategy = self._parse_strategy_selection(response)
        
        # Call the selected strategy to get action
        action_info = self._call_selected_policy(selected_strategy, state, available_actions)
        
        info = {
            "orchestrator_type": "context_based",
            "selected_strategy": selected_strategy,
            # "llm_response": response,
            # "state_info": state_info,
            # "policy_meta_info": policy_meta_info
        }
        
        return action_info, info
    
        
    def _prepare_chain_input(
        self, 
        state_info: Dict[str, Any], 
        policy_meta_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare input for the LLM chain with policy meta information in JSON format."""
        # Get objectives description and format instructions
        objectives_description = ContextBasedPromptTemplates.get_objectives_description(self.objectives)
        format_instructions = ContextBasedPromptTemplates.get_format_instructions("policy_selection")
        
        # Use base method and add context-based specific variables
        return self._prepare_chain_input_base(
            state_info=state_info,
            policy_meta_info=policy_meta_info,
            objectives_description=objectives_description,
            format_instructions=format_instructions,
        )
    
    def _parse_strategy_selection(self, response):
        """Parse strategy selection from LLM response."""
        # Check if policy_tools is empty (shouldn't happen if initialization succeeded, but safety check)
        if not self.policy_tools:
            raise ValueError(
                "No policy tools available. Cannot select strategy.\n"
                "This should not happen if orchestrator was initialized correctly.\n"
                f"Policy configs provided: {list(self.policy_configs.keys())}\n"
                "Please check that policy folders were provided and contain valid config.json files."
            )
        
        # Handle different response types
        if isinstance(response, dict):
            # Claude thinking model returns {"content": "...", "reasoning": "..."}
            if "content" in response and "reasoning" in response:
                content = response["content"]
                reasoning = response["reasoning"]
                
                # Log Claude's reasoning
                logging.info(f"Claude reasoning: {reasoning}")
                
                if not content or not isinstance(content, str) or content.strip() == "":
                    logging.warning("Empty content from Claude, using fallback")
                    return list(self.policy_tools.keys())[0]
                
                json_result = self._extract_and_parse_json(content)
                if json_result:
                    return json_result
                # If extraction fails, fall through to fallback
                
            else:
                # Other dict format, normalize using base class method
                normalized_response = self._normalize_llm_response(response)
        else:
            # String response, normalize using base class method
            normalized_response = self._normalize_llm_response(response)
        
        if not normalized_response:
            logging.warning("Empty response from LLM, using fallback")
            logging.warning(f"Available policies: {list(self.policy_tools.keys())}")
            return list(self.policy_tools.keys())[0]
        
        json_result = self._extract_and_parse_json(normalized_response)
        if json_result:
            return json_result
        
        # If JSON extraction fails, use fallback
        logging.warning("JSON extraction failed, using fallback policy")
        logging.warning(f"Response that failed to parse: {normalized_response[:500]}")  # Log first 500 chars
        logging.warning(f"Available policies: {list(self.policy_tools.keys())}")
        return list(self.policy_tools.keys())[0]
    
    def _extract_and_parse_json(self, response: str) -> Optional[str]:
        """Extract JSON from response and parse it to get selected policy."""
        # Define validator function for policy selection
        def validate_policy_selection(parsed_json):
            selected_policy = parsed_json.get("selected_policy")
            if selected_policy in self.policy_tools.keys():
                logging.info(f"Successfully extracted policy from JSON: {selected_policy}")
                return True
            else:
                logging.warning(f"Extracted policy not in available policies: {selected_policy}")
                return False
        
        # Use base class method to extract JSON
        parsed_json = self._extract_json_from_response(
            response, 
            required_fields=["selected_policy"],
            validator_func=validate_policy_selection
        )
        
        return parsed_json["selected_policy"] if parsed_json else None

    def _log_prompt_details(self, chain_input: Dict[str, Any]):
        """Log detailed information about the prompt being sent to the LLM."""
        if not self.verbose:
            return
            
        logging.info("=" * 80)
        logging.info("CONTEXT-BASED ORCHESTRATOR PROMPT DETAILS")
        logging.info("=" * 80)
        
        try:
            # Format and log system prompt
            system_prompt = ContextBasedPromptTemplates.SYSTEM_PROMPT.format(**chain_input)
            logging.info("SYSTEM PROMPT:")
            logging.info("-" * 40)
            logging.info(system_prompt)
            
            # Format and log user prompt
            user_prompt = ContextBasedPromptTemplates.FEW_SHOT_POLICY_SELECTION.format(**chain_input)
            logging.info("USER PROMPT:")
            logging.info("-" * 40)
            logging.info(user_prompt)
            
        except Exception as e:
            logging.error(f"Error formatting prompts for logging: {e}")
        
        logging.info("=" * 80)
        logging.info("END PROMPT DETAILS")
        logging.info("=" * 80)