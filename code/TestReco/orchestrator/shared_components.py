"""
Shared components for all orchestrators.

This module contains common components that are used across different orchestrator types,
including prompts, memory configurations, and utility functions.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationSummaryMemory, ConversationBufferWindowMemory

from tools.policies import PolicyFactory
from orchestrator.langchain_wrapper import LangChainWrapper


class SharedComponents:
    """Collection of shared components for orchestrators."""
    
    @staticmethod
    def get_long_term_memory_prompt() -> PromptTemplate:
        """Get the prompt template for long-term memory summarization."""
        return PromptTemplate(
            input_variables=["new_lines", "summary"],
            template="""You are an educational AI assistant that summarizes student learning interactions. 
Your task is to create a concise summary of the student's learning progress and the orchestrator's policy selection patterns.

Current summary:
{summary}

New interactions:
{new_lines}

Create a new summary that captures:
1. Student's learning patterns (mastery levels, performance trends)
2. Orchestrator's policy selection patterns (which policies work well for different student states)
3. Key insights about what strategies lead to successful learning outcomes

New summary:"""
        )
    
    @staticmethod
    def create_long_term_memory(llm, max_token_limit: int = 1024):
        """Create a ConversationSummaryMemory instance with educational-specific prompt."""
        return ConversationSummaryMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            memory_key="orchestrator_summary",
            prompt=SharedComponents.get_long_term_memory_prompt()
        )
    
    @staticmethod
    def create_short_term_memory(k: int = 3):
        """Create a ConversationBufferWindowMemory instance for recent interactions."""
        return ConversationBufferWindowMemory(
            k=k,
            return_messages=True,
            memory_key="recent_interactions"
        )
    
    @staticmethod
    def format_memory_interaction(mastery_info: str, fail_info: str, selected_strategy: str, 
                                action_type: str, correctness_summary: str, mastery_change: str,
                                r_perf: float, r_gap: float, r_apt: float) -> tuple:
        """
        Format interaction data for memory storage.
        
        Returns:
            tuple: (input_dict, output_dict) for memory save_context
        """
        input_dict = {
            "input": (
                "Student State:\n"
                f"- Mastery: {mastery_info}\n"
                f"- Number of Failed Questions: {fail_info}"
            )
        }
        
        output_dict = {
            "output": (
                f"Orchestrator Selected Policy: {selected_strategy}\n"
                f"Policy Action: {action_type}\n"
                f"Student Outcome:\n"
                f"- Correctness: {correctness_summary}\n"
                f"- Mastery Change: {mastery_change}\n"
                f"- Reward: performance={r_perf:.2f}, gap={r_gap:.2f}, aptitude={r_apt:.2f}"
            )
        }
        
        return input_dict, output_dict
    
    @staticmethod
    def format_short_term_memory_for_prompt(stm_memory) -> str:
        """
        Format short-term memory for use in prompts.
        
        Args:
            stm_memory: ConversationBufferWindowMemory instance
            
        Returns:
            str: Formatted string of recent interactions
        """
        # Get the messages from the memory
        entries = stm_memory.chat_memory.messages
        if not entries:
            return "No recent interactions available."
        
        # Limit to the most recent 3 interactions (6 messages: 3 pairs)
        # Take the last 6 messages to get the most recent 3 interactions
        recent_entries = entries[-6:] if len(entries) >= 6 else entries
        
        formatted = []
        # Process messages in pairs (user_msg, ai_msg)
        for i, pair in enumerate(zip(recent_entries[::2], recent_entries[1::2])):
            user_msg, ai_msg = pair
            formatted.append(
                f"{i+1}. {user_msg.content.strip()}\n   {ai_msg.content.strip()}"
            )
        
        return "\n\n".join(formatted)


class BaseOrchestrator:
    """
    Base orchestrator class that provides common functionality for all orchestrators.
    
    This class contains shared methods and attributes that are common across
    different orchestrator implementations (context-based, tool-call, reflection-based).
    """
    
    def __init__(
        self,
        env,
        llm,
        policy_configs: Dict[str, Dict[str, Any]],
        verbose: bool = True,
        objectives: List[str] = None,
        **kwargs
    ):
        """
        Initialize the base orchestrator.
        
        Args:
            env: Environment instance
            llm: Language model instance
            policy_configs: Dictionary mapping policy names to their configs
            verbose: Whether to enable verbose logging
            objectives: List of optimization objectives
            **kwargs: Additional arguments for specific orchestrator types
        """
        self.env = env
        # Wrap custom LLM with LangChain compatibility
        self.custom_llm = llm
        self.llm = LangChainWrapper(llm)
        self.policy_configs = policy_configs
        self.verbose = verbose
        self.objectives = objectives
        
        # Check if policy_configs is empty
        if not policy_configs:
            raise ValueError(
                "No policy configurations provided. The orchestrator requires at least one policy.\n"
                "Please provide policy folders using the --policy_folders argument.\n"
                "Example: --policy_folders path/to/policy1 path/to/policy2\n"
                "Each folder should contain:\n"
                "  - config.json (with 'agent_type' field)\n"
                "  - A trained model file (.pth or .pt)\n"
                "  - policy_level_profile.json (for meta information)"
            )
        
        # Create policy tools
        self.policy_tools = {}
        self._create_policy_tools()
        
        # Verify that at least one policy tool was created
        if not self.policy_tools:
            error_details = "\n".join(self._policy_creation_errors) if hasattr(self, '_policy_creation_errors') and self._policy_creation_errors else "No detailed error messages available"
            raise ValueError(
                f"Failed to create any policy tools from {len(policy_configs)} policy configurations.\n"
                "This may indicate:\n"
                "  - Missing or invalid config.json files in policy folders\n"
                "  - Missing 'agent_type' field in config.json\n"
                "  - Invalid policy folder paths\n"
                f"Policy configs provided: {list(policy_configs.keys())}\n\n"
                f"Detailed errors:\n{error_details}"
            )
        
        # Initialize memory components
        self.long_term_memory = SharedComponents.create_long_term_memory(self.llm)
        self.short_term_memory = SharedComponents.create_short_term_memory(k=3)
    
    def _create_policy_tools(self):
        """Create policy tools and collect their meta information."""
        errors = []  # Collect errors for reporting
        
        for policy_name, policy_info in self.policy_configs.items():
            try:
                config = policy_info.get("config")
                model_dir = policy_info.get("folder_path")
                
                if not config:
                    error_msg = f"Policy '{policy_name}': Missing 'config' in policy_info"
                    logging.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                if "agent_type" not in config:
                    error_msg = f"Policy '{policy_name}': Missing 'agent_type' in config.json"
                    logging.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                if not model_dir:
                    error_msg = f"Policy '{policy_name}': Missing 'folder_path'"
                    logging.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                if not os.path.exists(model_dir):
                    error_msg = f"Policy '{policy_name}': Folder path does not exist: {model_dir}"
                    logging.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                if self.verbose:
                    logging.info(f"Creating policy tool for '{policy_name}' (type: {config['agent_type']}, folder: {model_dir})")
                
                policy_tool = PolicyFactory.create_policy(
                    env=self.env,
                    policy_type=config["agent_type"],
                    policy_config=config,
                    model_dir=model_dir
                )
                
                self.policy_tools[policy_name] = policy_tool
                
                if self.verbose:
                    logging.info(f"Successfully created policy tool: {policy_name}")
                    
            except Exception as e:
                error_msg = f"Error creating policy tool for '{policy_name}': {str(e)}"
                logging.error(error_msg, exc_info=True)
                errors.append(error_msg)
                continue
        
        # Store errors for use in error message if needed
        self._policy_creation_errors = errors
    
    def _extract_state_info(self) -> Dict[str, Any]:
        """Extract meaningful information from the environment state."""
        state_info = {
            "mastery": self.env.mastery,
            "number_of_failed_questions": len(self.env._get_valid_failed_questions()),
        }
        return state_info
    
    def _get_policy_meta_info(self) -> Dict[str, Any]:
        """Get meta information about each policy in the new JSON format."""
        policies = []
        
        for policy_name in self.policy_tools.keys():
            # Get the policy's model directory from config
            policy_config = self.policy_configs.get(policy_name, {})
            model_dir = policy_config.get("folder_path")
            
            if not model_dir:
                raise ValueError(f"Model directory not found for policy {policy_name}")
            
            if not os.path.exists(model_dir):
                raise ValueError(f"Model directory does not exist: {model_dir}")
            
            # Read policy_level_profile.json
            profile_file = os.path.join(model_dir, "policy_level_profile.json")
            if not os.path.exists(profile_file):
                raise ValueError(f"policy_level_profile.json not found for {policy_name} in {model_dir}")
            
            with open(profile_file, 'r') as f:
                profile_data = json.load(f)
            
            # Determine optimized objective
            objectives = profile_data.get("objectives", [])
            optimized_objective = objectives[0] if objectives else "unknown"
            
            # Calculate overall strength based on average scalar reward
            avg_scalar_reward = profile_data['avg_scalar_reward']
            if avg_scalar_reward >= 0.6:
                overall_strength = "high"
            elif avg_scalar_reward >= 0.4:
                overall_strength = "medium"
            else:
                overall_strength = "low"
            
            # Calculate stability based on standard deviation
            std_scalar_reward = profile_data['std_scalar_reward']
            if std_scalar_reward < 0.05:
                stability = "stable"
            elif std_scalar_reward < 0.15:
                stability = "moderate"
            else:
                stability = "volatile"
            
            # Read behavior hint, applicability, failure modes, and example snippets from profile
            behavior_hint = profile_data.get('behavior_hint', f"Policy optimized for {optimized_objective}")
            applicability = profile_data.get('applicability', "General purpose policy.")
            failure_mode = profile_data.get('failure_modes', "May not perform optimally in all scenarios.")
            example_snippet = profile_data.get('example_snippets', ["General action pattern"])
            
            # Create policy entry
            policy_entry = {
                "name": policy_name,
                "optimized_objective": optimized_objective,
                "overall_strength": overall_strength,
                "stability": stability,
                "behavior_hint": behavior_hint,
                "applicability": applicability,
                "failure_modes": failure_mode,
                "example_snippets": example_snippet,
                "avg_scalar_reward": avg_scalar_reward,
                "std_scalar_reward": std_scalar_reward,
                "objectives": objectives
            }
            
            policies.append(policy_entry)
            
            if self.verbose:
                logging.info(f"Loaded meta info for {policy_name} from {profile_file}")
        
        return {"policies": policies}
    
    def _get_tool_signatures(self) -> str:
        """Get tool signatures with shared schema and policy-specific information.

        Robust to:
        - different working directories (Streamlit/ngrok/CLI)
        - missing configs/tool_inout.json
        - missing/partial policy_level_profile.json (no KeyError on behavior_hint)
        - model_dir being relative paths
        """
        import os
        import json
        import logging
        from pathlib import Path

        def _find_project_root(start: Path) -> Path:
            """
            Try to find a stable project root so relative paths work no matter the CWD.
            Heuristics: walk up until we find a 'configs' directory.
            """
            for p in [start, *start.parents]:
                if (p / "configs").exists():
                    return p
            # fallback: directory containing this file
            return start.parent

        project_root = _find_project_root(Path(__file__).resolve())

        schema_path = project_root / "configs" / "tool_inout.json"
        if not schema_path.exists():
            if self.verbose:
                logging.error(f"Shared schema file not found: {schema_path}")
            return json.dumps({"schema": {}, "tools": []}, indent=2)

        try:
            with schema_path.open("r", encoding="utf-8") as f:
                schema_data = json.load(f)
            shared_schema = schema_data.get("schema", {}) or {}
        except Exception as e:
            if self.verbose:
                logging.error(f"Error reading shared schema ({schema_path}): {e}")
            return json.dumps({"schema": {}, "tools": []}, indent=2)

        tools = []
        for policy_name in self.policy_tools.keys():
            policy_config = self.policy_configs.get(policy_name, {}) or {}
            model_dir = policy_config.get("folder_path")

            if not model_dir:
                if self.verbose:
                    logging.warning(f"Model directory not found for policy {policy_name} (missing folder_path).")
                continue

            model_path = Path(model_dir)
            if not model_path.is_absolute():
                # prefer resolving relative to project root
                candidate = project_root / model_path
                model_path = candidate if candidate.exists() else Path(os.path.abspath(model_dir))

            if not model_path.exists():
                if self.verbose:
                    logging.warning(f"Model directory does not exist for {policy_name}: {model_path}")
                continue

            profile_path = model_path / "policy_level_profile.json"
            if not profile_path.exists():
                if self.verbose:
                    logging.warning(f"policy_level_profile.json not found for {policy_name} in {model_path}")
                continue

            try:
                with profile_path.open("r", encoding="utf-8") as f:
                    profile_data = json.load(f) or {}

                objectives = profile_data.get("objectives") or []
                optimized_objective = objectives[0] if len(objectives) > 0 else "unknown"

                # Use .get() to avoid KeyError
                description = (profile_data.get("behavior_hint") or f"Policy optimized for {optimized_objective}")

                tools.append({
                    "name": policy_name,
                    "description": description,
                    "optimized_objective": optimized_objective,
                })

                if self.verbose:
                    logging.info(f"Loaded tool info for {policy_name} from {profile_path}")

            except Exception as e:
                if self.verbose:
                    logging.error(f"Error reading policy profile for {policy_name} ({profile_path}): {e}")
                continue

        return json.dumps({"schema": shared_schema, "tools": tools}, indent=2)


    
    def _get_RO_tool_signatures(self) -> str:
        """Get tool signatures for Reflection-based Orchestrator with shared schema and policy-specific information."""
        # Load shared schema from configs/reflection_tool_inout.json
        schema_file = "configs/reflection_tool_inout.json"
        if not os.path.exists(schema_file):
            if self.verbose:
                logging.error(f"Shared schema file not found: {schema_file}")
            return json.dumps({"schema": {}, "tools": []}, indent=2)
        
        try:
            with open(schema_file, 'r') as f:
                schema_data = json.load(f)
            shared_schema = schema_data.get("schema", {})
        except Exception as e:
            if self.verbose:
                logging.error(f"Error reading shared schema: {e}")
            return json.dumps({"schema": {}, "tools": []}, indent=2)
        
        tools = []
        
        for policy_name in self.policy_tools.keys():
            # Get the policy's model directory from config
            policy_config = self.policy_configs.get(policy_name, {})
            model_dir = policy_config.get("folder_path")
            
            if not model_dir:
                if self.verbose:
                    logging.warning(f"Model directory not found for policy {policy_name}")
                continue
            
            if not os.path.exists(model_dir):
                if self.verbose:
                    logging.warning(f"Model directory does not exist: {model_dir}")
                continue
            
            # Read policy_level_profile.json
            profile_file = os.path.join(model_dir, "policy_level_profile.json")
            if not os.path.exists(profile_file):
                if self.verbose:
                    logging.warning(f"policy_level_profile.json not found for {policy_name} in {model_dir}")
                continue
            
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                
                # Get objectives (optimized_objective)
                objectives = profile_data.get("objectives", [])
                optimized_objective = objectives[0] if objectives else "unknown"
                
                # Get behavior_hint as description
                description = (profile_data.get("behavior_hint") or f"Policy optimized for {optimized_objective}")

                
                # Create tool entry
                tool_entry = {
                    "name": policy_name,
                    "description": description,
                    "optimized_objective": optimized_objective
                }
                
                tools.append(tool_entry)
                
                if self.verbose:
                    logging.info(f"Loaded RO tool info for {policy_name} from {profile_file}")
                    
            except Exception as e:
                if self.verbose:
                    logging.error(f"Error reading policy profile for {policy_name}: {e}")
                continue
        
        # Format as JSON string with schema and tools
        result = {
            "schema": shared_schema,
            "tools": tools
        }
        
        return json.dumps(result, indent=2)
    
    def save_interaction_with_feedback(
        self, 
        state_info: Dict[str, Any], 
        selected_strategy: str, 
        action_info: Dict[str, Any], 
        next_state: Dict[str, Any], 
        reward_dict: Dict[str, float], 
        info: Dict[str, Any]
    ):
        """Save interaction to memory with environment feedback."""
        try:
            # Extract mastery information
            mastery_info = state_info['mastery']
            
            # Extract failed questions information
            fail_info = state_info['number_of_failed_questions']
            
            # Map action to action type description
            action_descriptions = {
                0: "recommend failed questions",
                1: "recommend easy questions", 
                2: "recommend high-aptitude questions",
            }
            action_type = action_descriptions.get(action_info['action'], f"action {action_info['action']}")
            
            # Extract correctness information from environment feedback
            correctness_summary = "Unknown"
            if 'rolling_accuracy' in info:
                rolling_acc = info['rolling_accuracy']
                if rolling_acc >= 0.5:
                    correctness_summary = "Mostly correct"
                else:
                    correctness_summary = "Mostly incorrect"
            
            # Extract mastery change information from environment feedback
            mastery_change = "Unknown"
            if 'mastery' in info:
                # Calculate mastery change by comparing current and next state
                current_avg_mastery = np.mean(state_info['mastery'])
                next_avg_mastery = np.mean(list(info['mastery'].values()))
                
                # Calculate average mastery change
                change = next_avg_mastery - current_avg_mastery
                if change > 0:
                    mastery_change = "Improved"
                elif change < 0:
                    mastery_change = "Declined"
                else:
                    mastery_change = "Stable"
            
            # Extract reward information from environment feedback
            r_perf = reward_dict['performance']
            r_gap = reward_dict['gap']
            r_apt = reward_dict['aptitude']
            
            # Save short-term memory with full structure
            input_dict, output_dict = SharedComponents.format_memory_interaction(
                mastery_info, fail_info, selected_strategy, action_type, 
                correctness_summary, mastery_change, r_perf, r_gap, r_apt
            )
            self.short_term_memory.save_context(input_dict, output_dict)
            
            # Save to long-term memory (summary with outcome)
            self.long_term_memory.save_context(
                {"input": f"State: student mastery={mastery_info} | number of failed questions={fail_info}"},
                {"output": f"Strategy {selected_strategy} selected, action {action_type}, rewards: performance={r_perf:.2f}, gap={r_gap:.2f}, aptitude={r_apt:.2f}"}
            )
            if self.verbose:
                logging.info(f"Saved interaction with feedback to memory: {selected_strategy} selected")
                
        except Exception as e:
            if self.verbose:
                logging.error(f"Error saving interaction with feedback to memory: {e}")
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator."""
        return {
            "orchestrator_type": self.__class__.__name__.lower().replace("orchestrator", ""),
            "objectives": self.objectives,
            "available_policies": list(self.policy_tools.keys()),
            "policy_configs": self.policy_configs,
            "memory_info": {
                "long_term_memory_available": hasattr(self, 'long_term_memory'),
                "short_term_memory_available": hasattr(self, 'short_term_memory'),
                "long_term_summary": self.long_term_memory.load_memory_variables({}).get("orchestrator_summary", "No summary available"),
                "recent_interactions_count": len(self.short_term_memory.chat_memory.messages)
            }
        }
    
    def _log_prompt_details(self, chain_input: Dict[str, Any]):
        """Log detailed information about the prompt being sent to the LLM."""
        if not self.verbose:
            return
            
        logging.info("=" * 80)
        logging.info(f"{self.__class__.__name__.upper()} PROMPT DETAILS")
        logging.info("=" * 80)
        
        try:
            # This will be implemented by subclasses
            pass
        except Exception as e:
            logging.error(f"Error formatting prompts for logging: {e}")
        
        logging.info("=" * 80)
        logging.info("END PROMPT DETAILS")
        logging.info("=" * 80)
    
    def _log_llm_response(self, response):
        """Log the LLM response for debugging."""
        if not self.verbose:
            return
            
        logging.info("\n" + "=" * 80)
        logging.info(f"{self.__class__.__name__.upper()} LLM RESPONSE")
        logging.info("=" * 80)
        
        # Log raw response
        logging.info("RAW RESPONSE:")
        logging.info("-" * 40)
        logging.info(f"Response type: {type(response)}")
        logging.info(f"Response content: {response}")
        
        # If response is a dict, log its structure
        if isinstance(response, dict):
            logging.info("\nRESPONSE STRUCTURE:")
            logging.info("-" * 40)
            for key, value in response.items():
                if isinstance(value, str) and len(value) > 200:
                    logging.info(f"  {key}: {value[:200]}... (truncated, total length: {len(value)})")
                else:
                    logging.info(f"  {key}: {value}")
        
        logging.info("=" * 80)
        logging.info("END LLM RESPONSE")
        logging.info("=" * 80)
    
    def _prepare_chain_input_base(
        self, 
        state_info: Dict[str, Any], 
        policy_meta_info: Dict[str, Any],
        objectives_description: str,
        format_instructions: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare base input for the LLM chain.
        
        Args:
            state_info: Current state information
            policy_meta_info: Policy meta information
            objectives_description: Description of objectives
            format_instructions: Format instructions for the prompt
            **kwargs: Additional arguments for specific orchestrator types
        
        Returns:
            Dict containing base input variables
        """
        # Get memory context
        long_term_context = self.long_term_memory.load_memory_variables({})
        memory_summary = long_term_context["orchestrator_summary"]
        if memory_summary == "":
            memory_summary = "No long-term memory available."
        
        # Convert recent interactions to readable format
        recent_interactions_text = SharedComponents.format_short_term_memory_for_prompt(self.short_term_memory)
        
        # Convert policy meta info to JSON string
        policy_meta_json = json.dumps(policy_meta_info, indent=2)
        
        # Get tool signatures
        tool_signatures = self._get_tool_signatures()

        # Only needed by reflection orchestrator
        orchestrator_name = self.__class__.__name__.lower()
        if "reflection" in orchestrator_name:
            RO_tool_signatures = self._get_RO_tool_signatures()
        else:
            RO_tool_signatures = json.dumps({"schema": {}, "tools": []}, indent=2)
        
        # Prepare base input dict
        input_dict = {
            # Variables for system prompt
            "target_skill_bundle": self.env.target_skill_bundle,
            "objectives": objectives_description,
            "long_term_memory": memory_summary,
            "format_instructions": format_instructions,

            # Variables for user prompt
            "mastery": state_info["mastery"],
            "number_of_failed_questions": state_info["number_of_failed_questions"],
            "policy_meta_info": policy_meta_json,
            "recent_interactions": recent_interactions_text,
            "tool_signatures": tool_signatures,
            "RO_tool_signatures": RO_tool_signatures,
        }
        
        # Add any additional variables from kwargs
        input_dict.update(kwargs)
        
        return input_dict
    
    def _normalize_llm_response(self, response) -> str:
        """
        Normalize LLM response to string format.
        
        Args:
            response: Raw response from LLM (could be dict, string, etc.)
            
        Returns:
            str: Normalized string response
        """
        # Handle response format
        if isinstance(response, dict):
            if 'content' in response:
                response = response['content']
            elif 'text' in response:
                response = response['text']
            else:
                # If there is no content field, try other possible fields
                response = str(response)
        
        # Handle LangChain AIMessage objects
        if hasattr(response, 'content'):
            response = response.content
        
        # Ensure response is a string
        if not response or not isinstance(response, str) or response.strip() == "":
            # Handle empty response
            logging.warning("Empty response from LLM")
            return ""
        
        return response.strip()
    
    def _extract_json_from_response(self, response: str, required_fields: List[str], validator_func=None) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from response using common patterns.
        
        Args:
            response: String response from LLM
            required_fields: List of required fields in the JSON
            validator_func: Optional function to validate the parsed JSON
            
        Returns:
            Dict containing the parsed JSON, or None if extraction/validation fails
        """
        import re
        import json
        
        # Try multiple JSON extraction patterns
        patterns = [
            r'\{[^{}]*\}',  # Simple JSON object (single line)
            r'\{[\s\S]*?\}',  # Multi-line JSON object
            r'```json\s*(\{[\s\S]*?\})\s*```',  # Markdown code block
            r'```\s*(\{[\s\S]*?\})\s*```',  # Generic code block
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                # If pattern has groups, use the first group
                json_str = match if isinstance(match, str) else match[0]
                try:
                    parsed_json = json.loads(json_str)
                    
                    # Check if all required fields are present
                    if all(field in parsed_json for field in required_fields):
                        # Apply custom validation if provided
                        if validator_func and not validator_func(parsed_json):
                            continue
                        
                        return parsed_json
                            
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _call_selected_policy(
        self, 
        selected_policy: str, 
        state: np.ndarray, 
        available_actions: Optional[List[int]] = None
    ) -> Any:
        """Call the selected policy to get action."""
        # Get the selected policy tool
        policy_tool = self.policy_tools[selected_policy]
        # Call the policy to get action
        action_info = policy_tool(state, available_actions=available_actions)
        
        return action_info
