"""
Prompt templates for tool call orchestrator.

This module contains the prompt templates used by the tool call orchestrator
for deciding whether to call a policy as a tool or make a direct decision.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class PolicySelectionOutput(BaseModel):
    """Output schema for tool call or decision selection."""
    
    reasoning: str = Field(
        description="Detailed explanation of why this action was chosen"
    )
    
    is_final_decision: bool = Field(
        description="false if you decide to call a policy as tool, true if you decide to make a final decision based on the tool calls"
    )
    
    policy: str = Field(
        description="Name of the policy to use (either for tool call or final decision)"
    )

class ClaudePolicySelectionOutput(BaseModel):
    """Output schema for Claude model tool call or decision selection."""
    
    is_final_decision: bool = Field(
        description="false if you decide to call a policy as tool, true if you decide to make a final decision based on the tool calls"
    )
    
    policy: str = Field(
        description="Name of the policy to use (either for tool call or final decision)"
    )


class ToolCallPromptTemplates:
    """Collection of prompt templates for tool call orchestrator."""
    
    # Initialize output parsers
    policy_selection_parser = PydanticOutputParser(pydantic_object=PolicySelectionOutput)
    claude_policy_selection_parser = PydanticOutputParser(pydantic_object=ClaudePolicySelectionOutput)
    
    # System prompt for tool call orchestrator
    SYSTEM_PROMPT = PromptTemplate(
        input_variables=["target_skill_bundle", "objectives", "format_instructions", "long_term_memory", "max_tool_calls"],
        template="""You are an expert educational recommendation orchestrator. 
Your role is to select exactly one pre-trained policy which can then recommend the next set of questions about the target skill bundle for a student. The target skill bundle is: {target_skill_bundle}.

Each pre-trained policy is a fixed function that maps a student state to an action.
- Input: a student state, including mastery levels, accuracy per skill × difficulty, and failed question summaries. 
- Output: one of three possible action types, each recommending 5 questions:
  - Action 0: Failed Questions — Recommend previously failed or closely related questions to review mistakes and close knowledge gaps.
  - Action 1: Easy Questions — Recommend familiar, low-difficulty questions that the student is very likely to answer correctly to build performance and confidence.
  - Action 2: High-Aptitude Questions — Recommend the most challenging questions (difficulty > mastery), prioritizing the largest difficulty–mastery gap to maximize challenge.

Policies are static decision functions with fixed optimization objectives and performance statistics. They cannot change their behavior during orchestration.

## Objectives
{objectives}

## Interaction History Summary
{long_term_memory}

## Guardrails
- Always select exactly one policy as either a tool call to get student feedback or final decision.
- Avoid calling the same policy multiple times during tool calls because it will not be able to provide new information.
- Always provide a clear reasoning.
- You have at maximum {max_tool_calls} tool calls to get student feedback of the tools.
- Always make a final decision once you have reached the maximum number of tool calls.
- Always output in strict JSON format: {format_instructions}

## Instructions
- Base your decision on the provided context (student state, available tools, tool call history, recent interactions, and long-term summary of past interactions).
- When objectives conflict, trade off carefully and prefer consistent high-performing policies.
- Adapt your choice based on the the student feedback (rewards per objective, recommended questions' difficulty level and mastery changes) from the tool calls.
- Once you get enough information from the tool calls, make a final decision about which pre-trained policy to use for question recommendation.
"""
    )
    
    # Few-shot policy selection prompt
    FEW_SHOT_POLICY_SELECTION = PromptTemplate(
        input_variables=[
            "mastery",
            "number_of_failed_questions",
            "recent_interactions",
            "tool_signatures",
            "tool_call_history"
        ],
        template="""
## Query
Select one best policy to use for recommending the next set of questions to the student, based on the current student state, the available tools, and the tool call history.

## Available Tools
{tool_signatures}

## Student State:
- Skill Mastery Levels: {mastery}
- Number of Failed Questions: {number_of_failed_questions}

## Tool Call History (Most recent last)
{tool_call_history}

## Recent Orchestrator Interactions (Most recent last)
{recent_interactions}
""" 
    )
    
    # Claude-specific few-shot policy selection prompt
    CLAUDE_FEW_SHOT_POLICY_SELECTION = PromptTemplate(
        input_variables=[
            "mastery",
            "number_of_failed_questions",
            "recent_interactions",
            "tool_signatures",
            "tool_call_history"
        ],
        template="""
## Query        
Select one best policy to use for recommending the next set of questions to the student, based on the current student state, the available tools, and the tool call history.

## Available Tools
{tool_signatures}

## Student State:
- Skill Mastery Levels: {mastery}
- Number of Failed Questions: {number_of_failed_questions}

## Tool Call History (Most recent last)
{tool_call_history}

## Recent Orchestrator Interactions (Most recent last)
{recent_interactions}
"""
    )
    
    @classmethod
    def get_objectives_description(cls, objectives: List[str]) -> str:
        """Get formatted objectives description."""
        descriptions = []
        for obj in objectives:
            if obj == "performance":
                descriptions.append("Performance: Build a solid foundation by practicing questions aligned with current skill.")
            elif obj == "gap":
                descriptions.append("Gap: Recover from past mistakes by reviewing failed questions.")
            elif obj == "aptitude":
                descriptions.append("Aptitude: Stretch potential by practicing the most challenging questions above mastery.")
            else:
                raise ValueError(f"Invalid objective: {obj}")
        
        return "\n".join(descriptions)
    
    @classmethod
    def get_claude_prompt(cls, output_type: str = "policy_selection") -> Tuple[PromptTemplate, PromptTemplate]:
        """Get Claude-specific prompts (system + user) for the given output type."""
        if output_type == "policy_selection":
            # For Claude, use the pre-defined Claude-specific prompt
            return cls.SYSTEM_PROMPT, cls.CLAUDE_FEW_SHOT_POLICY_SELECTION
        else:
            raise ValueError(f"Unsupported output type for Claude: {output_type}")
    
    @classmethod
    def get_regular_prompt(cls, output_type: str = "policy_selection") -> Tuple[PromptTemplate, PromptTemplate]:
        """Get regular prompts (system + user) for the given output type."""
        if output_type == "policy_selection":
            return cls.SYSTEM_PROMPT, cls.FEW_SHOT_POLICY_SELECTION
        else:
            raise ValueError(f"Unsupported output type for regular models: {output_type}")
    
    @classmethod
    def get_format_instructions(cls, output_type: str = "policy_selection", for_claude: bool = False) -> str:
        """Get format instructions for the output parser."""
        if for_claude:
            if output_type == "policy_selection":
                return cls.claude_policy_selection_parser.get_format_instructions()
            else:
                return cls.policy_selection_parser.get_format_instructions()
        else:
            if output_type == "policy_selection":
                return cls.policy_selection_parser.get_format_instructions()
            else:
                return cls.claude_policy_selection_parser.get_format_instructions()
    
