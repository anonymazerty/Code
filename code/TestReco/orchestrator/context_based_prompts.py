"""
Prompt templates for context-based orchestrator.

This module contains the prompt templates used by the context-based orchestrator
for direct action selection and policy orchestration tasks.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class DirectActionOutput(BaseModel):
    """Output schema for direct action selection."""
    
    reasoning: str = Field(description="Detailed explanation of why this action was selected")
    action: int = Field(description="Selected action number (0 to max_actions-1)")


class PolicySelectionOutput(BaseModel):
    """Output schema for policy selection."""
    
    reasoning: str = Field(description="Detailed explanation of why this policy was selected")
    selected_policy: str = Field(description="Name of the selected policy to call")


class ClaudePolicySelectionOutput(BaseModel):
    """Output schema for Claude policy selection - no explicit reasoning required."""
    
    selected_policy: str = Field(description="Name of the selected policy to call")


class ContextBasedPromptTemplates:
    """Collection of prompt templates for context-based orchestrator."""
    
    # Initialize output parsers
    direct_action_parser = PydanticOutputParser(pydantic_object=DirectActionOutput)
    policy_selection_parser = PydanticOutputParser(pydantic_object=PolicySelectionOutput)
    claude_policy_selection_parser = PydanticOutputParser(pydantic_object=ClaudePolicySelectionOutput)
    
    # System prompt for educational orchestrator
    SYSTEM_PROMPT = PromptTemplate(
        input_variables=["target_skill_bundle", "objectives", "format_instructions", "long_term_memory"],
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
- Always select exactly one policy.
- Always provide a clear reasoning before selection.
- Always output in strict JSON format: {format_instructions}

## Instructions
- Base your decision on the provided context (student state, policy information, recent interactions, and long-term summary of past interactions).
- When objectives conflict, trade off carefully and prefer consistent high-performing policies.
- Adapt your choice based on recent history of student learning progression.
"""
    )
    
    # Few-shot policy selection prompt with JSON format
    FEW_SHOT_POLICY_SELECTION = PromptTemplate(
        input_variables=[
            "mastery",
            # "accuracy_per_skill_difficulty",
            # "failed_questions_summary",
            "number_of_failed_questions",
            "policy_meta_info",
            "recent_interactions",
        ],
        template="""
## Query
Select one best policy to use for recommending the next set of questions to the student, based on the current student state and the available pre-trained policies.

## Available Policies
{policy_meta_info}

## Student State:
- Skill Mastery Levels: {mastery}
- Number of Failed Questions: {number_of_failed_questions}

## Recent Orchestrator Interactions (Most recent last)
{recent_interactions}
""" 
    )
    
    # Claude-specific policy selection prompt with JSON format
    CLAUDE_FEW_SHOT_POLICY_SELECTION = PromptTemplate(
        input_variables=[
            "mastery",
            # "accuracy_per_skill_difficulty",
            # "failed_questions_summary",
            "number_of_failed_questions",
            "policy_meta_info",
            "recent_interactions",
        ],
        template="""
## Query        
Select one best policy to use for recommending the next set of questions to the student, based on the current student state and the available pre-trained policies.

## Available Policies
{policy_meta_info}

## Student State:
- Skill Mastery Levels: {mastery}
- Number of Failed Questions: {number_of_failed_questions}

## Recent Orchestrator Interactions (Most recent last)
{recent_interactions}
"""
    )
    
    @classmethod
    def get_format_instructions(cls, output_type: str = "direct_action", for_claude: bool = False) -> str:
        """Get format instructions for the output parser."""
        if for_claude:
            if output_type == "direct_action":
                return cls.direct_action_parser.get_format_instructions()
            else:  # policy_selection
                return cls.claude_policy_selection_parser.get_format_instructions()
        else:
            if output_type == "direct_action":
                return cls.direct_action_parser.get_format_instructions()
            else:  # policy_selection
                return cls.policy_selection_parser.get_format_instructions()
    
    # @classmethod
    # def get_claude_format_instructions(cls, output_type: str = "policy_selection") -> str:
    #     """Get format instructions for Claude (without reasoning requirement)."""
    #     if output_type == "policy_selection":
    #         return cls.claude_policy_selection_parser.get_format_instructions()
    #     elif output_type == "direct_action":
    #         return cls.direct_action_parser.get_format_instructions()
    #     else:
    #         raise ValueError(f"Unsupported output type for Claude: {output_type}")
    
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