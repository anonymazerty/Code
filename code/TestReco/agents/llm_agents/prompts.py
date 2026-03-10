"""
Prompt templates for LLM agents.

This module contains the prompt templates used by the LLM agents
for various tasks like question recommendation, simulation,
evaluation, etc.
"""

from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class RecommendationOutput(BaseModel):
    """Output schema for question recommendation."""

    reasoning: str = Field(description="Detailed explanation of the recommendation")
    difficulty: int = Field(
        description="Difficulty level from 1-3",
        ge=1,
        le=3,
    )
    skill: str = Field(
        description="Skill to test",
    )


class EvaluationOutput(BaseModel):
    """Output schema for reward evaluation."""

    performance: Dict[str, Any] = Field(description="Performance evaluation")
    gap: Dict[str, Any] = Field(description="Gap evaluation")


class ClaudeRecommendationOutput(BaseModel):
    """Output schema for question recommendation without reasoning field."""

    difficulty: int = Field(
        description="Difficulty level from 1-3",
        ge=1,
        le=3,
    )
    skill: str = Field(
        description="Skill to test",
    )


class PromptTemplates:
    """Collection of prompt templates for LLM agents."""

    # Initialize output parsers
    recommendation_parser = PydanticOutputParser(pydantic_object=RecommendationOutput)
    evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationOutput)

    # System prompt for educational AI assistant
    SYSTEM_PROMPT = PromptTemplate(
        input_variables=["target_skill_bundle", "objectives"],
        template="""You are an expert educational AI assistant tasked with recommending the next question for a student.
Your goal is to optimize the following objectives:
{objectives}

The student is currently focusing on the following skill bundle: {target_skill_bundle}

Your recommendation should be based on:
- The student's performance history
- Their current mastery levels
- The target skill bundle
- The optimization objectives

Best Practices:
- Focus only on recommending skills in the target skill bundle: {target_skill_bundle}.
- Avoid assuming mastery is high unless all values in the provided mastery levels are shown to be above 0.8.
- Prioritize low-mastery skills within the target bundle.
- Evaluate based on actual attempted questions when only one question is attempted in the history.
- Consider only "correct" or "incorrect" entries in performance history.
- Avoid repeatedly recommending the same (skill, difficulty) pair unless justified by the objectives.

You must always respond with a strict JSON object specifying the next question (skill and difficulty), supported by clear pedagogical reasoning grounded in the student's current data.""",
    )

    # Zero-shot prompt for question recommendation
    ZERO_SHOT_RECOMMENDATION = PromptTemplate(
        input_variables=[
            "mastery_summary",
            "skill_difficulty_accuracy_str",
            "target_skill_bundle",
            "objectives_description",
            "failed_questions_ratio",
            "format_instructions",
        ],
        template="""Your task is to recommend the best next question's skill and difficulty level satisfying the optimization objectives.
## Optimization Objectives:
{objectives_description}
        
## Current Student State
Mastery levels: {mastery_summary}
Skill-Difficulty Accuracy: {skill_difficulty_accuracy_str}
Failed questions to review: {failed_questions_ratio}

## Response Format:
{format_instructions}

IMPORTANT: Output MUST be a valid JSON object only. Do not include any headers, formatting, or additional explanation outside the JSON.""",
    )

    # Few-shot prompt for question recommendation
    FEW_SHOT_RECOMMENDATION = PromptTemplate(
        input_variables=[
            "mastery_summary",
            "skill_difficulty_accuracy_str",
            "target_skill_bundle",
            "objectives_description",
            "failed_questions_ratio",
            "format_instructions",
        ],
        template="""Your task is to recommend the best next question's skill and difficulty level satisfying the optimization objectives.
## Optimization Objectives:
{objectives_description}

## Current Student State
Mastery levels: {mastery_summary}
Skill-Difficulty Accuracy: {skill_difficulty_accuracy_str}
Failed questions to review: {failed_questions_ratio}

## Response Format:
{format_instructions}

## Examples:

### Example 1:
Student State:
- Mastery levels: Calculus: 0.87
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Calculus: diff1: 100.0% (2 questions), diff2: 100.0% (2 questions), diff3: 100.0% (1 questions)
- Failed questions to review: No failed questions to review.
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.

Response:
{{
    "reasoning": "Since all the last five questions have been answered correctly, the student has achieved the performance objective. The student has shown mastery above 0.8 in Calculus, so we should continue with more challenging questions to maintain engagement.",
    "difficulty": 3,
    "skill": "Calculus"
}}

### Example 2:
Student State:
- Mastery levels: Linear Algebra: 0.55, Calculus: 0.91
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Linear Algebra: diff1: 50.0% (2 questions)
  - Calculus: diff2: 100.0% (1 questions)
- Failed questions to review: Linear Algebra: 100.0% at difficulty 1
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.
Gap: There is no failed questions to review.

Response:
{{
    "reasoning": "Three out of the last five questions are incorrect, so the performance goal is not achieved. There are failed questions about linear algebra at difficulty 1, so the gap objective is not achieved. To improve both performance and gap, recommend Linear Algebra questions with difficulty 1, for which the student will also have higher chance to perform correctly.",
    "difficulty": 1,
    "skill": "Linear Algebra"
}}

### Example 3:
Student State:
- Mastery levels: Linear Algebra: 0.92, Calculus: 0.80
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Linear Algebra: diff1: 100.0% (2 questions), diff2: 50.0% (1 questions)
  - Calculus: diff1: 100.0% (1 questions), diff2: 100.0% (1 questions)
- Failed questions to review: Linear Algebra: 100.0% at difficulty 2
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.
Gap: There is no failed questions to review.

Response:
{{
    "reasoning": "Student show 4 correct answers in last 5 questions so that the performance objective is achieved. But there is failed question in difficulty 2 about linear algebra, so the gap objective is not achieved. To improve gap, recommend Calculus questions with difficulty 2, which is the same in the failed linear algebra question.",
    "difficulty": 2,
    "skill": "Calculus"
}}

IMPORTANT: Output MUST be a valid JSON object only. Do not include any headers, formatting, or additional explanation outside the JSON.""",
    )

    # Few-shot with CoT prompt for question recommendation
    FEW_SHOT_COT_RECOMMENDATION = PromptTemplate(
        input_variables=[
            "mastery_summary",
            "skill_difficulty_accuracy_str",
            "target_skill_bundle",
            "objectives_description",
            "failed_questions_ratio",
            "format_instructions",
        ],
        template="""Your task is to recommend the best next question's skill and difficulty level satisfying the optimization objectives.
## Optimization Objectives:
{objectives_description}

## Current Student State
Mastery levels: {mastery_summary}
Skill-Difficulty Accuracy: {skill_difficulty_accuracy_str}
Failed questions to review: {failed_questions_ratio}

## Response Format:
{format_instructions}

## Examples:

### Example 1:
Student State:
- Mastery levels: Calculus: 0.87
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Calculus: diff1: 100.0% (2 questions), diff2: 100.0% (2 questions), diff3: 100.0% (1 questions)
- Failed questions to review: No failed questions to review.
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.

Step-by-step reasoning:
1. Performance: Performance is an optimization objective, and 5/5 recent questions were correct.
2. Gap: Gap is not an optimization objective, so we should not consider it.
3. Since all objectives are satisfied, we should continue with more challenging questions to maintain engagement.
4. Mastery: Calculus at 0.87 (> 0.8), so we can recommend higher difficulty questions.
5. We should recommend Calculus with difficulty 3 to challenge the student further.

Recommendation:
{{
  "difficulty": 3,
  "skill": "Calculus"
}}

### Example 2:
Student State:
- Mastery levels: Linear Algebra: 0.55, Calculus: 0.91
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Linear Algebra: diff1: 50.0% (2 questions)
  - Calculus: diff2: 100.0% (1 questions)
- Failed questions to review: Linear Algebra: 100.0% at difficulty 1
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.
Gap: There is no failed questions to review.

Step-by-step reasoning:
1. Performance: Performance is an optimization objective. In performance history, only 3 out of 5 recent questions were correct.
2. Gap: Gap is an optimization objective, and there is failed questions to review. 100% failed question ratio in Linear Algebra difficulty 1, so we should recommend Linear Algebra questions with difficulty 1.
3. Since both objectives are not satisfied, we should keep recommending questions.
4. We should recommend Linear Algebra questions with difficulty 1 because it is the same difficulty as the failed question and it is easy enough to answer so that the performance can be kept.

Recommendation:
{{
  "difficulty": 1,
  "skill": "Linear Algebra"
}}

IMPORTANT: Output MUST be a valid JSON object only. Do not include any headers, formatting, or additional explanation outside the JSON.""",
    )


class ClaudeThinkingPromptTemplates(PromptTemplates):
    """Prompt templates specifically for Claude thinking model."""

    # Initialize output parser without reasoning field
    claude_recommendation_parser = PydanticOutputParser(
        pydantic_object=ClaudeRecommendationOutput
    )

    # Zero-shot prompt for question recommendation with Claude thinking
    ZERO_SHOT_RECOMMENDATION = PromptTemplate(
        input_variables=[
            "mastery_summary",
            "skill_difficulty_accuracy_str",
            "target_skill_bundle",
            "objectives_description",
            "failed_questions_ratio",
            "format_instructions",
        ],
        template="""Your task is to recommend the best next question's skill and difficulty level satisfying the optimization objectives.
## Optimization Objectives:
{objectives_description}
        
## Current Student State
Mastery levels: {mastery_summary}
Skill-Difficulty Accuracy: {skill_difficulty_accuracy_str}
Failed questions to review: {failed_questions_ratio}

## Response Format:
{format_instructions}

IMPORTANT: Output MUST be a valid JSON object only. Do not include any headers, formatting, or additional explanation outside the JSON.""",
    )

    # Few-shot prompt for question recommendation with Claude thinking
    FEW_SHOT_RECOMMENDATION = PromptTemplate(
        input_variables=[
            "mastery_summary",
            "skill_difficulty_accuracy_str",
            "target_skill_bundle",
            "objectives_description",
            "failed_questions_ratio",
            "format_instructions",
        ],
        template="""Your task is to recommend the best next question's skill and difficulty level satisfying the optimization objectives.

## Current Student State
Mastery levels: {mastery_summary}
Skill-Difficulty Accuracy: {skill_difficulty_accuracy_str}
Failed questions to review: {failed_questions_ratio}

## Instruction:
You must recommend the best next question's skill and difficulty level from the target skill bundle.

## Response Format:
{format_instructions}

## Examples:

### Example 1:
Student State:
- Mastery levels: Calculus: 0.87
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Calculus: diff1: 100.0% (2 questions), diff2: 100.0% (2 questions), diff3: 100.0% (1 questions)
- Failed questions to review: No failed questions to review.
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.
Gap: There is no failed questions to review.

Response:
{{
    "difficulty": 3,
    "skill": "Calculus"
}}

### Example 2:
Student State:
- Mastery levels: Linear Algebra: 0.55, Calculus: 0.91
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Linear Algebra: diff1: 50.0% (2 questions)
  - Calculus: diff2: 100.0% (1 questions)
- Failed questions to review: Linear Algebra: 50.0% at difficulty 1
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.
Gap: There is no failed questions to review.

Response:
{{
    "difficulty": 1,
    "skill": "Linear Algebra"
}}

IMPORTANT: Output MUST be a valid JSON object only. Do not include any headers, formatting, or additional explanation outside the JSON.""",
    )

    # Few-shot with CoT prompt for question recommendation with Claude thinking
    FEW_SHOT_COT_RECOMMENDATION = PromptTemplate(
        input_variables=[
            "mastery_summary",
            "skill_difficulty_accuracy_str",
            "target_skill_bundle",
            "objectives_description",
            "failed_questions_ratio",
            "format_instructions",
        ],
        template="""Your task is to recommend the best next question's skill and difficulty level satisfying the optimization objectives.
## Optimization Objectives:
{objectives_description}

## Current Student State
Mastery levels: {mastery_summary}
Skill-Difficulty Accuracy: {skill_difficulty_accuracy_str}
Failed questions to review: {failed_questions_ratio}

## Response Format:
{format_instructions}

## Examples:

### Example 1:
Student State:
- Mastery levels: Calculus: 0.87
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Calculus: diff1: 100.0% (2 questions), diff2: 100.0% (2 questions), diff3: 100.0% (1 questions)
- Failed questions to review: No failed questions to review.
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.

Step-by-step reasoning:
1. Performance: Performance is an optimization objective, and 5/5 recent questions were correct.
2. Gap: Gap is not an optimization objective, so we should not consider it.
3. Since all objectives are satisfied, we should continue with more challenging questions to maintain engagement.
4. Mastery: Calculus at 0.87 (> 0.8), so we can recommend higher difficulty questions.
5. We should recommend Calculus with difficulty 3 to challenge the student further.

Recommendation:
{{
  "difficulty": 3,
  "skill": "Calculus"
}}

### Example 2:
Student State:
- Mastery levels: Linear Algebra: 0.55, Calculus: 0.91
- Skill-Difficulty Accuracy: Skill-Difficulty Accuracy:
  - Linear Algebra: diff1: 50.0% (2 questions)
  - Calculus: diff2: 100.0% (1 questions)
- Failed questions to review: Linear Algebra: 100.0% at difficulty 1
- Optimization Objectives:
Performance: Four of the recent five questions have been answered correctly.
Gap: There is no failed questions to review.

Step-by-step reasoning:
1. Performance: Performance is an optimization objective. In performance history, only 3 out of 5 recent questions were correct.
2. Gap: Gap is an optimization objective, and there is failed questions to review. 100% failed question ratio in Linear Algebra difficulty 1, so we should recommend Linear Algebra questions with difficulty 1.
3. Since both objectives are not satisfied, we should keep recommending questions.
4. We should recommend Linear Algebra questions with difficulty 1 because it is the same difficulty as the failed question and it is easy enough to answer so that the performance can be kept.

Recommendation:
{{
  "difficulty": 1,
  "skill": "Linear Algebra"
}}

IMPORTANT: Output MUST be a valid JSON object only. Do not include any headers, formatting, or additional explanation outside the JSON.""",
    )
