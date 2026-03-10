import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from agents.llm_agents.prompts import ClaudeThinkingPromptTemplates, PromptTemplates
from generators.factory import model_factory
from generators.model import Message
from utils.question_recommender import QuestionRecommender
from utils.recommendation_recorder import RecommendationRecorder


class SimpleLLMAgent:
    """
    Simple LLM-based agent for recommending questions based on history.

    This agent does not use MCTS and instead directly recommends the next question
    based on the student's history and available questions.
    """

    def __init__(
        self,
        num_objectives: int,
        model_name: str = "gpt-4",
        weights: Optional[np.ndarray] = None,
        question_bank_path: Optional[str] = None,
        record_recommendations: bool = False,
        target_skill_bundle: Optional[List[str]] = None,
        prompt_type: str = "zero_shot",
        objectives: Optional[List[str]] = None,
    ):
        """
        Initialize the SimpleLLMAgent.

        Args:
            action_dim: Number of possible actions (questions) - for backward compatibility
            num_objectives: Number of objectives
            model_name: Name of the LLM model to use
            weights: Weights for each objective
            question_bank_path: Path to the question bank JSON file
            record_recommendations: Whether to record recommendation information
            target_skill_bundle: List of target skills to focus on during learning
            prompt_type: Type of prompt template to use ("zero_shot", "few_shot", or "few_shot_cot")
            objectives: List of objectives to optimize (e.g., ["performance", "gap"])
        """
        self.num_objectives = num_objectives
        self.weights = (
            weights if weights is not None else np.ones(num_objectives) / num_objectives
        )
        self.target_skill_bundle = (
            target_skill_bundle if target_skill_bundle else ["default_skill"]
        )
        self.objectives = objectives or ["performance"]
        
        # Define skill-difficulty action space (consistent with RL agents)
        self.skills = self.target_skill_bundle
        self.num_skills = len(self.skills)
        
        # Default difficulties - will be updated if environment is provided later
        self.difficulties = [1, 2, 3]  # Default fallback
        self.num_difficulties = len(self.difficulties)
        
        # Action space: (skill, difficulty) combinations only
        self.action_dim = self.num_skills * self.num_difficulties

        # Initialize LLM model
        self.model_name = model_name
        self.model = model_factory(self.model_name)

        # Initialize question recommender
        self.question_recommender = (
            QuestionRecommender(question_bank_path) if question_bank_path else None
        )

        # Initialize recommendation recorder
        self.recommendation_recorder = RecommendationRecorder(
            enabled=record_recommendations
        )

        # Initialize prompt templates
        self.prompts = PromptTemplates()
        self.claude_prompts = ClaudeThinkingPromptTemplates()

        # Check if we're using Claude thinking model
        self.is_claude_thinking = self.model_name == "claude-3.7-sonnet-thinking"

        # Select prompt template based on type
        if self.is_claude_thinking:
            if prompt_type == "zero_shot":
                self.prompt_template = self.claude_prompts.ZERO_SHOT_RECOMMENDATION
            elif prompt_type == "few_shot":
                self.prompt_template = self.claude_prompts.FEW_SHOT_RECOMMENDATION
            elif prompt_type == "few_shot_cot":
                self.prompt_template = self.claude_prompts.FEW_SHOT_COT_RECOMMENDATION
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
        else:
            if prompt_type == "zero_shot":
                self.prompt_template = self.prompts.ZERO_SHOT_RECOMMENDATION
            elif prompt_type == "few_shot":
                self.prompt_template = self.prompts.FEW_SHOT_RECOMMENDATION
            elif prompt_type == "few_shot_cot":
                self.prompt_template = self.prompts.FEW_SHOT_COT_RECOMMENDATION
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

    def _skill_difficulty_to_action(self, skill: str, difficulty: float) -> int:
        """Convert (skill, difficulty) pair to action index."""
        skill_idx = self.skills.index(skill)
        difficulty_idx = self.difficulties.index(difficulty)
        return skill_idx * self.num_difficulties + difficulty_idx

    @property
    def select_action(self):
        """Get the select_action method with recommendation recording."""
        return self.recommendation_recorder(self._select_action)

    def _select_action(self, env) -> Dict[str, Any]:
        """
        Select an action (question) based on the current state.

        Args:
            env: Environment (used for extracting question information)

        Returns:
            Dictionary containing:
            - question_idx: Selected question ID (matching question)
            - action_idx: Selected action index in action space
            - recommendation_info: Dictionary with recommendation details
        """
        # Extract state information from environment
        mastery_summary, skill_difficulty_accuracy_str, failed_questions_ratio_str = (
            self._extract_state_info(env)
        )

        # First, get difficulty and skills recommendation from LLM
        recommendation = self._get_recommendation(
            mastery_summary,
            skill_difficulty_accuracy_str,
            failed_questions_ratio_str,
        )

        # Prepare recommendation info for recording
        recommendation_info = {
            # "recommendation_prompt": self._get_last_prompt(),
            "recommendation_response": self._get_last_response(),
            "recommended_difficulty": recommendation.get("difficulty"),
            "recommended_skill": recommendation.get("skill"),
            "reasoning": recommendation.get("reasoning"),
        }

        # Convert LLM recommendation to action index
        skill = recommendation.get("skill")
        difficulty = recommendation.get("difficulty")
        
        if skill is None or difficulty is None:
            raise ValueError("Skill or difficulty is None, cannot find matching questions")
        
        # Convert (skill, difficulty) to action index
        action_idx = self._skill_difficulty_to_action(skill, difficulty)
        
        # Find matching questions based on recommendation
        if self.question_recommender:
            # Get failed questions from environment
            failed_questions = list(env.failed_question.keys()) if hasattr(env, 'failed_question') else []
            
            matching_questions, matchedFlag = (
                self.question_recommender.find_matching_questions(
                    difficulty=difficulty,
                    skill=skill,
                    failed_questions=failed_questions,
                )
            )
            
            if matchedFlag:
                return {
                    "question_idx": matching_questions,  # actual question ID
                    "action_idx": action_idx,  # action space index
                    "recommendation_info": recommendation_info,
                }

        # If no question recommender or no matching question, return None
        return {
            "question_idx": None,
            "action_idx": None,
            "recommendation_info": recommendation_info,
        }

    def get_recommendation_recordings(self) -> List[Dict[str, Any]]:
        """Get all recorded recommendations."""
        return self.recommendation_recorder.get_recordings()

    def _extract_state_info(self, env) -> Tuple[str, str, str]:
        """
        Extract readable state information from the state vector.

        Args:
            env: Environment instance

        Returns:
            Tuple of (mastery_summary, skill_difficulty_accuracy_str, failed_questions_ratio_str)
        """
        # Get mastery levels for target skills
        mastery_levels = {
            skill: round(float(env.mastery[env.skill_to_idx[skill]]), 2)
            for skill in env.target_skill_bundle
        }
        mastery_summary = ", ".join([f"{skill}: {level}" for skill, level in mastery_levels.items()])
        
        # Convert skill-difficulty accuracy matrix to string format
        skill_difficulty_accuracy_str = ""
        if np.sum(env.skill_difficulty_accuracy) == 0:
            skill_difficulty_accuracy_str = "No skill-difficulty accuracy data available."
        else:
            skill_difficulty_accuracy_str = "Skill-Difficulty Accuracy:\n"
            for i, skill in enumerate(env.target_skill_bundle):
                if i < env.skill_difficulty_accuracy.shape[0]:
                    accuracies = env.skill_difficulty_accuracy[i]
                    counts = env.skill_difficulty_counts[i] if hasattr(env, 'skill_difficulty_counts') else np.zeros_like(accuracies)
                    skill_difficulty_accuracy_str += f"  - {skill}: "
                    accuracy_info = []
                    for diff_idx, accuracy in enumerate(accuracies):
                        if counts[diff_idx] > 0:  # Only show if we have data
                            accuracy_info.append(f"diff{diff_idx+1}: {accuracy:.1%} ({counts[diff_idx]} questions)")
                    if accuracy_info:
                        skill_difficulty_accuracy_str += ", ".join(accuracy_info) + "\n"
                    else:
                        skill_difficulty_accuracy_str += "no data available\n"
        
        # Convert failed questions ratio matrix to string format
        failed_questions_ratio_str = ""
        
        # Check if there are any failed questions
        if np.sum(env.failed_questions_ratio) == 0:
            failed_questions_ratio_str = "No failed questions to review."
        else:
            failed_questions_ratio_str = "Failed questions summary:\n"
            for i, skill in enumerate(env.target_skill_bundle):
                if i < env.failed_questions_ratio.shape[0]:
                    ratios = env.failed_questions_ratio[i]
                    # Only show skills that have failed questions
                    if np.sum(ratios) > 0:
                        failed_questions_ratio_str += f"  - {skill}: "
                        difficulty_info = []
                        for diff_idx, ratio in enumerate(ratios):
                            if ratio > 0:
                                difficulty_info.append(f"{ratio:.1%} at difficulty {diff_idx + 1}")
                        failed_questions_ratio_str += ", ".join(difficulty_info) + "\n"
            
            # If no skills have failed questions (edge case)
            if failed_questions_ratio_str == "Failed questions summary:\n":
                failed_questions_ratio_str = "No failed questions to review."

        return (
            mastery_summary,
            skill_difficulty_accuracy_str,
            failed_questions_ratio_str,
        )

    def _format_available_questions(self, env) -> str:
        """
        Format the available questions for the LLM prompt.

        Args:
            env: Environment instance with question information

        Returns:
            Formatted string describing available questions
        """
        if self.question_recommender:
            # Use question recommender
            questions_str = self.question_recommender.get_recommendation(env)
        else:
            # Use skill mapping from environment
            questions_str = ""
            for i in range(env.num_questions):
                skills = ", ".join(env.question_skills_difficulty_map[i])
                questions_str += f"Question {i}: Skills tested: {skills}\n"
            return questions_str

    def _get_objectives_description(self) -> str:
        """Generate a description of the objectives based on the current objectives list."""
        objective_descriptions = {
            "performance": "Performance: Four of the recent five questions have been answered correctly.",
            "gap": "Gap: There is no failed questions to review.",
        }

        # Generate numbered list of objectives
        descriptions = []
        for i, obj in enumerate(self.objectives, 1):
            if obj in objective_descriptions:
                desc = objective_descriptions[obj]
                descriptions.append(f"{desc}")

        return "\n".join(descriptions)

    def _get_recommendation(
        self,
        mastery_summary: str,
        skill_difficulty_accuracy_str: str,
        failed_questions_ratio_str: str,
    ) -> Dict[str, Any]:
        """
        Get difficulty and skills recommendation from LLM.

        Returns:
            Dictionary containing:
            - difficulty (int): Recommended difficulty level
            - skill (str): Recommended skill to test
            - reasoning (str): LLM's reasoning for the recommendation
        """
        # Generate objectives description
        objectives_description = self._get_objectives_description()

        # Prepare LLM prompt
        prompt = self.prompt_template.format(
            mastery_summary=mastery_summary,
            skill_difficulty_accuracy_str=skill_difficulty_accuracy_str,
            target_skill_bundle=self.target_skill_bundle,
            objectives_description=objectives_description,
            failed_questions_ratio=failed_questions_ratio_str,
            format_instructions=(
                self.claude_prompts.claude_recommendation_parser.get_format_instructions()
                if self.is_claude_thinking
                else self.prompts.recommendation_parser.get_format_instructions()
            ),
        )

        # Query LLM for recommendation
        messages = [
            Message(
                role="system",
                content=self.prompts.SYSTEM_PROMPT.format(
                    target_skill_bundle=self.target_skill_bundle,
                    objectives=objectives_description,
                ),
            ),
            Message(role="user", content=prompt),
        ]

        # Log the prompt
        logging.info("=== SimpleLLMAgent Action Selection ===")
        # logging.info(f"Agent: {self.model_name}")
        logging.info(f"Mastery summary: {mastery_summary}")
        logging.info(f"Skill-difficulty accuracy: {skill_difficulty_accuracy_str}")

        # Log the response
        if self.is_claude_thinking:
            # max token 2048
            response = self.model.generate_chat(messages, max_tokens=2048)
            response_content = self._clean_response(response["content"])
        else:
            response = self.model.generate_chat(messages)
            response = self._clean_response(response)

        if self.is_claude_thinking:
            # For Claude thinking, get content and reasoning from the response dictionary
            try:
                # Parse the JSON response using Claude parser
                parsed_response = (
                    self.claude_prompts.claude_recommendation_parser.parse(
                        response_content
                    )
                )
                # Get reasoning from the response dictionary
                reasoning = response["reasoning"]
                logging.info(f"Reasoning: {reasoning}")
                logging.info(f"Parsed response: {parsed_response}")
                return {
                    "difficulty": parsed_response.difficulty,
                    "skill": parsed_response.skill,
                    "reasoning": reasoning,
                }
            except Exception as e:
                logging.warning(f"Warning: Failed to parse Claude response: {e}")
                return {
                    "difficulty": None,
                    "skill": None,
                    "reasoning": "Failed to parse recommendation",
                }
        else:
            # For non-Claude models, parse the response using LangChain's parser
            try:
                parsed_response = self.prompts.recommendation_parser.parse(response)
                logging.info(f"Parsed response: {parsed_response}")
                return {
                    "difficulty": parsed_response.difficulty,
                    "skill": parsed_response.skill,
                    "reasoning": parsed_response.reasoning,
                }
            except Exception as e:
                logging.warning(f"Warning: Failed to parse LLM response: {e}")
                return {
                    "difficulty": None,
                    "skill": None,
                    "reasoning": "Failed to parse recommendation",
                }
        logging.info("================================")

    def _parse_evaluation_scores(self, response: str) -> np.ndarray:
        """
        Parse the LLM's evaluation response to extract scores for each objective.

        Args:
            response: LLM's evaluation response

        Returns:
            Array of scores for each objective
        """
        try:
            parsed_response = self.prompts.evaluation_parser.parse(response)
            scores = np.zeros(self.num_objectives)

            # Extract scores from parsed response
            scores[0] = parsed_response.performance["score"]
            scores[1] = parsed_response.gap["score"]

            return scores
        except Exception as e:
            logging.warning(f"Warning: Failed to parse evaluation response: {e}")
            return np.ones(self.num_objectives) * 0.5  # Default middle value

    def _get_last_prompt(self) -> Optional[str]:
        """Get the last prompt from the model."""
        return self.model.last_prompt if hasattr(self.model, "last_prompt") else None

    def _get_last_response(self) -> Optional[str]:
        """Get the last response from the model."""
        return (
            self.model.last_response if hasattr(self.model, "last_response") else None
        )

    def _clean_response(self, response: str) -> str:
        """Clean the response to extract only the JSON part."""
        try:
            # Find the first occurrence of '{' and the last occurrence of '}'
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return response[start:end]
        except Exception:
            pass
        return response
