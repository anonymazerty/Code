import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class QuestionRecommender:
    """
    Question recommender that manages question database and handles question recommendations.
    """

    def __init__(self, question_bank_path: str):
        """
        Initialize the question recommender.

        Args:
            question_bank_path: Path to the question bank JSON file
        """
        self.questions = self._load_question_bank(question_bank_path)
        self.topic_to_questions = self._index_questions_by_topic()
        self.subtopic_to_questions = self._index_questions_by_subtopic()
        self.difficulty_to_questions = self._index_questions_by_difficulty()

    def _load_question_bank(self, file_path: str) -> List[Dict[str, Any]]:
        """Load questions from JSON file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "questions" in data:
                    return data["questions"]
                return data
        except Exception as e:
            logging.error(f"Error loading question bank: {e}")
            return []

    def _index_questions_by_topic(self) -> Dict[str, List[int]]:
        """Create an index mapping topics to question indices."""
        topic_index = {}
        for i, question in enumerate(self.questions):
            topic = question.get("topic", "")
            if topic:
                if topic not in topic_index:
                    topic_index[topic] = []
                topic_index[topic].append(i)
        return topic_index

    def _index_questions_by_subtopic(self) -> Dict[str, List[int]]:
        """Create an index mapping subtopics to question indices."""
        subtopic_index = {}
        for i, question in enumerate(self.questions):
            subtopic = question.get("subtopic", "")
            if subtopic:
                if subtopic not in subtopic_index:
                    subtopic_index[subtopic] = []
                subtopic_index[subtopic].append(i)
        return subtopic_index

    def _index_questions_by_difficulty(self) -> Dict[int, List[int]]:
        """Create an index mapping difficulty levels to question indices."""
        difficulty_index = {}
        for i, question in enumerate(self.questions):
            difficulty = int(question.get("level", 1))
            if difficulty not in difficulty_index:
                difficulty_index[difficulty] = []
            difficulty_index[difficulty].append(i)
        return difficulty_index

    def find_matching_questions(
        self,
        difficulty: int,
        skill: str,
        exclude_questions: Optional[List[int]] = None,
        failed_questions: Optional[List[int]] = None,
        max_results: int = 50,
    ) -> Tuple[int, bool]:
        """
        Find questions that match the given difficulty and skill.

        Args:
            difficulty: Target difficulty level
            skill: Skill (topic/subtopic) to test
            exclude_questions: List of question indices to exclude
            failed_questions: List of failed question indices to prioritize
            max_results: Maximum number of questions to return

        Returns:
            Tuple of (question_index, is_matching)
        """
        exclude_questions = exclude_questions or []
        failed_questions = failed_questions or []
        matching_questions = set()

        # Find questions matching the difficulty
        difficulty_matches = set(self.difficulty_to_questions.get(difficulty, []))

        # Find questions matching any of the skills
        skill_matches = set()
        # Try topic match
        topic_matches = set(self.topic_to_questions.get(skill, []))
        # Try subtopic match
        subtopic_matches = set(self.subtopic_to_questions.get(skill, []))
        # Combine matches
        skill_matches.update(topic_matches, subtopic_matches)

        # Find questions that match both difficulty and skills
        matching_questions = difficulty_matches.intersection(skill_matches)

        # Remove excluded questions
        matching_questions = matching_questions - set(exclude_questions)
        
        # Convert to list
        if matching_questions:
            matching_questions_list = list(matching_questions)
            
            # Check if there are failed questions in the matching set
            failed_in_matching = [q for q in matching_questions_list if q in failed_questions]
            
            if failed_in_matching:
                # Prioritize failed questions
                selected_question = np.random.choice(failed_in_matching)
                logging.info(f"Prioritizing failed question {selected_question} for difficulty {difficulty} and skill {skill}")
                return selected_question, True
            else:
                # No failed questions in matching set, choose randomly
                selected_question = matching_questions_list[np.random.randint(0, len(matching_questions_list))]
                return selected_question, True
        else:
            logging.warning(
                f"No matching questions found satisfying difficulty {difficulty} and skill {skill}, we will return a random question either by difficulty or skill"
            )
            return (
                np.random.randint(0, len(self.questions)),
                False,
            )

    def get_question_info(self, question_idx: int) -> Dict[str, Any]:
        """Get full information for a question by its index."""
        if 0 <= question_idx < len(self.questions):
            return self.questions[question_idx]
        return {}
