"""
Utilities for loading and processing benchmark datasets.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def load_json_or_jsonl(
    file_path: str, limit: Optional[int] = None
) -> List[Union[Dict, List[Dict]]]:
    items = []
    try:
        with open(file_path, "r") as f:
            try:
                # Try loading entire file as JSON first
                data = json.load(f)
                if isinstance(data, dict) and "questions" in data:
                    items = data.get("questions", [])
                elif isinstance(data, list):
                    items = data
                else:
                    # Unexpected JSON structure, fallback to JSONL
                    raise json.JSONDecodeError("Fallback to JSONL", "", 0)
            except json.JSONDecodeError:
                # Fallback to parsing as JSONL (one JSON object per line)
                f.seek(0)
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    try:
                        item = json.loads(line.strip())
                        items.append(item)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading file: {e}")
    return items


class BenchmarkLoader:
    """
    Utility class for loading and processing benchmark datasets.
    """

    @staticmethod
    def load_medmcqa(
        file_path: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load MedMCQA dataset from a QA-type JSONL file.

        Args:
            file_path: Path to the JSONL file
            limit: Optional limit on number of questions to load (for testing)

        Returns:
            List of question dictionaries
        """
        raw_items = load_json_or_jsonl(file_path, limit)
        questions = []

        for i, question in enumerate(raw_items):
            processed = {
                "id": i,
                "text": question.get("question", ""),
                "options": [
                    question.get("opa", ""),
                    question.get("opb", ""),
                    question.get("opc", ""),
                    question.get("opd", ""),
                ],
                "answer": question.get("cop", 0) - 1,
                "explanation": question.get("exp", ""),
                "subject": question.get("subject_name", ""),
                "topic": question.get("topic_name", ""),
            }
            questions.append(processed)

        return questions

    @staticmethod
    def load_math_bench(file_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load Math Bench QA dataset.

        Args:
            qa_file_path: Path to the QA.json file
            limit: Optional limit on number of questions to load (for testing)

        Returns:
            List of question dictionaries
        """
        raw_items = load_json_or_jsonl(file_path, limit)
        questions = []

        for i, item in enumerate(raw_items):
            # Handle both direct question object and {"questions": [...]}
            if "questions" in item:
                question_list = item["questions"]
            else:
                question_list = [item]

            for j, question in enumerate(question_list):
                if limit and len(questions) >= limit:
                    break
                processed = {
                    "id": question.get("id", f"math_bench_{i}_{j}"),
                    "text": question.get("question", ""),
                    "options": question.get("options", []),
                    "answer": question.get("correct_answer", -1),
                    "subject": "Mathematics",
                    "topic": question.get("topic", ""),
                    "subtopic": question.get("subtopic", ""),
                    "difficulty": question.get("level", 0),
                }
                questions.append(processed)

        return questions

    @staticmethod
    def load_sequential_trajectories(
        file_path: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load sequential trajectory data.

        Args:
            file_path: Path to the sequential_trajectories.json file
            limit: Optional limit on number of trajectories to load

        Returns:
            List of trajectory dictionaries with normalized IDs and state values
        """
        trajectories = []

        def _parse_val(val: Any) -> Optional[int]:
            """Convert numeric string to int, 'NULL' to None, else leave as is."""
            if isinstance(val, str) and val.upper() == "NULL":
                return None
            try:
                return int(val)
            except Exception:
                return val

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

                # Determine list of trajectories
                if isinstance(data, list):
                    trajectory_list = data
                else:
                    trajectory_list = data.get("trajectories", [])

                for i, traj in enumerate(trajectory_list):
                    if limit and i >= limit:
                        break

                    # Parse trajectory ID
                    raw_id = traj.get("trajectory_id", i)
                    traj_id = _parse_val(raw_id)

                    steps = []
                    for step in traj.get("steps", []):
                        raw_state = step.get("state", {})
                        raw_next = step.get("next_state", {})
                        # Normalize state fields
                        state = {
                            "topic_id": _parse_val(raw_state.get("topic_id")),
                            "subtopic_id": _parse_val(raw_state.get("subtopic_id")),
                            "question_level": _parse_val(
                                raw_state.get("question_level")
                            ),
                            "question_id": _parse_val(raw_state.get("question_id")),
                            "timestamp": raw_state.get("timestamp"),
                        }
                        # Normalize next_state fields
                        next_state = {
                            "topic_id": _parse_val(raw_next.get("topic_id")),
                            "subtopic_id": _parse_val(raw_next.get("subtopic_id")),
                            "question_level": _parse_val(
                                raw_next.get("question_level")
                            ),
                            "question_id": _parse_val(raw_next.get("question_id")),
                            "timestamp": raw_next.get("timestamp"),
                        }
                        steps.append(
                            {
                                "state": state,
                                "action": step.get("action"),
                                "reward": step.get("reward"),
                                "next_state": next_state,
                            }
                        )

                    trajectories.append(
                        {
                            "trajectory_id": traj_id,
                            "steps": steps,
                        }
                    )
        except Exception as e:
            print(f"Error loading sequential trajectory data: {e}")

        return trajectories

    @staticmethod
    def create_question_skill_map(
        question_bank: List[Dict[str, Any]],
    ) -> Tuple[List[str], Dict[int, Dict[str, Any]]]:
        """
        Extract skill information from question bank as text.

        Args:
            question_bank: List of question dictionaries

        Returns:
            Tuple of (unique skills list, mapping of question indices to skill and difficulty info)
        """
        # Create a dictionary mapping question IDs to their skills and difficulty
        question_idx_to_skills_difficulty = {}
        skills_set = set()

        # Extract question-skill relationships from question bank
        for i, question in enumerate(question_bank):
            question_idx = i
            skills = []

            if "topic" in question and question["topic"]:
                skills.append(question["topic"])
                skills_set.add(question["topic"])

            # Create dictionary with skills and difficulty for this question
            question_info = {
                "skills": skills,
                "difficulty": question["difficulty"]
            }
            
            question_idx_to_skills_difficulty[question_idx] = question_info

        # Convert set to list for consistent ordering
        unique_skills = sorted(list(skills_set))

        return unique_skills, question_idx_to_skills_difficulty


def load_benchmark_data(args):
    """
    Load benchmark data and prepare a question bank.

    Args:
        args: Command-line arguments

    Returns:
        List of question dictionaries with corresponding question_skill_map
    """
    if args.data_type == "QA":
        if args.benchmark == "medmcqa":
            args.benchmark_path = "benchmarks/MedMCQA/processed/QA_small.json"
            print(f"Loading benchmark data from {args.benchmark_path}")
            questions = BenchmarkLoader.load_medmcqa(
                args.benchmark_path, limit=args.limit
            )
        elif args.benchmark == "math_bench":
            args.benchmark_path = "benchmarks/math_bench/processed/QA.json"
            print(f"Loading benchmark data from {args.benchmark_path}")
            questions = BenchmarkLoader.load_math_bench(
                args.benchmark_path, limit=args.limit
            )
        else:
            raise ValueError(f"Unsupported benchmark: {args.benchmark}")

        print(f"Loaded {len(questions)} questions from {args.benchmark}")

    elif args.data_type == "trajectory":
        if args.benchmark == "math_bench":
            args.trajectory_path = (
                "benchmarks/math_bench/processed/sequential_trajectories.json"
            )
            print(f"Loading trajectory data from {args.trajectory_path}")
            trajectories = BenchmarkLoader.load_sequential_trajectories(
                args.trajectory_path, limit=args.limit
            )
            print(f"Loaded {len(trajectories)} trajectories")
        else:
            raise ValueError(
                f"Unsupported benchmark: {args.benchmark} for trajectory data"
            )

    # Filter questions to only include those matching target skill bundle
    if hasattr(args, 'target_skill_bundle') and args.target_skill_bundle:
        filtered_questions = []
        for question in questions:
            # Only check if question's topic is in target_skill_bundle
            if "topic" in question and question["topic"]:
                if question["topic"] in args.target_skill_bundle:
                    filtered_questions.append(question)
        
        questions = filtered_questions
        print(f"Filtered to {len(questions)} questions matching target skill bundle: {args.target_skill_bundle}")

    # Create question-skill map
    unique_skills, question_skill_map = BenchmarkLoader.create_question_skill_map(
        questions
    )

    print(f"Extracted {len(unique_skills)} unique skills")
    print(
        f"First 5 skills: {unique_skills[:5] if len(unique_skills) >= 5 else unique_skills}"
    )
    print(f"Mapped {len(question_skill_map)} questions to skills")

    return questions, question_skill_map, unique_skills
