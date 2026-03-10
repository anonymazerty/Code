from __future__ import annotations

import os
import time
import json
import random
from typing import Dict, List, Any, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from quizcomp_llm_study.config import MODEL_TIERS, SURVEY_QUESTIONS
from quizcomp_llm_study.interactive_prompt import (
    PromptVariant,
    build_interactive_composition_system_prompt,
    build_interactive_composition_user_prompt,
    build_interactive_survey_prompt,
    build_quiz_analysis_prompt,
)


class InteractiveLLMTeacherAgent:
    """
    Simulates a teacher INTERACTIVELY using the QuizComp composition system.
    """

    def __init__(
        self,
        initial_topic_distribution: List[float],
        initial_difficulty_distribution: List[float],
        initial_num_mcqs: int,
        model_path: str,
        alfa_value: float = 0.5,
        api_base_url: str = "http://localhost:8000",
        seed: int = None,
        prompt_variant: PromptVariant = "detailed",
    ):
        self.initial_topic_distribution = initial_topic_distribution
        self.initial_difficulty_distribution = initial_difficulty_distribution
        self.initial_num_mcqs = initial_num_mcqs

        # Track current preferences in parameters 
        self.current_topic_distribution = list(initial_topic_distribution)
        self.current_difficulty_distribution = list(initial_difficulty_distribution)

        self.model_path = model_path
        self.alfa_value = alfa_value
        self.api_base_url = api_base_url
        self.composition_api_url = f"{api_base_url}/compose/quiz"
        self.generation_api_url = f"{api_base_url}/gen/quizzes"

        self.prompt_variant: PromptVariant = prompt_variant

        self.data_uuid: Optional[str] = None

        self.seed = seed or random.randint(0, 1000000)
        self.rng = random.Random(self.seed)

        self.interaction_history: List[Dict[str, Any]] = []
        self.current_quiz: Optional[Dict[str, Any]] = None
        self.current_quiz_id: Optional[int] = None
        self.iterations_count: int = 0

        # Token and call tracking
        self.total_tokens = 0
        self.total_llm_calls = 0

        from quizcomp_llm_study.config import LLM_PROVIDER, OLLAMA_BASE_URL

        self.llm_provider = LLM_PROVIDER
        model_shortname = MODEL_TIERS["all"][0]

        if self.llm_provider == "ollama":
            self.model_name = model_shortname
            self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            print(f"Initialized Interactive LLM Teacher Agent with Ollama model: {self.model_name}")

        elif self.llm_provider == "openai":
            self.model_name = model_shortname
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            print(f"Initialized Interactive LLM Teacher Agent with OpenAI model: {self.model_name}")

        else:
            model_mapping = {
                "gpt-5": "openai/gpt-4o",
                "chatgpt-4o": "openai/chatgpt-4o-latest",
                "chatgpt-4o-mini": "openai/gpt-4o-mini",
            }
            self.model_name = model_mapping.get(model_shortname, model_shortname)
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
            print(f"Initialized Interactive LLM Teacher Agent with OpenRouter model: {self.model_name}")

        print(f"Interaction prompt variant: {self.prompt_variant}")

    def start_composition_session(self, max_iterations: int = 5) -> Dict[str, Any]:
        print(f"\n{'=' * 70}")
        print("STARTING INTERACTIVE COMPOSITION SESSION")
        print(f"Model: {self.model_name}")
        print(f"Prompt variant: {self.prompt_variant}")
        print(f"Initial topics: {self.initial_topic_distribution}")
        print(f"Initial difficulty: {self.initial_difficulty_distribution}")
        print(f"Max attempts: {max_iterations}")
        print(f"{'=' * 70}\n")

        session_start_time = time.time()

        self._generate_quiz_universe()

        # Attempt #1
        self._make_composition_request(
            topic_distribution=self.current_topic_distribution,
            difficulty_distribution=self.current_difficulty_distribution,
            mode="fresh",
            start_quiz_id=None,
        )

        for _ in range(1, max_iterations):
            if self.current_quiz is None:
                print("WARNING: current_quiz is None, stopping session.")
                break

            current_attempt = self.iterations_count
            decision = self._get_agent_decision(current_attempt=current_attempt)

            if decision.get("action") == "accept":
                print(f"\n✓ Agent accepted quiz after {self.iterations_count} attempts")
                break

            mode = decision.get("mode", "improve")

            if mode == "improve":
                self._make_composition_request(
                    topic_distribution=self.current_topic_distribution,
                    difficulty_distribution=self.current_difficulty_distribution,
                    mode="improve",
                    start_quiz_id=self.current_quiz_id,
                )
            else:
                self.current_topic_distribution = list(decision["new_topic_distribution"])
                self.current_difficulty_distribution = list(decision["new_difficulty_distribution"])

                self._make_composition_request(
                    topic_distribution=self.current_topic_distribution,
                    difficulty_distribution=self.current_difficulty_distribution,
                    mode="fresh",
                    start_quiz_id=None,
                )

        session_duration = time.time() - session_start_time

        session_summary = {
            "iterations": self.iterations_count,
            "final_quiz": self.current_quiz,
            "final_quiz_id": self.current_quiz_id,
            "interaction_history": self.interaction_history,
            "duration_seconds": session_duration,
            "model_name": self.model_name,
            "prompt_variant": self.prompt_variant,
        }

        print(f"\n{'=' * 70}")
        print("COMPOSITION SESSION COMPLETE")
        print(f"Duration: {session_duration:.1f}s")
        print(f"Attempts: {self.iterations_count}")
        print(f"Final Quiz ID: {self.current_quiz_id}")
        print(f"{'=' * 70}\n")

        return session_summary

    def complete_survey_after_session(self, session_summary: Dict[str, Any]) -> Dict[str, int]:
        print("\n Completing Survey Based on Interactive Experience ")

        trajectory_text = self._build_trajectory_text(session_summary)

        final_quiz = session_summary.get("final_quiz") or {}
        final_match = float(final_quiz.get("targetMatch", 0.0))

        modes = [str(h.get("mode", "")).lower() for h in session_summary.get("interaction_history", [])]
        used_fresh = any(m == "fresh" for m in modes)
        used_improve = any(m == "improve" for m in modes)

        system_prompt, user_prompt = build_interactive_survey_prompt(
            trajectory_summary=trajectory_text,
            initial_topics=self.initial_topic_distribution,
            initial_difficulty=self.initial_difficulty_distribution,
            num_questions=self.initial_num_mcqs,
            num_attempts=int(session_summary.get("iterations", 0)),
            total_time=float(session_summary.get("duration_seconds", 0.0)),
            final_match=final_match,
            used_fresh=used_fresh,
            used_improve=used_improve,
        )

        response_text = self._call_llm(system_prompt, user_prompt, timeout=120)
        survey_responses = self._parse_survey_responses(response_text)

        print("Survey responses:")
        for key, value in survey_responses.items():
            if key != "reasoning":
                print(f"  {key}: {value}/5")

        return survey_responses

    def _make_composition_request(
        self,
        topic_distribution: List[float],
        difficulty_distribution: List[float],
        mode: str,
        start_quiz_id: Optional[int],
    ) -> Dict[str, Any]:
        self.iterations_count += 1
        attempt_start_time = time.time()

        print(f"\n Attempt {self.iterations_count} ({mode} mode) ")
        print(f"Topic distribution: {topic_distribution}")
        print(f"Difficulty distribution: {difficulty_distribution}")
        if start_quiz_id is not None:
            print(f"Starting from quiz ID: {start_quiz_id}")

        request_payload = {
            "dataUUID": self.data_uuid,
            "teacherTopic": topic_distribution,
            "teacherLevel": difficulty_distribution,
            "pathToModel": self.model_path,
            "alfaValue": self.alfa_value,
            "startQuizId": start_quiz_id,
        }

        response = requests.post(self.composition_api_url, json=request_payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        quiz_data = result.get("best_quiz", result)
        quiz_id = quiz_data.get("quiz_id")
        target_match = float(quiz_data.get("targetMatch", 0.0))

        attempt_duration = time.time() - attempt_start_time
        print(f" Quiz generated: ID={quiz_id}, Match={target_match:.2%}, Time={attempt_duration:.2f}s")

        self.current_quiz = quiz_data
        self.current_quiz_id = quiz_id

        interaction_record = {
            "iteration": self.iterations_count,
            "mode": mode,
            "topic_distribution": topic_distribution,
            "difficulty_distribution": difficulty_distribution,
            "start_quiz_id": start_quiz_id,
            "returned_quiz_id": quiz_id,
            "target_match": target_match,
            "api_time_s": attempt_duration,
            "quiz_data": quiz_data,
        }
        self.interaction_history.append(interaction_record)

        return quiz_data

    def _get_agent_decision(self, current_attempt: int) -> Dict[str, Any]:
        print("\n  Asking agent for decision...")

        system_prompt = build_interactive_composition_system_prompt(variant=self.prompt_variant)
        user_prompt = build_interactive_composition_user_prompt(
            current_quiz=self.current_quiz or {},
            interaction_history=self.interaction_history,
            current_attempt=current_attempt,
            variant=self.prompt_variant,
        )

        response_text = self._call_llm(system_prompt, user_prompt, timeout=90)
        decision = self._parse_decision(response_text, current_attempt)

        print(f"  Decision: {decision.get('action')}")
        if decision.get("action") == "compose":
            print(f"  Mode: {decision.get('mode')}")

        return decision

    def _call_llm(self, system_prompt: str, user_prompt: str, timeout: int = 60) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(model=self.model_name, messages=messages)
        
        # Track tokens and calls
        self.total_llm_calls += 1
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += response.usage.total_tokens
        else:
            # Estimate tokens if not provided (rough estimate: ~4 chars per token)
            response_text = response.choices[0].message.content
            estimated_tokens = (len(system_prompt) + len(user_prompt) + len(response_text)) // 4
            self.total_tokens += estimated_tokens
        
        return response.choices[0].message.content

    def _parse_decision(self, response_text: str, current_attempt: int) -> Dict[str, Any]:
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            data = json.loads(json_str)
            action = (data.get("action") or "compose").lower()

            if action in ["improve", "fresh"]:
                mode = action
                action = "compose"
            elif action == "accept":
                return {"action": "accept"}

            mode = (data.get("mode") or "improve").lower()

            if mode in ["stay the same", "keep same", "same", "stay", "keep"]:
                mode = "improve"
            if mode not in ["improve", "fresh"]:
                print(f"WARNING: Unknown mode '{mode}', defaulting to 'improve'")
                mode = "improve"

            decision: Dict[str, Any] = {"action": "compose", "mode": mode}

            if mode == "fresh":
                new_topics = data.get("new_topic_distribution", self.current_topic_distribution)
                new_diffs = data.get("new_difficulty_distribution", self.current_difficulty_distribution)

                expected_topic_len = len(self.initial_topic_distribution)
                expected_diff_len = len(self.initial_difficulty_distribution)

                if not isinstance(new_topics, list) or len(new_topics) != expected_topic_len:
                    print(
                        f"WARNING: invalid topic distribution on attempt {current_attempt}. "
                        f"Expected len={expected_topic_len}, got {len(new_topics) if isinstance(new_topics, list) else type(new_topics)}"
                    )
                    decision["mode"] = "improve"
                    return decision

                if not isinstance(new_diffs, list) or len(new_diffs) != expected_diff_len:
                    print(
                        f"WARNING: invalid difficulty distribution on attempt {current_attempt}. "
                        f"Expected len={expected_diff_len}, got {len(new_diffs) if isinstance(new_diffs, list) else type(new_diffs)}"
                    )
                    decision["mode"] = "improve"
                    return decision

                decision["new_topic_distribution"] = new_topics
                decision["new_difficulty_distribution"] = new_diffs

            return decision

        except Exception as e:
            print(f"WARNING: Could not parse decision response: {e}")
            print(f"Response was: {response_text[:300]}")
            return {"action": "compose", "mode": "improve"}

    def _parse_survey_responses(self, response_text: str) -> Dict[str, int]:
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            json_str = json_str.replace(",\n}", "\n}").replace(",}", "}")
            data = json.loads(json_str)

            responses: Dict[str, int] = {}
            field_mapping = {
                "accomplishment": ["accomplishment", "q1"],
                "effort": ["effort", "effort required", "effort_required", "q2"],
                "mental_demand": ["mental_demand", "mental-demand", "mentaldemand", "mental demand", "q3"],
                "controllability": ["controllability", "q4"],
                "temporal_demand": ["temporal_demand", "temporal-demand", "temporaldemand", "temporal demand", "q5"],
                "satisfaction": ["satisfaction", "q6"],
            }

            for standard_key, possible_keys in field_mapping.items():
                value = None
                for key in possible_keys:
                    if key in data:
                        value = data[key]
                        break
                try:
                    responses[standard_key] = int(value) if value is not None else 3
                except (ValueError, TypeError):
                    responses[standard_key] = 3

            for k in field_mapping.keys():
                responses[k] = max(1, min(5, responses[k]))

            return responses

        except Exception as e:
            print(f"WARNING: Could not parse survey responses: {e}")
            print(f"Response was: {response_text[:400]}")
            return {
                "accomplishment": 3,
                "effort": 3,
                "mental_demand": 3,
                "controllability": 3,
                "temporal_demand": 3,
                "satisfaction": 3,
            }

    def _build_trajectory_text(self, session_summary: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("QUIZ COMPOSITION SESSION - YOUR EXPERIENCE")
        lines.append(f"Total time: {float(session_summary.get('duration_seconds', 0.0)):.1f} seconds")
        lines.append(f"Number of composition attempts: {int(session_summary.get('iterations', 0))}")
        lines.append("")

        for interaction in session_summary.get("interaction_history", []):
            lines.append(f"--- Attempt {interaction.get('iteration')} ({interaction.get('mode')} mode) ---")
            lines.append(f"Requested topic distribution: {interaction.get('topic_distribution')}")
            lines.append(f"Requested difficulty distribution: {interaction.get('difficulty_distribution')}")
            lines.append(f"Generated quiz ID: {interaction.get('returned_quiz_id')}")
            lines.append(f"Match quality: {float(interaction.get('target_match', 0.0)):.2%}")
            lines.append(f"Generation time: {float(interaction.get('api_time_s', 0.0)):.2f}s")
            lines.append("")

        lines.append("Final Quiz Details:")
        final_quiz = session_summary.get("final_quiz") or {}
        lines.append(f"  Quiz ID: {session_summary.get('final_quiz_id')}")
        lines.append(f"  Number of questions: {final_quiz.get('num_mcqs', 'N/A')}")
        final_match = final_quiz.get("targetMatch", None)
        if isinstance(final_match, (int, float)):
            lines.append(f"  Final match quality: {float(final_match):.2%}")
        else:
            lines.append("  Final match quality: N/A")

        return "\n".join(lines)

    def _generate_quiz_universe(self) -> None:
        print("\n  Generating quiz universe...")

        gen_request = {
            "MCQs": ["data/math.csv"],
            "numQuizzes": 100,
            "numMCQs": self.initial_num_mcqs,
            "listTopics": [],
            "numTopics": len(self.initial_topic_distribution),
            "numDifficulties": len(self.initial_difficulty_distribution),
            "topicMode": 1,
            "levelMode": 1,
            "orderLevel": 2,
        }

        response = requests.post(self.generation_api_url, json=gen_request, timeout=120)
        response.raise_for_status()
        result = response.json()

        self.data_uuid = result.get("RequestID")
        universe_path = result.get("PathToQuizzes")

        print(f"  Universe generated: {self.data_uuid}")
        print(f"  Universe path: {universe_path}")