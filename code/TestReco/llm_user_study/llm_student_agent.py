import os
import time
import json
import random
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from openai import OpenAI

from llm_user_study.persona_builder import LLMPersona
from llm_user_study.config import MODEL_TIERS



class LLMStudentAgent:
    """
    Simulates a student using an LLM based on a persona.
    Handles question answering, survey responses, and behavioral simulation.
    """
    
    def __init__(self, persona: LLMPersona, prompt_type: str = "", seed: int = None, llm_model = None, model_name: str = None):
        self.persona = persona
        self.prompt_type = prompt_type
        self.seed = seed or random.randint(0, 1000000)
        self.rng = random.Random(self.seed)
        
        # Current state
        self.current_mastery = persona.pre_qualification_score
        self.questions_seen = 0
        
        # Use provided LLM model from orchestrator or create own client
        self.llm_model = llm_model
        
        if llm_model is not None:
            # Use orchestrator's LLM model (already configured with OpenRouter)
            self.client = None  # Will use llm_model.generate_chat instead
            self.model_name = model_name or MODEL_TIERS[persona.education_level][0]
        else:
            model_shortname = model_name or MODEL_TIERS[persona.education_level][0]
            
            # Map model shortnames to actual OpenRouter/OpenAI model IDs
            model_mapping = {
                "chatgpt-4o": "openai/chatgpt-4o-latest",
                "chatgpt-4o-mini": "openai/gpt-4o-mini",
                "chatgpt-4-turbo": "openai/gpt-4-turbo",
                "chatgpt-3.5-turbo": "openai/gpt-3.5-turbo",
                "gpt-5-2025-08-07": "openai/gpt-5-2025-08-07",
                "claude-3.7-sonnet-thinking": "anthropic/claude-3-7-sonnet-20250219:thinking",
                "llama-3-8b": "meta-llama/llama-3-8b-instruct",
                "llama-3-70b": "meta-llama/llama-3-70b-instruct",
                "gemma-2-9b-it": "google/gemma-2-9b-it",
                "mistral-24b-instruct": "mistralai/mistral-small-24b-instruct-2501"
            }
            
            self.model_name = model_mapping.get(model_shortname, model_shortname)
            
            # Use direct OpenAI API for openai/ models if OPENAI_API_KEY is available
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key and self.model_name.startswith("openai/"):
                self.client = OpenAI(api_key=openai_api_key)
                self.model_name = self.model_name.replace("openai/", "")  # Remove prefix for direct API
            else:
                # Fallback to OpenRouter
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY")
                )
        
        # Conversation history for context
        self.conversation_history = []
    
    def answer_question(
        self,
        question: Dict[str, Any],
        difficulty: float,
        context: Optional[str] = None
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Simulate answering a multiple choice question using LLM.
        
        Args:
            question: Question dict with 'text', 'options', 'answer' (0-indexed)
            difficulty: Scaled difficulty (0-1 range)
            context: Optional context string
        
        Returns:
            (chosen_option_index, response_time_ms, metadata)
        """
        self.questions_seen += 1

        chosen_idx, reasoning, llm_latency_s, input_tokens, output_tokens = \
            self._llm_answer_question(question, difficulty)

        # Calculate realistic response time (3-12 seconds)
        base_time = 5000  # 5 seconds baseline
        difficulty_factor = 1.0 + (difficulty * 0.5)
        noise = self.rng.uniform(0.8, 1.2)
        response_time = base_time * difficulty_factor * noise

        metadata = {
            'reasoning': reasoning,
            'mastery_at_attempt': self.current_mastery,
            'llm_latency_s': llm_latency_s,
            'llm_input_tokens': input_tokens,
            'llm_output_tokens': output_tokens,
        }

        return chosen_idx, response_time, metadata
    
    def _llm_answer_question(
        self,
        question: Dict[str, Any],
        difficulty: float
    ) -> Tuple[int, str, float, int, int]:
        """Use LLM to answer the question.

        Returns:
            (chosen_idx, reasoning, latency_s, input_tokens, output_tokens)
        """
        import re

        # Build prompt - use 'text' field for question text
        q_text = question.get('text', question.get('question', 'Question text not available'))
        options = question.get('options', [])

        options_text = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)])

        prompt = f"""Based on your current understanding (mastery: {self.current_mastery:.1%}), try to answer this question as a ##STUDENT with a {self.persona.education_level} education level.:
Question:
{q_text}

Options:
{options_text}

Provide your answer as a single number (0, 1, 2, or 3).
Format your response as: ANSWER: [number]
"""

        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        response_text = ""

        try:
            print(f"Using model: {self.model_name}")

            if self.llm_model is not None:
                from generators.model import Message, token_usage_callback
                print("system prompt:")
                print(self.prompt_type)
                messages = [
                    Message(role="system", content=self.persona.get_system_prompt(prompt_type=self.prompt_type)),
                    Message(role="user", content=prompt)
                ]
                if hasattr(self.llm_model, 'custom_llm'):
                    response_text = self.llm_model.custom_llm.generate_chat(messages, max_tokens=1024, temperature=0.0)
                else:
                    response_text = self.llm_model.generate_chat(messages, max_tokens=1024, temperature=0.0)
                input_tokens = token_usage_callback.request_prompt_tokens
                output_tokens = token_usage_callback.request_completion_tokens
            else:
                messages = [
                    {"role": "system", "content": self.persona.get_system_prompt(prompt_type=self.prompt_type)},
                    {"role": "user", "content": prompt}
                ]
                print("SYSTEM:")
                print(messages[0]["content"])
                print("\nUSER:")
                print(messages[1]["content"])

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                response_text = response.choices[0].message.content
                if hasattr(response, 'usage') and response.usage:
                    input_tokens  = getattr(response.usage, 'prompt_tokens', 0) or 0
                    output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

        except Exception as e:
            import traceback
            print(f"LLM query failed: {type(e).__name__}: {str(e)}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            latency_s = time.time() - start_time
            return 0, str(e), latency_s, 0, 0

        latency_s = time.time() - start_time

        match = re.search(r'ANSWER:\s*(\d+)', response_text)
        if match:
            answer_idx = int(match.group(1))
            if 0 <= answer_idx < len(options):
                print(f"LLM answered: {answer_idx} for question (mastery: {self.current_mastery:.1%})")
                return answer_idx, response_text, latency_s, input_tokens, output_tokens

        print(f"Warning: Could not parse LLM answer, defaulting to 0. Response: {response_text[:100]}")
        return 0, f"Parse failed: {response_text}", latency_s, input_tokens, output_tokens
    

    def update_mastery(self, new_mastery: float):
        """Update current mastery level (called after learning step)."""
        self.current_mastery = new_mastery
    
    def complete_survey(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete NASA-TLX style survey based on session experience.
        
        Args:
            session_data: Information about the session including orchestrator log and learning trajectory
        
        Returns:
            Survey responses (1-5 Likert scale)
        """
        # Extract orchestrator log and session info
        orchestrator_log = session_data.get('orchestrator_log', 'No log available')
        learning_trajectory = session_data.get('learning_trajectory', orchestrator_log)
        mastery_change = session_data.get('mastery_change', 0.0)
        
        # Get survey-specific system prompt with learning trajectory
        system_prompt = self.persona.get_system_prompt(prompt_type="survey", learning_trajectory=learning_trajectory)
        
        prompt = f"""Based on your experience, please rate the following on a scale of 1-5 (1=Very Low, 5=Very High):

Q1 - Feeling of Accomplishment: How much do you feel that the recommended questions helped you improve your understanding of the topic?

Q2 - Effort Required: How much effort was required to follow the recommendations and work through the questions?

Q3 - Mental Demand: How mentally demanding was it to understand and solve the recommended questions?

Q4 - Perceived Controllability: How well did the difficulty of the recommended questions match your current level throughout the session?

Q5 - Temporal Demand: How time-pressured did you feel while completing the recommended questions within the allotted time?

Q6 - Frustration: How frustrated did you feel while working with the recommended questions?

Q7 - Trust: How much did you trust the system to recommend appropriate questions for your learning progress?

Format your response as JSON:
{{
  "accomplishment": [1-5],
  "effort_required": [1-5],
  "mental_demand": [1-5],
  "perceived_controllability": [1-5],
  "temporal_demand": [1-5],
  "frustration": [1-5],
  "trust": [1-5]
}}
"""
        
        import re

        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        response_text = ""

        try:
            if self.llm_model is not None:
                print(f"Using model: {self.model_name} for survey")
                from generators.model import Message, token_usage_callback
                messages = [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=prompt)
                ]
                print("PROMPT TO LLM (Survey):")
                print("SYSTEM:")
                print(messages[0].content)
                print("\nUSER:")
                print(messages[1].content)
                if hasattr(self.llm_model, 'custom_llm'):
                    response_text = self.llm_model.custom_llm.generate_chat(messages, max_tokens=1024, temperature=0.0)
                    print("LLM survey response:")
                    print(response_text)
                else:
                    response_text = self.llm_model.generate_chat(messages, max_tokens=1024, temperature=0.0)
                    print("LLM survey response 2:")
                    print(response_text)
                input_tokens  = token_usage_callback.request_prompt_tokens
                output_tokens = token_usage_callback.request_completion_tokens
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                print("\n" + "="*80)
                print("PROMPT TO LLM (Survey):")
                print("="*80)
                print("SYSTEM:")
                print(messages[0]["content"])
                print("\nUSER:")
                print(messages[1]["content"])
                print("="*80 + "\n")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                response_text = response.choices[0].message.content
                if hasattr(response, 'usage') and response.usage:
                    input_tokens  = getattr(response.usage, 'prompt_tokens', 0) or 0
                    output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

        except Exception as e:
            print(f"LLM survey failed: {e}")
            token_info = {'latency_s': time.time() - start_time,
                          'input_tokens': 0, 'output_tokens': 0}
            return None, token_info

        latency_s = time.time() - start_time
        token_info = {'latency_s': latency_s,
                      'input_tokens': input_tokens,
                      'output_tokens': output_tokens}

        if not response_text:
            return None, token_info

        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            survey_data = json.loads(json_match.group(0))
            return self._validate_survey(survey_data, mastery_change), token_info

        return None, token_info
    
    def _validate_survey(
        self,
        survey_data: Dict[str, Any],
        mastery_change: float
    ) -> Dict[str, Any]:
        """Validate survey responses are in valid range."""
        
        # Map controllability to perceived_controllability for database
        if 'controllability' in survey_data:
            survey_data['perceived_controllability'] = survey_data.pop('controllability')
        
        # Set default for would_use_again (not asked to LLM)
        survey_data['would_use_again'] = 1
        
        # Validate all required fields exist and are in range
        required_fields = [
            'accomplishment', 'effort_required', 'mental_demand',
            'perceived_controllability', 'temporal_demand', 'frustration', 
            'trust', 'would_use_again'
        ]
        
        for field in required_fields:
            if field not in survey_data or not isinstance(survey_data[field], (int, float)):
                default_val = 1 if field == 'would_use_again' else 3
                survey_data[field] = default_val
            else:
                survey_data[field] = int(np.clip(survey_data[field], 1, 5))
        
        # Remove any extra fields that shouldn't be in the database
        valid_fields = set(required_fields + ['free_text'])
        survey_data = {k: v for k, v in survey_data.items() if k in valid_fields}
        
        return survey_data
    
    def reset_for_new_session(self):
        """Reset agent state for a new simulation run."""
        self.current_mastery = self.persona.pre_qualification_score
        self.questions_seen = 0
        self.conversation_history = []
