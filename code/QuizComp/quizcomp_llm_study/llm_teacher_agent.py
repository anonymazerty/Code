import os
import time
import json
import random
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from quizcomp_llm_study.teacher_persona import TeacherPersona
from quizcomp_llm_study.config import MODEL_TIERS, SURVEY_QUESTIONS
from quizcomp_llm_study.prompt import (
    build_composition_prompt,
    build_survey_prompt
)


class LLMTeacherAgent:
    
    def __init__(self, persona: TeacherPersona, prompt_type: str = "detailed", 
                 seed: int = None):
        self.persona = persona
        self.prompt_type = prompt_type
        self.seed = seed or random.randint(0, 1000000)
        self.rng = random.Random(self.seed)
        self.iterations = 0
        
        # Token and call tracking
        self.total_tokens = 0
        self.total_llm_calls = 0
        
        # Import config here to get LLM_PROVIDER
        from quizcomp_llm_study.config import LLM_PROVIDER, MODEL_TIERS, OLLAMA_BASE_URL
        
        self.llm_provider = LLM_PROVIDER
        model_shortname = MODEL_TIERS["all"][0]
        
        if self.llm_provider == "ollama":
            # Use Ollama
            self.model_name = model_shortname
            self.client = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama"  # Ollama doesn't need a real API key
            )
            print(f"Initialized LLM Teacher Agent with Ollama model: {self.model_name}")
        
        elif self.llm_provider == "openai":
            # Use OpenAI directly
            self.model_name = model_shortname
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            print(f"Initialized LLM Teacher Agent with OpenAI model: {self.model_name}")
        
        else:
            # Use OpenRouter
            model_mapping = {
                "gpt-5": "openai/gpt-4o",
                "chatgpt-4o": "openai/chatgpt-4o-latest",
                "chatgpt-4o-mini": "openai/gpt-4o-mini",
            }
            
            self.model_name = model_mapping.get(model_shortname, model_shortname)
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
            print(f"Initialized LLM Teacher Agent with OpenRouter model: {self.model_name}")
    
    def compose_quiz(
        self,
        iteration: int,
        previous_quizzes: List[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], str]:
        self.iterations += 1
        
        # Build prompts using prompt.py
        system_prompt, user_prompt = build_composition_prompt(
            iteration=iteration,
            previous_quizzes=previous_quizzes or [],
            prompt_type=self.prompt_type
        )
        
        # Get LLM response
        response_text, reasoning = self._call_llm(system_prompt, user_prompt)
        
        # Parse composition request
        composition_request = self._parse_composition_request(response_text, iteration)
        
        return composition_request, reasoning
    
    def complete_survey(self, trajectory_summary: str) -> Dict[str, int]:
        # Build prompts using prompt.py
        system_prompt, user_prompt = build_survey_prompt(trajectory_summary)
        
        # Get LLM response
        response_text, _ = self._call_llm(system_prompt, user_prompt, timeout=120)
        
        # Parse survey responses
        survey_responses = self._parse_survey_responses(response_text)
        
        return survey_responses
    
    def _call_llm(self, system_prompt: str, user_prompt: str, 
                  timeout: int = 60) -> Tuple[str, str]:
        """Call OpenRouter LLM and return response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            call_kwargs = {
                "model": self.model_name,
                "messages": messages
            }

            response = self.client.chat.completions.create(**call_kwargs)
            response_text = response.choices[0].message.content
            
            # Track tokens and calls
            self.total_llm_calls += 1
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens
            else:
                # Estimate tokens if not provided (rough estimate: ~4 chars per token)
                estimated_tokens = (len(system_prompt) + len(user_prompt) + len(response_text)) // 4
                self.total_tokens += estimated_tokens
            
            return response_text, response_text
        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
            return "", f"Error: {str(e)}"
    
    def _parse_composition_request(self, response_text: str, iteration: int) -> Dict[str, Any]:
        """Parse composition request from LLM response."""
        try:
            # Try to extract JSON from response
            if '{' in response_text and '}' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return {
                    'action': data.get('action', 'compose'),
                    'mode': data.get('mode', 'fresh' if iteration == 0 else 'improve'),
                    'topic_distribution': data.get('topic_distribution', [0.33, 0.33, 0.34]),
                    'difficulty_distribution': data.get('difficulty_distribution', [0.3, 0.4, 0.3]),
                    'reasoning': data.get('reasoning', '')
                }
        except:
            pass
        
        # Default fallback
        return {
            'action': 'compose',
            'mode': 'fresh' if iteration == 0 else 'improve',
            'topic_distribution': [0.33, 0.33, 0.34],
            'difficulty_distribution': [0.3, 0.4, 0.3],
            'reasoning': 'Default composition'
        }
    
    def _parse_survey_responses(self, response_text: str) -> Dict[str, int]:
        """Parse survey responses from LLM response."""
        try:
            # Try to extract JSON from response
            if '{' in response_text and '}' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                # Extract Q1-Q6 responses
                responses = {}
                for q in SURVEY_QUESTIONS:
                    key = q['key']
                    value = data.get(key)
                    if value is not None:
                        # Clamp to 1-5
                        responses[key] = max(1, min(5, int(value)))
                    else:
                        # Default to 3 (neutral) if missing
                        responses[key] = 3
                
                return responses
        except:
            pass
        
        # Default fallback - neutral ratings
        return {q['key']: 3 for q in SURVEY_QUESTIONS}
