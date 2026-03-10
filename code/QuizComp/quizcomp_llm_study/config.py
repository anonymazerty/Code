"""
Configuration for QuizComp LLM Teacher Simulation.
"""

import os
from pathlib import Path

# Database 
DB_PATH = Path(__file__).parent / "llm_results_gpt_general.db"

# LLM Configuration
LLM_PROVIDER = "openai"  # Options: "openrouter", "openai"
OLLAMA_BASE_URL = "http://localhost:11434/v1"  
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"  # OpenRouter API endpoint

# LLM Models
if LLM_PROVIDER == "ollama":
    
    MODEL_TIERS = {
        "all": ["llama3.1:8b"]  
    }
    MODEL_NAME = "llama3.1:8b"  # Default model
elif LLM_PROVIDER == "openai":
    
    openai_model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5-2025-08-07")
    MODEL_TIERS = {
        "all": [openai_model]
    }
    MODEL_NAME = openai_model
else:
    # OpenRouter models - Mistral Small 24B
    openrouter_model = os.environ.get("OPENROUTER_MODEL_NAME", "mistralai/mistral-small-3.2-24b-instruct")
    MODEL_TIERS = {
        "all": [openrouter_model]
    }
    MODEL_NAME = openrouter_model

# Simulation parameters
SIMULATIONS_PER_PERSONA = 1  # How many times to run each persona
PROMPT_TYPE = "detailed"  # Type of prompt template to use

# Quiz composition parameters
MAX_ITERATIONS = 10  # Maximum number of composition iterations per session 
QUESTIONS_PER_QUIZ = 10  # Target number of questions per quiz

# Topic (Math)
TOPIC = "Mathematics"

# Survey questions (Q1-Q6 only)
SURVEY_QUESTIONS = [
    {
        "id": "q1",
        "key": "accomplishment",
        "text": "Feeling of Accomplishment: How successful do you feel you were in building a quiz that matches your needs (topics and difficulty) during the session?"
    },
    {
        "id": "q2",
        "key": "effort",
        "text": "Effort Required: How much effort was required to inspect the proposed quizzes and decide whether to keep or change them?"
    },
    {
        "id": "q3",
        "key": "mental_demand",
        "text": "Mental Demand: How mentally demanding was it to read and evaluate the candidate quizzes?"
    },
    {
        "id": "q4",
        "key": "controllability",
        "text": "Perceived Controllability: How much control did you feel you had over the final quiz?"
    },
    {
        "id": "q5",
        "key": "temporal_demand",
        "text": "Temporal Demand: How time-pressured did you feel while composing the quiz within the allotted time?"
    },
    {
        "id": "q6",
        "key": "satisfaction",
        "text": "Satisfaction: Overall, how satisfied are you with the final quiz you produced?"
    }
]

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
