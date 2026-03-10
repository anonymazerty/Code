import os
from typing import Dict, List

# Database configuration
DB_PATH = os.environ.get("REALUSER_DB_PATH", "./TestReco/results.db")
DB_URL = os.environ.get("REALUSER_DB_URL", f"sqlite:///{DB_PATH}")

# Study configuration
TOPIC = "Fundamental Mathematics"
BENCHMARK = "math_bench"
ORCHESTRATOR_TYPE = "tool_call"

# Simulation parameters
SIMULATIONS_PER_PERSONA = 5  
PROMPT_TYPE = "general" 

MODEL_TIERS = {
    "graduate": [
        "gpt-5-2025-08-07", 
    ],
    "undergraduate": [
        "gpt-5-2025-08-07", 
    ],
    "high_school": [
        "gpt-5-2025-08-07", 
    ]
}

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# TestReco configuration
PRETEST_N = 12  # Number of pretest questions
STEPS_N = 10  # Number of learning steps
QUESTIONS_PER_STEP = 5  # Questions per recommendation

# Policy objectives
OBJECTIVES = ["performance", "gap", "aptitude"]

MODEL_NAME="chatgpt-4o-mini"  # For orchestrator only orchestrator

# Pricing per 1M tokens (student simulation models only)
# GPT-5: OpenAI direct API pricing
# LLaMA 3 8B, Mistral Small 24B: OpenRouter weighted-average pricing
MODEL_PRICING = {
    "gpt-5-2025-08-07": {
        "input_per_1m":  2.50,
        "output_per_1m": 20.00,
    },
    "llama-3-8b": {
        "input_per_1m":  0.036,
        "output_per_1m": 0.038,
    },
    "mistral-24b-instruct": {
        "input_per_1m":  0.053,
        "output_per_1m": 0.095,
    },
}

# Output directories
OUTPUT_DIR = "llm_user_study/results"
PERSONA_DIR = "llm_user_study/personas"
LOGS_DIR = "llm_user_study/logs"

# Helper function to select model for a persona
def get_model_for_persona(education_level: str, persona_index: int) -> str:
    """
    Selects a model for a given persona based on their education level and index.
    """
    models = MODEL_TIERS.get(education_level.lower(), MODEL_TIERS["undergraduate"])
    return models[persona_index % len(models)]
