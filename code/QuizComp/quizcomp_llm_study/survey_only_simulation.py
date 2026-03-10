import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables from .env
load_dotenv()

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from quizcomp_llm_study.teacher_persona import load_teacher_personas
from quizcomp_llm_study.llm_teacher_agent import LLMTeacherAgent
from quizcomp_llm_study.db_saver import SimulationDatabaseSaver
from quizcomp_llm_study.config import DB_PATH, PROMPT_TYPE


def run_survey_only_simulation(persona_id: int = None):
    # Load personas
    personas = load_teacher_personas()
    
    if persona_id is not None:
        personas = [p for p in personas if p.persona_id == persona_id]
        if not personas:
            print(f"ERROR: Persona {persona_id} not found!")
            return
    
    # Get model info from config
    from quizcomp_llm_study.config import LLM_PROVIDER, MODEL_TIERS
    model_name = MODEL_TIERS["all"][0]
    
    print(f"\n{'='*70}")
    print(f"SURVEY-ONLY SIMULATION")
    print(f"{'='*70}")
    print(f"Personas: {len(personas)}")
    print(f"Using real composition history from personas")
    print(f"Model: {model_name} ({LLM_PROVIDER})")
    print(f"{'='*70}\n")
    
    # Initialize database
    db_saver = SimulationDatabaseSaver(DB_PATH)
    
    # Number of runs per persona
    RUNS_PER_PERSONA = 5
    
    for persona in personas:
        for run_num in range(1, RUNS_PER_PERSONA + 1):
            print(f"\n--- Persona {persona.persona_id} (Session {persona.session_id}) - Run {run_num}/{RUNS_PER_PERSONA} ---")
            
            # Create LLM agent
            agent = LLMTeacherAgent(persona)
        
            # Build trajectory summary from REAL composition history including only: attempts, times, match quality, iterations
            trajectory_summary = persona.get_composition_trajectory_summary()
            
            if run_num == 1:  # Only show details on first run
                print(f"Real composition history (WITHOUT survey answers):")
                print(f"  - {persona.num_compose_attempts} composition attempts")
                print(f"  - {persona.total_time_s:.1f}s ({persona.total_time_s/60:.1f} min)")
                print(f"\nTrajectory summary shown to LLM:")
                print(f"{trajectory_summary[:300]}...")  # Show first 300 chars
            
            print(f"\nAsking {model_name} to complete survey based on this history...")
            
            # Complete survey
            survey_responses = agent.complete_survey(trajectory_summary)
            
            print(f"\nLLM Survey Responses (Q1-Q6):")
            for key, value in survey_responses.items():
                print(f"  {key}: {value}/5")
            
            if run_num == 1:  # Only show real answers on first run
                print(f"\nReal Teacher Responses (Q1-Q6):")
                print(f"  accomplishment: {persona.survey_q1_accomplishment}/5")
                print(f"  effort: {persona.survey_q2_effort}/5")
                print(f"  mental_demand: {persona.survey_q3_mental_demand}/5")
                print(f"  controllability: {persona.survey_q4_controllability}/5")
                print(f"  temporal_demand: {persona.survey_q5_temporal_demand}/5")
                print(f"  satisfaction: {persona.survey_q6_satisfaction}/5")
            
            # Get token and call stats from agent
            total_tokens = agent.total_tokens
            total_llm_calls = agent.total_llm_calls
            
            print(f"\nCost tracking:")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Total LLM calls: {total_llm_calls}")
            
            # Create simulation result
            from quizcomp_llm_study.simulation_engine import SimulationResult
            
            result = SimulationResult(
                persona_id=persona.persona_id,
                run_number=run_num,
                session_id=f"survey_only_{persona.persona_id}_run{run_num}_{int(datetime.now().timestamp())}",
                num_iterations=persona.num_compose_attempts,
                final_quiz_size=0,  # Not simulating composition
                final_match_quality=0.0,
                composition_attempts=[],  # Using real history, not simulated
                survey_responses=survey_responses,
                duration_seconds=0.0,
                timestamp=datetime.now().isoformat(),
                prompt_type="survey_only",  # Fixed value for survey-only mode
                model_name=model_name,
                total_tokens=total_tokens,
                total_llm_calls=total_llm_calls
            )
            
            # Save to database
            db_saver.save_simulation(result, persona)
            print(f"\n✓ Saved run {run_num} to database: {DB_PATH}")
    
    print(f"\n{'='*70}")
    print(f"SURVEY-ONLY SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run survey-only simulation using real teacher composition history"
    )
    parser.add_argument(
        "--persona-id",
        type=int,
        help="Run for specific persona ID only"
    )
    
    args = parser.parse_args()
    
    # Check API key based on provider
    from quizcomp_llm_study.config import LLM_PROVIDER
    
    if LLM_PROVIDER == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY not found in .env file!")
            print("Please create a .env file with: OPENROUTER_API_KEY=your-key-here")
            exit(1)
    elif LLM_PROVIDER == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not found in .env file!")
            print("Please add to .env file: OPENAI_API_KEY=your-key-here")
            exit(1)
    elif LLM_PROVIDER == "ollama":
        print("Using Ollama - make sure it's running:")
        print("  ollama serve")
        print("  ollama pull llama3:8b")
        print("  ollama pull gemma2:9b")
        print()
    
    run_survey_only_simulation(
        persona_id=args.persona_id
    )
