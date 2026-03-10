import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quizcomp_llm_study.teacher_persona import load_teacher_personas
from quizcomp_llm_study.interactive_simulation_engine import (
    InteractiveQuizCompSimulator,
    MultipleInteractiveRunsRunner
)
from quizcomp_llm_study.db_saver import SimulationDatabaseSaver
from quizcomp_llm_study.config import DB_PATH, LLM_PROVIDER, MODEL_TIERS


def run_interactive_simulations(
    persona_id: int = None,
    api_base_url: str = "http://localhost:8000",
    max_personas: int = None,
    test_mode: bool = False,
    prompt_variant: str = "detailed",
    db_path: str = None,
    resume: bool = True
):
    
    # Auto-select database path based on LLM and prompt variant if not specified
    if db_path is None:
        llm_name = LLM_PROVIDER
        if LLM_PROVIDER == "openai":
            llm_name = "gpt"
        elif LLM_PROVIDER == "ollama":
            llm_name = "llama"
        db_path = Path(__file__).parent / f"llm_results_{llm_name}_{prompt_variant}.db"
    else:
        db_path = Path(db_path)
    
    print(f"\n{'='*70}")
    print(f"INTERACTIVE LLM TEACHER SIMULATION")
    print(f"{'='*70}")
    print(f"API Base URL: {api_base_url}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Model: {MODEL_TIERS['all'][0]}")
    print(f"Prompt Variant: {prompt_variant}")
    print(f"Resume Mode: {resume}")
    
    # Initialize database
    db_saver = SimulationDatabaseSaver(db_path)
    print(f"Database: {db_path}")
    print(f"{'='*70}\n")
    
    # Test test
    if test_mode:
        print("Running TEST simulation with custom parameters...")
        simulator = InteractiveQuizCompSimulator(api_base_url=api_base_url, prompt_variant=prompt_variant)
        
        # Example: Math quiz with mixed topics and medium difficulty
        result = simulator.run_simulation_from_scratch(
            persona_id=9999,  # Test persona ID
            run_number=0,
            initial_topic_distribution=[0.2, 0.3, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            initial_difficulty_distribution=[0.2, 0.3, 0.3, 0.1, 0.1, 0.0],
            initial_num_mcqs=10,
            model_path="models/dqn_t1_math_r2",  # Use math model (trained with 16 features: 10 topics + 6 difficulties)
            alfa_value=0.5,
            prompt_variant=prompt_variant
        )
        
        print("\n Test simulation complete!")
        print(f"  Iterations: {result.num_iterations}")
        print(f"  Final match: {result.final_match_quality:.2%}")
        print(f"  Survey responses: {result.survey_responses}")
        return
    
    # Load personas
    print("Loading teacher personas...")
    personas = load_teacher_personas()
    print(f"Loaded {len(personas)} personas")
    
    # Filter by persona_id if specified
    if persona_id is not None:
        personas = [p for p in personas if p.persona_id == persona_id]
        if not personas:
            print(f"ERROR: Persona {persona_id} not found!")
            return
        print(f"Running for persona {persona_id} only")
    
    # Limit number of personas if specified
    if max_personas is not None:
        personas = personas[:max_personas]
        print(f"Limited to first {max_personas} personas")
    
    # Check that we have valid data UUIDs
    print("\nChecking persona data...")
    valid_personas = []
    for persona in personas:
        if persona.compose_attempts and persona.compose_attempts[0].get('data_uuid'):
            valid_personas.append(persona)
        else:
            print(f"  Skipping persona {persona.persona_id} (no data_uuid)")
    
    print(f"Valid personas for simulation: {len(valid_personas)}")
    
    if not valid_personas:
        print("\nERROR: No valid personas found!")
        print("Personas need at least one composition attempt with a data_uuid.")
        return
    
    # Run simulations
    print(f"\n{'='*70}")
    print(f"STARTING INTERACTIVE SIMULATIONS")
    print(f"{'='*70}\n")
    
    runner = MultipleInteractiveRunsRunner(
        api_base_url=api_base_url,
        db_saver=db_saver,
        prompt_variant=prompt_variant
    )
    
    results = runner.run_for_all_personas(
        personas=valid_personas,
        runs_per_persona=5,
        resume=resume
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total simulations: {len(results)}")
    
    if results:
        avg_iterations = sum(r.num_iterations for r in results) / len(results)
        avg_match = sum(r.final_match_quality for r in results) / len(results)
        avg_duration = sum(r.duration_seconds for r in results) / len(results)
        avg_tokens = sum(r.total_tokens for r in results) / len(results)
        avg_llm_calls = sum(r.total_llm_calls for r in results) / len(results)
        total_tokens = sum(r.total_tokens for r in results)
        total_llm_calls = sum(r.total_llm_calls for r in results)
        
        print(f"Average iterations: {avg_iterations:.1f}")
        print(f"Average final match: {avg_match:.2%}")
        print(f"Average duration: {avg_duration:.1f}s")
        
        print(f"\n Cost Metrics:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Average tokens per simulation: {avg_tokens:.0f}")
        print(f"  Total LLM calls: {total_llm_calls:,}")
        print(f"  Average LLM calls per simulation: {avg_llm_calls:.1f}")
        
        print(f"\nSurvey Response Averages (LLM):")
        for key in ['accomplishment', 'effort', 'mental_demand', 'controllability', 
                   'temporal_demand', 'satisfaction']:
            avg_value = sum(r.survey_responses.get(key, 3) for r in results) / len(results)
            print(f"  {key}: {avg_value:.2f}/5")
    
    print(f"\nResults saved to: {DB_PATH}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Run interactive LLM teacher simulations for QuizComp"
    )
    parser.add_argument(
        "--persona-id",
        type=int,
        help="Run for specific persona ID only"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server"
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        help="Maximum number of personas to process (for testing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a single test simulation with custom parameters"
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=["general", "inbetween", "detailed"],
        default="detailed",
        help="Prompt variant to use: general (minimal), inbetween (moderate), or detailed (full guidance)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Custom database path (default: auto-select based on LLM and prompt variant)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode (re-run all simulations even if they exist)"
    )
    
    args = parser.parse_args()
    
    # Check API key based on provider
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
        print("  ollama pull llama3.1:8b")
        print()
    
    # Check that composition API is reachable
    import requests
    try:
        response = requests.get(f"{args.api_url}/docs", timeout=5)
        print(f"✓ Composition API is reachable at {args.api_url}")
    except Exception as e:
        print(f"WARNING: Could not reach composition API at {args.api_url}")
        print(f"  Error: {e}")
        print(f"  Make sure the QuizComp API server is running!")
        print(f"  You can start it with: python -m app.main")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    run_interactive_simulations(
        persona_id=args.persona_id,
        api_base_url=args.api_url,
        max_personas=args.max_personas,
        test_mode=args.test,
        prompt_variant=args.prompt_variant,
        db_path=args.db_path,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
