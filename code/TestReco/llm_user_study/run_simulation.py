import os
import json
import argparse
from datetime import datetime

from llm_user_study.simulation_engine import TestRecoSimulator, MultipleRunsRunner


class LLMStudyPipeline:
    """
    Pipeline for LLM user study simulation.
    
    Workflow:
    1. Load personas from personas.json
    2. Run simulations
    """
    
    def __init__(self, output_base: str = "llm_user_study", personas_path: str = None, filter_persona_ids: list = None, prompt_type: str = "detailed", force: bool = False, runs: int = 1):
        self.output_base = output_base
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.personas_path = personas_path or os.path.join(output_base, "personas.json")
        self.filter_persona_ids = filter_persona_ids
        self.prompt_type = prompt_type
        self.force = force
        self.runs = runs
        
        # Create output directories
        self.dirs = {
            'base': output_base,
            'results': os.path.join(output_base, f'results_{self.timestamp}'),
            'logs': os.path.join(output_base, 'logs')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"Initialized LLM Study Pipeline")
        print(f"Output directory: {self.output_base}")
        print(f"Personas file: {self.personas_path}")
        print(f"Timestamp: {self.timestamp}")
    
    def run_simulations(self) -> str:
        """
        Run TestReco simulations for all personas.
        Returns path to results directory.
        """
        print("\n" + "="*70)
        print("RUNNING SIMULATIONS")
        print("="*70)
        
        # Load personas
        with open(self.personas_path, 'r') as f:
            data = json.load(f)
            persona_dicts = data['personas']
        
        # Convert to LLMPersona objects
        from llm_user_study.persona_builder import LLMPersona
        from real_user_study.loader import load_topic_questions_with_difficulties
        
        personas = [LLMPersona(**p) for p in persona_dicts]
        
        # Load questions dict for all personas (needed for pretest display in prompts)
        questions, _ = load_topic_questions_with_difficulties("Fundamental Mathematics")
        questions_dict = {i: q for i, q in enumerate(questions)}
        for persona in personas:
            persona.set_questions_dict(questions_dict)
        
        # Filter out personas already executed with this prompt type and current model from MODEL_TIERS
        from llm_user_study.config import DB_PATH, MODEL_TIERS
        import sqlite3
        
        # Get the current model from MODEL_TIERS (all education levels use the same model now)
        current_model = MODEL_TIERS["graduate"][0]  # They're all the same
        
        if self.force:
            print(f"--force: skipping completed-persona check, running all {len(personas)} personas")
        else:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT simulated_session_id
                FROM study_sessions
                WHERE prompt_type = ? AND simulated_session_id IS NOT NULL AND model_name = ?
            """, (self.prompt_type, current_model))
            completed_session_ids = {row[0] for row in cursor.fetchall()}
            conn.close()

            if completed_session_ids:
                print(f"Found {len(completed_session_ids)} personas already completed with prompt_type='{self.prompt_type}' and model='{current_model}'")
                personas = [p for p in personas if p.session_id not in completed_session_ids]
                print(f"Remaining {len(personas)} personas to execute")
        
        # Filter if persona IDs provided
        if self.filter_persona_ids:
            personas = [p for p in personas if p.persona_id in self.filter_persona_ids]
            print(f"Filtered to {len(personas)} personas: {self.filter_persona_ids}")
        
        print(f"Loaded {len(personas)} personas for simulation")
        print(f"Using prompt type: {self.prompt_type}")

        #personas = personas[:1]  # TESTING: Limit to first 1 personas
        
        # Create simulator with prompt type
        simulator = TestRecoSimulator(prompt_type=self.prompt_type)
        
        # Create batch runner
        runner = MultipleRunsRunner(simulator)
        
        # Run all simulations
        runner.run_all_personas(
            personas,
            runs_per_persona=self.runs,
            output_dir=self.dirs['results']
        )
        
        print(f"\n✓ Simulations complete: Results saved to {self.dirs['results']}")
        return self.dirs['results']


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM User Study Simulation Pipeline"
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='llm_user_study',
        help='Base output directory (default: llm_user_study)'
    )
    
    parser.add_argument(
        '--personas-path',
        type=str,
        default='llm_user_study/personas.json',
        help='Path to personas JSON file (default: llm_user_study/personas.json)'
    )
    
    parser.add_argument(
        '--persona-ids',
        type=str,
        help='Comma-separated list of persona IDs to simulate (e.g., "45,46,47,48,49")'
    )
    
    parser.add_argument(
        '--prompt-type',
        type=str,
        choices=['detailed', 'general', 'survey', 'simple'],
        default='detailed',
        help='Prompt template to use: "detailed" or "general" (default: detailed)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override model for all education levels (e.g. llama-3-8b, mistral-24b-instruct, gpt-5-2025-08-07)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-run all personas even if already in DB (new rows are appended)'
    )

    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of runs per persona (default: 1)'
    )

    args = parser.parse_args()
    
    # Override MODEL_TIERS if --model provided
    if args.model:
        import llm_user_study.config as _cfg
        for k in _cfg.MODEL_TIERS:
            _cfg.MODEL_TIERS[k] = [args.model]
        print(f"Model overridden to: {args.model}")

    # Parse persona IDs if provided
    filter_ids = None
    if args.persona_ids:
        filter_ids = [int(x.strip()) for x in args.persona_ids.split(',')]
        print(f"Will simulate only persona IDs: {filter_ids}")

    # Create and run pipeline
    pipeline = LLMStudyPipeline(
        output_base=args.output_dir,
        personas_path=args.personas_path,
        filter_persona_ids=filter_ids,
        prompt_type=args.prompt_type,
        force=args.force,
        runs=args.runs,
    )
    pipeline.run_simulations()


if __name__ == "__main__":
    main()
