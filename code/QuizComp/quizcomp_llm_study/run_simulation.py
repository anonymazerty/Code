import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from quizcomp_llm_study.teacher_persona import load_teacher_personas
from quizcomp_llm_study.simulation_engine import QuizCompSimulator, MultipleRunsRunner
from quizcomp_llm_study.db_saver import SimulationDatabaseSaver
from quizcomp_llm_study.config import SIMULATIONS_PER_PERSONA, PROMPT_TYPE, MODEL_TIERS


class QuizCompLLMStudyPipeline:
    
    def __init__(
        self,
        output_base: str = "quizcomp_llm_study",
        personas_path: str = None,
        filter_persona_ids: list = None,
        prompt_type: str = PROMPT_TYPE
    ):
        self.output_base = Path(output_base)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use script directory for default personas path
        script_dir = Path(__file__).parent
        self.personas_path = personas_path or (script_dir / "teacher_personas.json")
        self.filter_persona_ids = filter_persona_ids
        self.prompt_type = prompt_type
        
        # Create output directories
        self.dirs = {
            'base': self.output_base,
            'results': self.output_base / f'results_{self.timestamp}',
            'logs': self.output_base / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized QuizComp LLM Study Pipeline")
        print(f"Output directory: {self.output_base}")
        print(f"Personas file: {self.personas_path}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Prompt type: {self.prompt_type}")
    
    def run_simulations(self) -> str:
        print("\n" + "="*70)
        print("RUNNING QUIZCOMP LLM TEACHER SIMULATIONS")
        print("="*70)
        
        # Load personas
        print(f"\nLoading personas from {self.personas_path}...")
        personas = load_teacher_personas(str(self.personas_path))
        print(f"Loaded {len(personas)} teacher personas")
        
        # Filter out personas already executed
        from quizcomp_llm_study.config import DB_PATH
        db_saver = SimulationDatabaseSaver(DB_PATH)
        
        current_model = MODEL_TIERS["all"][0]
        completed = db_saver.get_completed_simulations(self.prompt_type, current_model)
        
        if completed:
            print(f"Found {len(completed)} personas already completed with "
                  f"prompt_type='{self.prompt_type}' and model='{current_model}'")
            personas = [p for p in personas 
                       if not any((p.persona_id, run) in completed 
                                 for run in range(SIMULATIONS_PER_PERSONA))]
            print(f"Remaining {len(personas)} personas to execute")
        
        # Filter if persona IDs provided
        if self.filter_persona_ids:
            personas = [p for p in personas if p.persona_id in self.filter_persona_ids]
            print(f"Filtered to {len(personas)} personas: {self.filter_persona_ids}")
        
        if not personas:
            print("No personas to simulate!")
            return str(self.dirs['results'])
        
        # Run simulations
        runner = MultipleRunsRunner(
            output_dir=str(self.dirs['results']),
            db_saver=db_saver
        )
        
        results = runner.run_all(
            personas=personas,
            simulations_per_persona=SIMULATIONS_PER_PERSONA,
            prompt_type=self.prompt_type
        )
        
        print(f"\n{'='*70}")
        print(f"ALL SIMULATIONS COMPLETE")
        print(f"{'='*70}")
        print(f"Total simulations: {len(results)}")
        print(f"Results directory: {self.dirs['results']}")
        print(f"Database: {DB_PATH}")
        print(f"{'='*70}\n")
        
        return str(self.dirs['results'])


def main():
    parser = argparse.ArgumentParser(
        description="Run QuizComp LLM teacher simulations"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="detailed",
        choices=["detailed", "simple"],
        help="Type of prompt to use"
    )
    parser.add_argument(
        "--filter-persona-ids",
        type=int,
        nargs="+",
        help="Only run simulations for these persona IDs"
    )
    parser.add_argument(
        "--personas-file",
        type=str,
        help="Path to teacher personas JSON file"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("Please set it before running simulations.")
        sys.exit(1)
    
    # Run pipeline
    pipeline = QuizCompLLMStudyPipeline(
        prompt_type=args.prompt_type,
        filter_persona_ids=args.filter_persona_ids,
        personas_path=args.personas_file
    )
    
    results_dir = pipeline.run_simulations()
    
    print(f"\n Simulation complete!")
    print(f" Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
