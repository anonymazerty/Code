import os
import json
import time
import logging
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import datetime as dt
from pathlib import Path

from quizcomp_llm_study.interactive_llm_teacher_agent import InteractiveLLMTeacherAgent
from quizcomp_llm_study.teacher_persona import TeacherPersona
from quizcomp_llm_study.db_saver import SimulationDatabaseSaver
from quizcomp_llm_study.config import MAX_ITERATIONS, MODEL_TIERS


@dataclass
class InteractiveSimulationResult:
    """Results from an interactive teacher simulation."""
    persona_id: int
    run_number: int
    session_id: str
    simulation_type: str  # "interactive"
    
    # Initial parameters
    initial_topic_distribution: List[float]
    initial_difficulty_distribution: List[float]
    initial_num_mcqs: int
    data_uuid: str
    model_path: str
    alfa_value: float
    
    # Composition metrics
    num_iterations: int
    final_quiz_id: int
    final_match_quality: float
    final_quiz_size: int
    
    # Composition trajectory (actual API interactions)
    composition_attempts: List[Dict[str, Any]]
    
    # Survey responses (Q1-Q6)
    survey_responses: Dict[str, int]
    
    # Session metadata
    duration_seconds: float
    timestamp: str
    model_name: str
    
    # Cost tracking
    total_tokens: int = 0
    total_llm_calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InteractiveQuizCompSimulator:
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        max_iterations: int = MAX_ITERATIONS,
        seed: int = 42,
        prompt_variant: str = "detailed"
    ):
        
        self.api_base_url = api_base_url
        self.max_iterations = max_iterations
        self.seed = seed
        self.prompt_variant = prompt_variant
        
        print(f"Initialized Interactive QuizComp Simulator")
        print(f"  API Base URL: {api_base_url}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Prompt Variant: {prompt_variant}")
    
    def run_interactive_simulation(
        self,
        persona: TeacherPersona,
        run_number: int = 0
    ) -> InteractiveSimulationResult:
        
        start_time = time.time()
        session_id = f"interactive_{persona.persona_id}_run{run_number}_{int(start_time)}"
        
        print(f"\n{'='*70}")
        print(f"INTERACTIVE SIMULATION for Persona {persona.persona_id} (Run {run_number})")
        print(f"Session ID: {session_id}")
        print(f"{'='*70}")
        
        # Extract initial parameters from persona's first composition attempt
        if not persona.compose_attempts:
            raise ValueError(f"Persona {persona.persona_id} has no composition attempts!")
        
        first_attempt = persona.compose_attempts[0]
        initial_topic = first_attempt.get('teacher_topic_json')
        initial_difficulty = first_attempt.get('teacher_level_json')
        initial_num_mcqs = first_attempt.get('num_mcqs', 10)
        
        
        # Determine model path 
        model_path = "models/dqn_t1_math_r2"  
        alfa_value = 0.5  # Default alfa
        
        print(f"\nInitial Parameters (from persona's first attempt):")
        print(f"  Topics: {initial_topic}")
        print(f"  Difficulty: {initial_difficulty}")
        print(f"  Num MCQs: {initial_num_mcqs}")
        print(f"  Num MCQs: {initial_num_mcqs}")
        print(f"  Model: {model_path}")
        print(f"  Alfa: {alfa_value}")
        print(f"  (Universe will be generated automatically)")
        
        # Create interactive agent
        model_name = MODEL_TIERS["all"][0]
        agent = InteractiveLLMTeacherAgent(
            initial_topic_distribution=initial_topic,
            initial_difficulty_distribution=initial_difficulty,
            initial_num_mcqs=initial_num_mcqs,
            model_path=model_path,
            alfa_value=alfa_value,
            api_base_url=self.api_base_url,
            seed=self.seed + persona.persona_id + run_number,
            prompt_variant=self.prompt_variant
        )
        
        # Run interactive composition session
        print(f"\n Starting Interactive Composition Session ")
        session_summary = agent.start_composition_session(max_iterations=self.max_iterations)
        
        # Complete survey based on actual experience
        print(f"\n Agent Completing Survey ")
        survey_responses = agent.complete_survey_after_session(session_summary)
        
        # Build result
        duration = time.time() - start_time
        final_quiz = session_summary['final_quiz']
        
        result = InteractiveSimulationResult(
            persona_id=persona.persona_id,
            run_number=run_number,
            session_id=session_id,
            simulation_type="interactive",
            initial_topic_distribution=initial_topic,
            initial_difficulty_distribution=initial_difficulty,
            initial_num_mcqs=initial_num_mcqs,
            data_uuid=agent.data_uuid,  # Use the generated UUID
            model_path=model_path,
            alfa_value=alfa_value,
            num_iterations=session_summary['iterations'],
            final_quiz_id=session_summary['final_quiz_id'],
            final_match_quality=float(final_quiz.get('targetMatch', 0)),
            final_quiz_size=final_quiz.get('num_mcqs', initial_num_mcqs),
            composition_attempts=session_summary['interaction_history'],
            survey_responses=survey_responses,
            duration_seconds=duration,
            timestamp=dt.datetime.now().isoformat(),
            model_name=model_name,
            total_tokens=agent.total_tokens,
            total_llm_calls=agent.total_llm_calls
        )
        
        print(f"\n{'='*70}")
        print(f"INTERACTIVE SIMULATION COMPLETE")
        print(f"Duration: {duration:.1f}s")
        print(f"Iterations: {result.num_iterations}")
        print(f"Final Quiz ID: {result.final_quiz_id}")
        print(f"Final Match: {result.final_match_quality:.2%}")
        print(f"\n Cost tracking:")
        print(f"  Total tokens: {result.total_tokens}")
        print(f"  Total LLM calls: {result.total_llm_calls}")
        print(f"\nLLM Survey Responses:")
        for key, value in survey_responses.items():
            if key != 'reasoning':
                print(f"  {key}: {value}/5")
        print(f"\nReal Teacher Survey Responses (for comparison):")
        print(f"  accomplishment: {persona.survey_q1_accomplishment}/5")
        print(f"  effort: {persona.survey_q2_effort}/5")
        print(f"  mental_demand: {persona.survey_q3_mental_demand}/5")
        print(f"  controllability: {persona.survey_q4_controllability}/5")
        print(f"  temporal_demand: {persona.survey_q5_temporal_demand}/5")
        print(f"  satisfaction: {persona.survey_q6_satisfaction}/5")
        print(f"{'='*70}")
        
        return result
    
    def run_simulation_from_scratch(
        self,
        persona_id: int,
        run_number: int,
        initial_topic_distribution: List[float],
        initial_difficulty_distribution: List[float],
        initial_num_mcqs: int,
        model_path: str = "models/dqn_t1_math_r2",  
        alfa_value: float = 0.5,
        prompt_variant: str = None
    ) -> InteractiveSimulationResult:
       
        # Use provided variant or simulator's default
        variant = prompt_variant if prompt_variant is not None else self.prompt_variant
        # Use provided variant or simulator's default
        variant = prompt_variant if prompt_variant is not None else self.prompt_variant
        start_time = time.time()
        session_id = f"interactive_custom_{persona_id}_run{run_number}_{int(start_time)}"
        
        print(f"\n{'='*70}")
        print(f"CUSTOM INTERACTIVE SIMULATION (Persona {persona_id}, Run {run_number})")
        print(f"Session ID: {session_id}")
        print(f"{'='*70}")
        
        # Create interactive agent
        model_name = MODEL_TIERS["all"][0]
        agent = InteractiveLLMTeacherAgent(
            initial_topic_distribution=initial_topic_distribution,
            initial_difficulty_distribution=initial_difficulty_distribution,
            initial_num_mcqs=initial_num_mcqs,
            model_path=model_path,
            alfa_value=alfa_value,
            api_base_url=self.api_base_url,
            seed=self.seed + persona_id + run_number,
            prompt_variant=variant
        )
        
        # Run interactive composition session
        session_summary = agent.start_composition_session(max_iterations=self.max_iterations)
        
        # Complete survey
        survey_responses = agent.complete_survey_after_session(session_summary)
        
        # Build result
        duration = time.time() - start_time
        final_quiz = session_summary['final_quiz']
        
        result = InteractiveSimulationResult(
            persona_id=persona_id,
            run_number=run_number,
            session_id=session_id,
            simulation_type="interactive",
            initial_topic_distribution=initial_topic_distribution,
            initial_difficulty_distribution=initial_difficulty_distribution,
            initial_num_mcqs=initial_num_mcqs,
            data_uuid=agent.data_uuid,  # Use the generated UUID
            model_path=model_path,
            alfa_value=alfa_value,
            num_iterations=session_summary['iterations'],
            final_quiz_id=session_summary['final_quiz_id'],
            final_match_quality=float(final_quiz.get('targetMatch', 0)),
            final_quiz_size=final_quiz.get('num_mcqs', initial_num_mcqs),
            composition_attempts=session_summary['interaction_history'],
            survey_responses=survey_responses,
            duration_seconds=duration,
            timestamp=dt.datetime.now().isoformat(),
            model_name=model_name,
            total_tokens=agent.total_tokens,
            total_llm_calls=agent.total_llm_calls
        )
        
        print(f"\n{'='*70}")
        print(f"CUSTOM INTERACTIVE SIMULATION COMPLETE")
        print(f"Duration: {duration:.1f}s")
        print(f"Iterations: {result.num_iterations}")
        print(f"Final Match: {result.final_match_quality:.2%}")
        print(f"\n Cost tracking:")
        print(f"  Total tokens: {result.total_tokens}")
        print(f"  Total LLM calls: {result.total_llm_calls}")
        print(f"{'='*70}")
        
        return result


class MultipleInteractiveRunsRunner:
   
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        output_dir: str = "quizcomp_llm_study/interactive_results",
        db_saver: SimulationDatabaseSaver = None,
        prompt_variant: str = "detailed"
    ):
        self.api_base_url = api_base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_saver = db_saver
        self.prompt_variant = prompt_variant
        
        self.simulator = InteractiveQuizCompSimulator(
            api_base_url=api_base_url,
            prompt_variant=prompt_variant
        )
    
    def get_existing_simulations(self) -> Dict[int, List[int]]:
        if not self.db_saver:
            return {}
        
        import sqlite3
        existing = {}
        
        try:
            conn = sqlite3.connect(self.db_saver.db_path)
            cursor = conn.cursor()
            
            # Check if simulation_type column exists
            cursor.execute("PRAGMA table_info(simulations)")
            columns = [col[1] for col in cursor.fetchall()]
            has_sim_type = 'simulation_type' in columns
            
            if has_sim_type:
                cursor.execute("""
                    SELECT DISTINCT persona_id, run_number
                    FROM simulations
                    WHERE simulation_type = 'interactive'
                    ORDER BY persona_id, run_number
                """)
            else:
                cursor.execute("""
                    SELECT DISTINCT persona_id, run_number
                    FROM simulations
                    ORDER BY persona_id, run_number
                """)
            
            for persona_id, run_number in cursor.fetchall():
                if persona_id not in existing:
                    existing[persona_id] = []
                existing[persona_id].append(run_number)
            
            conn.close()
        except Exception as e:
            print(f"Warning: Could not check existing simulations: {e}")
            return {}
        
        return existing
    
    def run_for_all_personas(
        self,
        personas: List[TeacherPersona],
        runs_per_persona: int = 1,
        resume: bool = True
    ):
        
        # Check what's already been completed
        existing_sims = self.get_existing_simulations() if resume else {}
        
        # Calculate what needs to be done
        total_needed = 0
        total_existing = 0
        for persona in personas:
            existing_runs = existing_sims.get(persona.persona_id, [])
            total_existing += len(existing_runs)
            total_needed += runs_per_persona - len(existing_runs)
        
        print(f"\n{'='*70}")
        print(f"RUNNING INTERACTIVE SIMULATIONS")
        print(f"{'='*70}")
        print(f"Total personas: {len(personas)}")
        print(f"Runs per persona: {runs_per_persona}")
        if resume:
            print(f"Existing simulations: {total_existing}")
            print(f"Remaining simulations: {total_needed}")
        else:
            print(f"Total simulations: {len(personas) * runs_per_persona}")
        print(f"API Base URL: {self.api_base_url}")
        print(f"Prompt Variant: {self.prompt_variant}")
        print(f"{'='*70}\n")
        
        results = []
        skipped = 0
        
        for persona in personas:
            existing_runs = existing_sims.get(persona.persona_id, [])
            
            # Skip persona entirely if all runs are complete
            if resume and len(existing_runs) >= runs_per_persona:
                skipped += runs_per_persona
                print(f"⏭  SKIP Persona {persona.persona_id} - all {runs_per_persona} runs already complete")
                continue
            
            for run_num in range(runs_per_persona):  # 0-based indexing: 0, 1, 2, 3, 4
                # Skip this specific run if already completed
                if resume and run_num in existing_runs:
                    skipped += 1
                    print(f"⏭  Skip persona {persona.persona_id}, run {run_num} (already done)")
                    continue
                
                try:
                    print(f"\n{'▶'*3} Running persona {persona.persona_id}, run {run_num} {'▶'*3}")
                    result = self.simulator.run_interactive_simulation(
                        persona=persona,
                        run_number=run_num
                    )
                    
                    # Save to database
                    if self.db_saver:
                        self.db_saver.save_interactive_simulation(result, persona)
                        print(f" Saved to database")
                    
                    # Save to JSON file
                    result_file = self.output_dir / f"interactive_p{persona.persona_id}_r{run_num}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                    print(f" Saved to {result_file}")
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"\n ERROR: Failed simulation for persona {persona.persona_id}, run {run_num}")
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\n{'='*70}")
        print(f"ALL INTERACTIVE SIMULATIONS COMPLETE")
        if resume:
            print(f"Skipped (already done): {skipped}")
            print(f"New simulations: {len(results)}")
            print(f"Total in database: {skipped + len(results)}")
        else:
            print(f"Successful: {len(results)}/{len(personas) * runs_per_persona}")
        print(f"Results saved to: {self.output_dir}")
        if self.db_saver:
            print(f"Database: {self.db_saver.db_path}")
        print(f"{'='*70}")
        
        return results
