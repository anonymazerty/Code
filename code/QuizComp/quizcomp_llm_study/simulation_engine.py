import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import datetime as dt
from pathlib import Path

from quizcomp_llm_study.teacher_persona import TeacherPersona
from quizcomp_llm_study.llm_teacher_agent import LLMTeacherAgent
from quizcomp_llm_study.db_saver import SimulationDatabaseSaver
from quizcomp_llm_study.config import (
    MAX_ITERATIONS, QUESTIONS_PER_QUIZ, MODEL_TIERS, PROMPT_TYPE
)


@dataclass
class SimulationResult:
    """Results from a single teacher simulation run."""
    persona_id: int
    run_number: int
    session_id: str
    
    # Composition metrics
    num_iterations: int
    final_quiz_size: int
    final_match_quality: float
    
    # Composition trajectory
    composition_attempts: List[Dict[str, Any]]
    
    # Survey responses (Q1-Q6 only)
    survey_responses: Dict[str, int]
    
    # Session metadata
    duration_seconds: float
    timestamp: str
    prompt_type: str
    model_name: str
    
    # Cost tracking
    total_tokens: int = 0
    total_llm_calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuizCompSimulator:
    def __init__(
        self,
        prompt_type: str = PROMPT_TYPE,
        max_iterations: int = MAX_ITERATIONS,
        questions_per_quiz: int = QUESTIONS_PER_QUIZ,
        seed: int = 42
    ):
        self.prompt_type = prompt_type
        self.max_iterations = max_iterations
        self.questions_per_quiz = questions_per_quiz
        self.seed = seed
        
        print(f"Initialized QuizComp Simulator")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Questions per quiz: {questions_per_quiz}")
        print(f"  Prompt type: {prompt_type}")
    
    def run_simulation(
        self,
        persona: TeacherPersona,
        run_number: int = 0
    ) -> SimulationResult:
    
        start_time = time.time()
        session_id = f"sim_{persona.persona_id}_run{run_number}_{int(start_time)}"
        
        print(f"\n{'='*70}")
        print(f"Running simulation for Persona {persona.persona_id} (Run {run_number})")
        print(f"Session ID: {session_id}")
        print(f"{'='*70}")
        
        # Create LLM teacher agent
        model_name = MODEL_TIERS["all"][0]
        agent = LLMTeacherAgent(
            persona=persona,
            prompt_type=self.prompt_type,
            seed=self.seed + persona.persona_id + run_number
        )
        
        # Run composition iterations
        composition_attempts = []
        previous_quizzes = []
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # Agent decides on composition parameters
            composition_request, reasoning = agent.compose_quiz(
                iteration=iteration,
                previous_quizzes=previous_quizzes
            )
            
            print(f"Action: {composition_request.get('action', 'compose')}")
            print(f"Mode: {composition_request.get('mode', 'fresh')}")
            print(f"Reasoning: {composition_request.get('reasoning', 'N/A')[:100]}")
            
            # Check if agent wants to accept current quiz
            if composition_request.get('action') == 'accept' and previous_quizzes:
                print("Agent accepted current quiz. Ending composition.")
                break
            
            # Simulate quiz generation (mock)
            generated_quiz = self._mock_generate_quiz(
                composition_request,
                iteration
            )
            
            composition_attempts.append({
                'iteration': iteration,
                'request': composition_request,
                'generated_quiz': generated_quiz,
                'timestamp': dt.datetime.now().isoformat()
            })
            
            previous_quizzes.append(generated_quiz)
            
            print(f"Generated quiz: {generated_quiz['num_mcqs']} questions, "
                  f"match={generated_quiz['target_match']:.2%}")
        
        # Build trajectory summary
        trajectory_summary = self._build_trajectory_summary(composition_attempts)
        
        # Complete survey
        print(f"\n--- Completing Survey ---")
        survey_responses = agent.complete_survey(trajectory_summary)
        
        print(f"Survey responses:")
        for key, value in survey_responses.items():
            print(f"  {key}: {value}/5")
        
        # Calculate final metrics
        final_quiz = previous_quizzes[-1] if previous_quizzes else {}
        duration = time.time() - start_time
        
        result = SimulationResult(
            persona_id=persona.persona_id,
            run_number=run_number,
            session_id=session_id,
            num_iterations=len(composition_attempts),
            final_quiz_size=final_quiz.get('num_mcqs', 0),
            final_match_quality=final_quiz.get('target_match', 0.0),
            composition_attempts=composition_attempts,
            survey_responses=survey_responses,
            duration_seconds=duration,
            timestamp=dt.datetime.now().isoformat(),
            prompt_type=self.prompt_type,
            model_name=model_name
        )
        
        print(f"\n{'='*70}")
        print(f"Simulation Complete")
        print(f"Duration: {duration:.1f}s")
        print(f"Iterations: {result.num_iterations}")
        print(f"Final quiz: {result.final_quiz_size} questions, "
              f"match={result.final_match_quality:.2%}")
        print(f"{'='*70}")
        
        return result
    
    def _mock_generate_quiz(
        self,
        composition_request: Dict[str, Any],
        iteration: int
    ) -> Dict[str, Any]:
        
        import random
        
        topic_dist = composition_request.get('topic_distribution', [0.33, 0.33, 0.34])
        diff_dist = composition_request.get('difficulty_distribution', [0.3, 0.4, 0.3])
        
        # Simulate quiz generation
        num_mcqs = self.questions_per_quiz
        
        # Mock match quality (improves with iterations)
        base_match = 0.6 + (iteration * 0.08)
        match_quality = min(0.95, base_match + random.uniform(-0.1, 0.1))
        
        return {
            'num_mcqs': num_mcqs,
            'topic_distribution': topic_dist,
            'difficulty_distribution': diff_dist,
            'target_match': match_quality,
            'generation_time_s': random.uniform(2.0, 5.0)
        }
    
    def _build_trajectory_summary(self, composition_attempts: List[Dict]) -> str:
        """Build trajectory summary for survey prompt."""
        lines = []
        lines.append("QUIZ COMPOSITION SESSION TRAJECTORY")
        lines.append(f"Total iterations: {len(composition_attempts)}")
        lines.append("")
        
        for attempt in composition_attempts:
            iteration = attempt['iteration']
            request = attempt['request']
            quiz = attempt['generated_quiz']
            
            lines.append(f"--- Iteration {iteration + 1} ---")
            lines.append(f"Mode: {request.get('mode', 'N/A')}")
            lines.append(f"Topic targets: {request.get('topic_distribution', [])}")
            lines.append(f"Difficulty targets: {request.get('difficulty_distribution', [])}")
            lines.append(f"Generated: {quiz.get('num_mcqs', 0)} questions")
            lines.append(f"Match quality: {quiz.get('target_match', 0):.2%}")
            lines.append(f"Reasoning: {request.get('reasoning', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)


class MultipleRunsRunner:

    def __init__(
        self,
        output_dir: str = "quizcomp_llm_study/results",
        db_saver: SimulationDatabaseSaver = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_saver = db_saver
    
    def run_all(
        self,
        personas: List[TeacherPersona],
        simulations_per_persona: int = 1,
        prompt_type: str = PROMPT_TYPE
    ):
       
        print(f"\n{'='*70}")
        print(f"STARTING BATCH SIMULATION")
        print(f"{'='*70}")
        print(f"Personas: {len(personas)}")
        print(f"Runs per persona: {simulations_per_persona}")
        print(f"Total simulations: {len(personas) * simulations_per_persona}")
        print(f"Prompt type: {prompt_type}")
        print(f"{'='*70}\n")
        
        simulator = QuizCompSimulator(prompt_type=prompt_type)
        
        all_results = []
        
        for persona in personas:
            for run_num in range(simulations_per_persona):
                try:
                    result = simulator.run_simulation(persona, run_num)
                    all_results.append(result)
                    
                    # Save to database
                    if self.db_saver:
                        self.db_saver.save_simulation(result, persona)
                    
                    # Save individual result
                    result_file = self.output_dir / f"persona_{persona.persona_id}_run{run_num}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                
                except Exception as e:
                    print(f"ERROR in persona {persona.persona_id} run {run_num}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save summary
        summary_file = self.output_dir / "summary.json"
        summary = {
            'total_simulations': len(all_results),
            'personas': len(personas),
            'runs_per_persona': simulations_per_persona,
            'prompt_type': prompt_type,
            'timestamp': dt.datetime.now().isoformat(),
            'results': [r.to_dict() for r in all_results]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"BATCH SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total simulations: {len(all_results)}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}\n")
        
        return all_results
