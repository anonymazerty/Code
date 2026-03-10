import os
import json
import time
import logging
import datetime as dt
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from llm_user_study.persona_builder import LLMPersona
from llm_user_study.llm_student_agent import LLMStudentAgent
from llm_user_study.db_saver import SimulationDatabaseSaver
from llm_user_study.config import (
    TOPIC, BENCHMARK, ORCHESTRATOR_TYPE, MODEL_NAME,
    PRETEST_N, STEPS_N, QUESTIONS_PER_STEP, OBJECTIVES, PROMPT_TYPE
)

from real_user_study.live_session import RealUserEngine
from real_user_study.loader import load_topic_questions_with_difficulties


@dataclass
class SimulationResult: # add questions for each steps (just use same learner db for llm db)
    """Results from a single simulation run."""
    persona_id: int
    run_number: int
    session_id: str
    
    # Performance metrics
    pretest_score: int
    pretest_mastery: float
    final_mastery: float
    mastery_improvement: float
    
    # Interaction metrics
    total_questions_answered: int
    total_steps: int
    avg_response_time: float
    
    # Learning trajectory
    mastery_per_step: List[float]
    accuracy_per_step: List[float]
    rewards_per_step: Dict[str, List[float]]
    
    # Step-by-step data for database
    learning_steps_data: List[Dict[str, Any]] 
    
    # Survey responses
    survey_responses: Dict[str, Any]
    
    # Session metadata
    duration_seconds: float
    timestamp: str
    orchestrator_type: str
    prompt_type: str
    
    # Orchestrator calls (optional, has default)
    orchestrator_calls: List[Dict[str, Any]] = None

    # LLM student calls: one record per question + one for survey
    llm_student_calls: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TestRecoSimulator:
    """
    Simulates TestReco learning sessions with LLM persona agents.
    """
    
    def __init__(
        self,
        topic: str = TOPIC,
        benchmark: str = BENCHMARK,
        orchestrator_type: str = ORCHESTRATOR_TYPE,
        prompt_type: str = PROMPT_TYPE,
        steps_n: int = STEPS_N,
        questions_per_step: int = QUESTIONS_PER_STEP,
        objectives: List[str] = None,
        seed: int = 42
    ):
        self.topic = topic
        self.benchmark = benchmark
        self.orchestrator_type = orchestrator_type
        self.prompt_type = prompt_type
        self.steps_n = steps_n
        self.questions_per_step = questions_per_step
        self.objectives = objectives or OBJECTIVES
        self.seed = seed
        
        # Load questions and difficulties
        print(f"Loading questions for {topic}...")
        self.questions, self.difficulties = load_topic_questions_with_difficulties(topic)
        print(f"Loaded {len(self.questions)} questions")
        
        # Create simulation engine
        self.engine = RealUserEngine(
            topic=topic,
            questions=self.questions,
            difficulties=self.difficulties,
            max_steps=steps_n,
            questions_per_step=questions_per_step,
            objectives=self.objectives,
            seed=seed
        )
        
        # Load policies for orchestrator
        self._setup_orchestrator()
        
    def _setup_orchestrator(self):
        """Initialize the orchestrator with trained policies."""
        from real_user_study.settings import POLICY_FOLDERS
        import importlib
        
        # Configure logging to write to logs/ directory
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"orchestrator_{timestamp}.log")
        
        # Configure logging with both file and console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Orchestrator logs will be saved to: {log_file}")
        
        # Dynamically import the orchestrator class
        if self.orchestrator_type == "tool_call":
            from orchestrator.tool_call_orchestrator import ToolCallOrchestrator
            orchestrator_class = ToolCallOrchestrator
        elif self.orchestrator_type == "reflection":
            from orchestrator.reflection_based_orchestrator import ReflectionBasedOrchestrator
            orchestrator_class = ReflectionBasedOrchestrator
        elif self.orchestrator_type == "context":
            from orchestrator.context_based_orchestrator import ContextBasedOrchestrator
            orchestrator_class = ContextBasedOrchestrator
        else:
            raise ValueError(f"Unknown orchestrator type: {self.orchestrator_type}")
        
        # Initialize orchestrator
        try:
            from generators.factory import model_factory
            llm = model_factory(MODEL_NAME)
            
            logging.info(f"Created LLM model: {MODEL_NAME}, type={type(llm)}")
            print(f"Created LLM model: {MODEL_NAME}, type={type(llm)}")
            
            # Build policy configs like real_user_study does
            policy_configs = self.engine._make_policy_configs(POLICY_FOLDERS)
            
            logging.info(f"Built {len(policy_configs)} policy configs")
            print(f"Built {len(policy_configs)} policy configs")
            
            self.engine.orchestrator = orchestrator_class(
                env=self.engine.env,
                llm=llm,
                policy_configs=policy_configs,
                verbose=True,
                objectives=self.objectives
            )
            logging.info(f"Initialized {self.orchestrator_type} orchestrator")
            print(f"Initialized {self.orchestrator_type} orchestrator")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"Could not initialize orchestrator: {e}")
            logging.error(f"Traceback:\n{error_details}")
            print(f"Warning: Could not initialize orchestrator: {e}")
            print(f"Traceback:\n{error_details}")
            self.engine.orchestrator = None
    
    def simulate_session(
        self,
        persona: LLMPersona,
        run_number: int = 0
    ) -> SimulationResult:
        """
        Simulate a complete learning session for a persona.
        
        Args:
            persona: The LLM persona to simulate
            run_number: Which run this is (0-2 typically)
        
        Returns:
            SimulationResult with all session data
        """
        print(f"\n{'='*60}")
        print(f"Simulating session for Persona {persona.persona_id}, Run {run_number}")
        # Get model from current MODEL_TIERS configuration
        from llm_user_study.config import MODEL_TIERS
        current_model = MODEL_TIERS[persona.education_level][0]
        print(f"Education: {persona.education_level}, Model: {current_model}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        session_id = f"persona_{persona.persona_id}_run_{run_number}_{int(start_time)}"
        
        # Create student agent - pass orchestrator's LLM model for efficiency
        agent = LLMStudentAgent(persona, prompt_type=self.prompt_type, seed=self.seed + run_number, llm_model=self.engine.orchestrator.llm)
        
        # Reset environment
        self.engine._manual_reset_state()
        
        # Phase 1: Pretest (record real user's pretest data - matching real_user_study)
        print("Phase 1: Pretest (real user's pretest stored)")
        
        # Use the real user's pretest data for logging
        pretest_qids = persona.pretest_questions
        pretest_correctness = persona.pretest_correctness
        pretest_score = persona.qualification_score
        pretest_mastery = persona.pre_qualification_score
        
        print(f"Pretest (real user data): {pretest_score}/{PRETEST_N} correct, Mastery: {pretest_mastery:.2%}")
        
        # Match real_user_study exactly: hard-code mastery to 0.4 for learning phase
        # (real study ignores pretest and starts everyone at 0.4)
        self.engine.env.mastery[0] = 0.4
        print(f"Learning phase starting mastery: {self.engine.env.mastery[0]:.2%}")
        
        # Phase 2: Learning Loop (LLM simulates the learner)
        print(f"\nPhase 2: Learning Loop ({self.steps_n} steps) - LLM simulates learner")
        learning_results = self._run_learning_loop(agent)

        # Phase 3: Survey (LLM simulates survey responses)
        print("\nPhase 3: Survey - LLM simulates responses")
        survey_responses, survey_token_info = self._run_survey(agent, learning_results)
        
        # Collect results
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract metrics
        mastery_trajectory = [step['mastery_after'] for step in learning_results['steps']]
        accuracy_trajectory = [step['rolling_accuracy'] for step in learning_results['steps']]
        
        rewards_per_step = {obj: [] for obj in self.objectives}
        for step in learning_results['steps']:
            for obj in self.objectives:
                rewards_per_step[obj].append(step['rewards'].get(obj, 0.0))
        
        result = SimulationResult(
            persona_id=persona.persona_id,
            run_number=run_number,
            session_id=session_id,
            pretest_score=pretest_score,
            pretest_mastery=pretest_mastery,
            final_mastery=mastery_trajectory[-1] if mastery_trajectory else pretest_mastery,
            mastery_improvement=(mastery_trajectory[-1] if mastery_trajectory else pretest_mastery) - pretest_mastery,
            total_questions_answered=learning_results['total_questions'],
            total_steps=len(learning_results['steps']),
            avg_response_time=learning_results['avg_response_time'],
            mastery_per_step=mastery_trajectory,
            accuracy_per_step=accuracy_trajectory,
            rewards_per_step=rewards_per_step,
            learning_steps_data=learning_results['steps'], 
            survey_responses=survey_responses,
            duration_seconds=duration,
            timestamp=dt.datetime.now().isoformat(),
            orchestrator_type=self.orchestrator_type,
            prompt_type=self.prompt_type,
            orchestrator_calls=learning_results.get('orchestrator_calls', []),
            llm_student_calls=learning_results.get('llm_student_calls', []) + [
                {
                    'call_type': 'survey',
                    'step_index': None,
                    'question_index': None,
                    'latency_s': survey_token_info.get('latency_s'),
                    'input_tokens': survey_token_info.get('input_tokens', 0),
                    'output_tokens': survey_token_info.get('output_tokens', 0),
                }
            ],
        )
        
        print(f"\nSession complete in {duration:.1f}s")
        print(f"Final mastery: {result.final_mastery:.2%} (Δ{result.mastery_improvement:+.2%})")
        
        return result
    
    def _run_learning_loop(self, agent: LLMStudentAgent) -> Dict[str, Any]:
        """Run the learning loop with orchestrator recommendations."""
        
        steps_data = []
        orchestrator_calls = []
        llm_student_calls = []
        total_questions = 0
        response_times = []
        
        for step_idx in range(self.steps_n):
            print(f"\n--- Step {step_idx + 1}/{self.steps_n} ---")
            
            # Get current observation
            obs = self.engine.env._get_obs()
            
            # Get orchestrator recommendation (action)
            action = None
            action_info = {}
            orchestrator_call_start = time.time()
            
            if self.engine.orchestrator:
                try:
                    action, action_info = self.engine.orchestrator.select_action(
                        obs, 
                        deterministic=False
                    )
                    # Extract integer action from dictionary if needed
                    if isinstance(action, dict) and 'action' in action:
                        action = action['action']
                except Exception as e:
                    print(f"Orchestrator failed: {e}, using random action")
                    action = np.random.choice(len(self.objectives))
            else:
                # Random action if no orchestrator
                action = np.random.choice(len(self.objectives))
            
            orchestrator_call_latency = time.time() - orchestrator_call_start
            
            # Capture orchestrator call metadata
            orchestrator_calls.append({
                'step_index': step_idx,
                'observation': obs.tolist() if hasattr(obs, 'tolist') else obs,
                'strategy': self.objectives[action] if action is not None else None,
                'action': int(action) if action is not None else None,
                'latency_s': orchestrator_call_latency,
                'input_tokens': action_info.get('input_tokens'),
                'output_tokens': action_info.get('output_tokens'),
                'total_tokens': action_info.get('total_tokens')
            })
            
            # Select questions based on action
            selected_qids = self.engine.env._select_questions_by_action(action)
            
            print(f"Action {action} ({self.objectives[action]}): Questions {selected_qids}")
            
            # Simulate answering each question
            mastery_before = self.engine.env.mastery[0]
            correctness = []
            chosen_answers = []  # Store what the LLM chose
            
            for qid in selected_qids:
                question = self.questions[qid]
                difficulty = self.difficulties[qid]['scaled_difficulty']
                
                chosen_idx, response_time, metadata = agent.answer_question(
                    question, difficulty
                )
                
                correct_idx = question.get('answer', 0)
                is_correct = (chosen_idx == correct_idx)
                correctness.append(is_correct)
                chosen_answers.append(chosen_idx)
                response_times.append(response_time)
                llm_student_calls.append({
                    'call_type': 'question',
                    'step_index': step_idx,
                    'question_index': len(correctness) - 1,
                    'latency_s': metadata.get('llm_latency_s'),
                    'input_tokens': metadata.get('llm_input_tokens', 0),
                    'output_tokens': metadata.get('llm_output_tokens', 0),
                })
                
                print(f"  Q{qid} (diff={difficulty:.2f}): {'✓' if is_correct else '✗'}")
            
            # Update environment manually (matching real_user_study pattern)
            qinfo_list = []
            for qid, correct in zip(selected_qids, correctness):
                qinfo = self.engine.env.question_skills_difficulty_map[qid]
                skills = np.array([0], dtype=int)
                
                self.engine.env.seen_materials.append((qid, qinfo["scaled_difficulty"], bool(correct)))
                
                qinfo_list.append({
                    "question_id": qid,
                    "skills": [self.topic],
                    "original_difficulty": qinfo["original_difficulty"],
                    "scaled_difficulty": qinfo["scaled_difficulty"],
                    "correct": bool(correct),
                    "skills_tested": skills,
                })
                
                self.engine.env._update_skill_difficulty_accuracy(
                    qid, bool(correct), skills, qinfo["original_difficulty"]
                )
            
            # Calculate rewards
            rewards = self.engine.env._calculate_batch_rewards(qinfo_list)
            
            # Update mastery
            self.engine.env._update_mastery(qinfo_list)
            
            # Update failed questions
            for qi in qinfo_list:
                self.engine.env._update_failed_question(
                    qi["question_id"],
                    qi["skills_tested"],
                    qi["correct"],
                    qi["original_difficulty"],
                    qi["scaled_difficulty"],
                )
            
            self.engine.env._update_aptitude_cache()
            self.engine.env._update_experience_cache()
            self.engine.env._update_gap_cache()
            
            self.engine.env.current_step += 1
            
            mastery_after = self.engine.env.mastery[0]
            rolling_accuracy = sum(correctness) / len(correctness) if correctness else 0.0
            
            # Update agent mastery
            agent.update_mastery(mastery_after)
            
            step_data = {
                'step_index': step_idx,
                'action': int(action),
                'selected_qids': selected_qids,
                'correctness': correctness,
                'chosen_answers': chosen_answers,  # Store chosen answers
                'mastery_before': float(mastery_before),
                'mastery_after': float(mastery_after),
                'rolling_accuracy': rolling_accuracy,
                'rewards': {
                    obj: float(rewards.get(obj, 0.0))
                    for obj in self.objectives
                }
            }
            
            steps_data.append(step_data)
            total_questions += len(selected_qids)
            
            print(f"  Mastery: {mastery_before:.2%} → {mastery_after:.2%} (Δ{mastery_after-mastery_before:+.2%})")
            print(f"  Accuracy: {rolling_accuracy:.1%}")
        
        avg_response_time = np.mean(response_times) if response_times else 0
        
        return {
            'steps': steps_data,
            'orchestrator_calls': orchestrator_calls,
            'llm_student_calls': llm_student_calls,
            'total_questions': total_questions,
            'avg_response_time': avg_response_time
        }
    
    def _run_survey(
        self,
        agent: LLMStudentAgent,
        learning_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run survey phase."""
        
        # Prepare session data for survey
        steps = learning_results['steps']
        initial_mastery = steps[0]['mastery_before'] if steps else 0.5
        final_mastery = steps[-1]['mastery_after'] if steps else 0.5
        mastery_change = final_mastery - initial_mastery
        
        # Build detailed learning trajectory
        trajectory_lines = []
        trajectory_lines.append(f"LEARNING SESSION TRAJECTORY")
        trajectory_lines.append(f"Initial Mastery: {initial_mastery:.2%}")
        trajectory_lines.append(f"Final Mastery: {final_mastery:.2%}")
        trajectory_lines.append(f"Improvement: {mastery_change:+.2%}")
        trajectory_lines.append("")
        
        for step in steps:
            step_idx = step['step_index']
            mastery_before = step['mastery_before']
            mastery_after = step['mastery_after']
            accuracy = step['rolling_accuracy']
            
            trajectory_lines.append(f"--- Step {step_idx + 1} ---")
            trajectory_lines.append(f"Mastery: {mastery_before:.2%} → {mastery_after:.2%} (Δ{mastery_after-mastery_before:+.2%})")
            trajectory_lines.append(f"Accuracy: {accuracy:.1%}")
            trajectory_lines.append("")
            
            # For each question in this step
            for i, qid in enumerate(step['selected_qids']):
                question = self.questions[qid]
                q_text = question.get('text', question.get('question', 'N/A'))
                options = question.get('options', [])
                correct_idx = question.get('answer', 0)
                chosen_idx = step.get('chosen_answers', [None])[i]
                was_correct = step['correctness'][i]
                difficulty = self.difficulties[qid]['scaled_difficulty']
                
                trajectory_lines.append(f"  Question {i+1} (ID: {qid}, Difficulty: {difficulty:.2f}):")
                trajectory_lines.append(f"  {q_text[:100]}{'...' if len(q_text) > 100 else ''}")
                trajectory_lines.append(f"  Options:")
                for opt_idx, opt in enumerate(options):
                    markers = []
                    if opt_idx == correct_idx:
                        markers.append("✓ CORRECT")
                    if opt_idx == chosen_idx:
                        markers.append("← YOUR CHOICE")
                    marker_str = " ".join(markers) if markers else ""
                    trajectory_lines.append(f"    {opt_idx}. {opt} {marker_str}")
                trajectory_lines.append(f"  Result: {'CORRECT ✓' if was_correct else 'INCORRECT ✗'}")
                trajectory_lines.append("")
            
            trajectory_lines.append("")
        
        learning_trajectory = "\n".join(trajectory_lines)
        
        session_data = {
            'mastery_change': mastery_change,
            'questions_answered': learning_results['total_questions'],
            'learning_trajectory': learning_trajectory
        }
        
        survey_responses, token_info = agent.complete_survey(session_data)

        print(f"Survey completed: {survey_responses}")

        return survey_responses, token_info
    
    def simulate_multiple_runs(
        self,
        persona: LLMPersona,
        num_runs: int = 3
    ) -> List[SimulationResult]:
        """Run multiple simulation sessions for a single persona."""
        results = []
        
        for run_idx in range(num_runs):
            result = self.simulate_session(persona, run_number=run_idx)
            results.append(result)
            
            # Brief pause between runs
            time.sleep(1)
        
        return results


class MultipleRunsRunner:
    """Run simulations for multiple personas and save to database."""
    
    def __init__(self, simulator: TestRecoSimulator):
        self.simulator = simulator
        self.db_saver = SimulationDatabaseSaver()
        self.saved_session_ids: List[int] = []
    
    def run_all_personas(
        self,
        personas: List[LLMPersona],
        runs_per_persona: int = 3,
        output_dir: str = "llm_user_study/results"
    ):
        """Run simulations for all personas."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        total_simulations = len(personas) * runs_per_persona
        print(f"\n{'='*70}")
        print(f"Starting batch simulation: {len(personas)} personas × {runs_per_persona} runs = {total_simulations} total")
        print(f"{'='*70}\n")
        
        for i, persona in enumerate(personas):
            print(f"\n[{i+1}/{len(personas)}] Processing Persona {persona.persona_id}")
            
            try:
                persona_results = self.simulator.simulate_multiple_runs(
                    persona,
                    num_runs=runs_per_persona
                )
                
                # Save each result to database
                for result in persona_results:
                    session_id = self.db_saver.save_simulation(persona, result)
                    self.saved_session_ids.append({
                        'llm_session_id': session_id,
                        'real_session_id': persona.session_id,
                        'persona_id': persona.persona_id,
                        'education': persona.education_level
                    })
                
                print(f"✓ Saved {len(persona_results)} runs to database")
                
            except Exception as e:
                print(f"✗ Error simulating persona {persona.persona_id}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total sessions saved to database: {len(self.saved_session_ids)}")
        print(f"\nSimulation Mapping (LLM Session → Real Learner):")
        print(f"-" * 70)
        for mapping in self.saved_session_ids[:20]:  # Show first 20
            print(f"  LLM Session {mapping['llm_session_id']:3d} simulates Real Learner {mapping['real_session_id']:3d} "
                  f"(Persona {mapping['persona_id']}, {mapping['education']})")
        if len(self.saved_session_ids) > 20:
            print(f"  ... and {len(self.saved_session_ids) - 20} more")
        print(f"{'='*70}")
