import sqlite3
import json
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv


_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

from llm_user_study.persona_builder import PersonaBuilder, LLMPersona
from llm_user_study.llm_student_agent import LLMStudentAgent
from llm_user_study.config import DB_PATH, TOPIC, MODEL_TIERS
from real_user_study.loader import load_topic_questions_with_difficulties
from generators.factory import model_factory
from orchestrator.langchain_wrapper import LangChainWrapper
from datetime import datetime, timezone


def load_real_learner_trajectory(session_id: int, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Load complete learning trajectory for a real learner."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get session info
    cursor.execute("""
        SELECT * FROM study_sessions WHERE id = ?
    """, (session_id,))
    session = cursor.fetchone()
    
    if not session:
        raise ValueError(f"Session {session_id} not found")
    
    # Get learning steps
    cursor.execute("""
        SELECT * FROM learning_steps
        WHERE session_id = ?
        ORDER BY step_index
    """, (session_id,))
    steps = cursor.fetchall()
    
    # Get attempts for each step
    cursor.execute("""
        SELECT * FROM attempts
        WHERE session_id = ? AND phase = 'learning'
        ORDER BY step_index, id
    """, (session_id,))
    attempts = cursor.fetchall()
    
    conn.close()
    
    # Organize attempts by step
    attempts_by_step = {}
    for attempt in attempts:
        step_idx = attempt['step_index']
        if step_idx not in attempts_by_step:
            attempts_by_step[step_idx] = []
        attempts_by_step[step_idx].append(dict(attempt))
    
    # Build trajectory data
    trajectory = {
        'session_id': session_id,
        'initial_mastery': session['pretest_mastery_init'],
        'final_mastery': session['final_mastery'],
        'steps': []
    }
    
    for step in steps:
        step_data = {
            'step_index': step['step_index'],
            'mastery_before': step['mastery_before'],
            'mastery_after': step['mastery_after'],
            'rolling_accuracy': step['rolling_accuracy'],
            'attempts': attempts_by_step.get(step['step_index'], [])
        }
        trajectory['steps'].append(step_data)
    
    return trajectory


def build_trajectory_string(trajectory: Dict[str, Any], questions: List[Dict]) -> str:
    """Build detailed trajectory string for LLM survey prompt."""
    lines = []
    lines.append("LEARNING SESSION TRAJECTORY")
    # Learning phase always starts at 0.4, not pretest mastery
    initial_learning_mastery = trajectory['steps'][0]['mastery_before'] if trajectory['steps'] else 0.4
    lines.append(f"Initial Mastery: {initial_learning_mastery:.2%}")
    lines.append(f"Final Mastery: {trajectory['final_mastery']:.2%}")
    lines.append(f"Improvement: {trajectory['final_mastery'] - initial_learning_mastery:+.2%}")
    lines.append("")
    
    for step_data in trajectory['steps']:
        step_idx = step_data['step_index']
        lines.append(f"--- Step {step_idx + 1} ---")
        lines.append(f"Mastery: {step_data['mastery_before']:.2%} → {step_data['mastery_after']:.2%} "
                    f"(Δ{step_data['mastery_after'] - step_data['mastery_before']:+.2%})")
        lines.append(f"Accuracy: {step_data['rolling_accuracy']:.1%}")
        lines.append("")
        
        # Show each question attempt
        for i, attempt in enumerate(step_data['attempts'], 1):
            q_idx = attempt['question_index']
            question = questions[q_idx]
            q_text = question.get('text', question.get('question', 'N/A'))
            options = question.get('options', [])
            
            lines.append(f"  Question {i} (ID: {q_idx}, Difficulty: {attempt['scaled_difficulty']:.2f}):")
            lines.append(f"  {q_text[:100]}{'...' if len(q_text) > 100 else ''}")
            lines.append(f"  Options:")
            
            for opt_idx, opt in enumerate(options):
                markers = []
                if opt_idx == attempt['correct_option_index']:
                    markers.append("✓ CORRECT")
                if opt_idx == attempt['chosen_option_index']:
                    markers.append("← YOUR CHOICE")
                marker_str = " ".join(markers) if markers else ""
                lines.append(f"    {opt_idx}. {opt} {marker_str}")
            
            result = "CORRECT ✓" if attempt['is_correct'] else "INCORRECT ✗"
            lines.append(f"  Result: {result}")
            lines.append("")
        
        lines.append("")
    
    return "\n".join(lines)


def select_test_retest_profiles(db_path: str = DB_PATH, profiles_per_level: int = 9) -> Dict[str, List[int]]:
    """
    Select profiles for test-retest analysis: N from each education level.
    
    Returns:
        Dict mapping education level to list of session IDs
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Map keys to actual database values (database has "high school" with space)
    edu_level_map = {
        'graduate': 'graduate',
        'undergraduate': 'undergraduate', 
        'high_school': 'high school'
    }
    
    profiles = {'graduate': [], 'undergraduate': [], 'high_school': []}
    
    for edu_key, edu_db_value in edu_level_map.items():
        cursor.execute("""
            SELECT DISTINCT s.id
            FROM study_sessions s
            JOIN users u ON s.user_id = u.id
            JOIN prolific_demographics pd ON u.username = pd.prolific_participant_id
            JOIN survey_responses sr ON s.id = sr.session_id
            WHERE (s.notes IS NULL OR s.notes NOT LIKE '%LLM%')
            AND pd.education_level = ?
            AND s.id IN (
                SELECT session_id FROM learning_steps 
                GROUP BY session_id HAVING COUNT(DISTINCT step_index) = 10
            )
            ORDER BY s.id
            LIMIT ?
        """, (edu_db_value, profiles_per_level))
        
        profiles[edu_key] = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return profiles


def simulate_survey_only(
    session_id: int, 
    model_name: str = None,
    run_number: int = None,
    db_path: str = DB_PATH
):
    """
    Simulate survey completion for a real learner's trajectory.
    
    Args:
        session_id: The real learner's session ID
        model_name: LLM model to use (if None, uses MODEL_TIERS based on education level)
        run_number: Run number for test-retest (if provided, saves to survey_test_retest table)
        db_path: Path to database
    """
    print(f"\n{'='*70}")
    print(f"SURVEY-ONLY SIMULATION")
    print(f"Real Learner Session: {session_id}")
    if run_number:
        print(f"Run: {run_number}")
    print(f"{'='*70}\n")
    
    # Load questions
    print("Loading questions...")
    questions, difficulties = load_topic_questions_with_difficulties(TOPIC)
    
    # Build persona from real learner
    print(f"Building persona from session {session_id}...")
    builder = PersonaBuilder(db_path)
    persona = builder.build_persona(session_id, persona_id=0)
    
    if not persona:
        print(f"✗ Could not build persona from session {session_id}")
        return None
    
    # Set questions dict
    questions_dict = {i: q for i, q in enumerate(questions)}
    persona.set_questions_dict(questions_dict)
    
    print(f"Persona: {persona.education_level}, Mastery: {persona.pre_qualification_score:.2%}")
    
    # Load real trajectory
    print("Loading real learner trajectory...")
    trajectory = load_real_learner_trajectory(session_id, db_path)
    print(f"Loaded {len(trajectory['steps'])} learning steps")
    
    # Build trajectory string
    trajectory_string = build_trajectory_string(trajectory, questions)
    
    # Get model - use provided model_name or fall back to MODEL_TIERS
    if model_name is None:
        model_name = MODEL_TIERS[persona.education_level][0]
    
    # Create LLM model - use agent's direct client for OpenRouter models (more reliable)
    # model_factory/LangChainWrapper has retry issues with OpenRouter
    print(f"\nCreating LLM model ({model_name})...")
    openrouter_models = ['llama-3-8b', 'llama-3-70b', 'gemma-2-9b-it', 'mistral-24b-instruct']
    
    if model_name in openrouter_models:
        # Use agent's internal OpenRouter client (bypasses LangChain retry issues)
        print(f"  (using direct OpenRouter client for {model_name})")
        agent = LLMStudentAgent(persona, seed=42, llm_model=None, model_name=model_name)
    else:
        try:
            llm = model_factory(model_name)
            wrapped_llm = LangChainWrapper(llm)
            agent = LLMStudentAgent(persona, seed=42, llm_model=wrapped_llm, model_name=model_name)
        except ValueError:
            # Model not in factory - let agent use its internal OpenAI/OpenRouter client
            print(f"  (using agent's internal client for {model_name})")
            agent = LLMStudentAgent(persona, seed=42, llm_model=None, model_name=model_name)
    
    # Create LLM agent
    print(f"Creating LLM agent...")
    
    # Prepare session data
    initial_learning_mastery = trajectory['steps'][0]['mastery_before'] if trajectory['steps'] else 0.4
    session_data = {
        'mastery_change': trajectory['final_mastery'] - initial_learning_mastery,
        'questions_answered': sum(len(s['attempts']) for s in trajectory['steps']),
        'learning_trajectory': trajectory_string
    }
    
    # Complete survey
    print("\nAsking LLM to complete survey based on real trajectory...")
    survey_responses, survey_token_info = agent.complete_survey(session_data)

    if survey_responses is None:
        print("✗ Error: LLM failed to complete survey")
        return
    
    print("\n" + "="*70)
    print("SURVEY RESPONSES")
    print("="*70)
    for key, value in survey_responses.items():
        if key != 'free_text':
            print(f"{key:30s}: {value}")
    if 'free_text' in survey_responses and survey_responses['free_text']:
        print(f"\nFree text response:")
        print(survey_responses['free_text'])
    print("="*70)
    
    # Save to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        if run_number is not None:
            # Save to test-retest table (for ICC analysis)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS survey_test_retest (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    education_level TEXT,
                    model_name TEXT,
                    run_number INTEGER,
                    accomplishment INTEGER,
                    effort_required INTEGER,
                    mental_demand INTEGER,
                    perceived_controllability INTEGER,
                    temporal_demand INTEGER,
                    frustration INTEGER,
                    trust INTEGER,
                    latency_s FLOAT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            _inp = survey_token_info.get('input_tokens', 0) or 0
            _out = survey_token_info.get('output_tokens', 0) or 0
            cursor.execute("""
                INSERT INTO survey_test_retest (
                    session_id, education_level, model_name, run_number,
                    accomplishment, effort_required, mental_demand,
                    perceived_controllability, temporal_demand,
                    frustration, trust,
                    latency_s, input_tokens, output_tokens, total_tokens,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                persona.education_level,
                model_name,
                run_number,
                survey_responses.get('accomplishment'),
                survey_responses.get('effort_required'),
                survey_responses.get('mental_demand'),
                survey_responses.get('perceived_controllability'),
                survey_responses.get('temporal_demand'),
                survey_responses.get('frustration'),
                survey_responses.get('trust'),
                survey_token_info.get('latency_s'),
                _inp,
                _out,
                _inp + _out,
                datetime.now(timezone.utc)
            ))
            print(f"\n✓ Saved to survey_test_retest (model={model_name}, run={run_number})")
        else:
            # Save to llm_survey_predictions table
            # Delete only entries for the same session AND model (to allow multiple models)
            cursor.execute("DELETE FROM llm_survey_predictions WHERE real_session_id = ? AND model_name = ?", (session_id, model_name))
            cursor.execute("""
                INSERT INTO llm_survey_predictions (
                    real_session_id,
                    accomplishment, effort_required, mental_demand,
                    perceived_controllability, temporal_demand,
                    frustration, trust, would_use_again, free_text,
                    created_at, model_name, education_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                survey_responses.get('accomplishment', None),
                survey_responses.get('effort_required', None),
                survey_responses.get('mental_demand', None),
                survey_responses.get('perceived_controllability', None),
                survey_responses.get('temporal_demand', None),
                survey_responses.get('frustration', None),
                survey_responses.get('trust', None),
                survey_responses.get('would_use_again', None),
                survey_responses.get('free_text', ''),
                datetime.now(timezone.utc),
                model_name,
                persona.education_level
            ))
            print(f"\n✓ Saved LLM survey prediction to database for session {session_id} (model={model_name})")
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"\n✗ Database save failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
    
    # Also save to JSON for reference
    output_dir = Path("llm_user_study/survey_only_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"survey_session_{session_id}.json"
    
    initial_learning_mastery = trajectory['steps'][0]['mastery_before'] if trajectory['steps'] else 0.4
    result = {
        'real_session_id': session_id,
        'persona': {
            'education_level': persona.education_level,
            'model_name': persona.model_name,
            'mastery': persona.pre_qualification_score
        },
        'trajectory': {
            'initial_mastery': initial_learning_mastery,  # Learning phase starts at 0.4
            'final_mastery': trajectory['final_mastery'],
            'improvement': trajectory['final_mastery'] - initial_learning_mastery,
            'num_steps': len(trajectory['steps'])
        },
        'survey_responses': survey_responses
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Saved survey results to: {output_file}")
    
    return survey_responses


def batch_simulate_surveys(
    session_ids: List[int] = None,
    model_name: str = None,
    runs: int = 1,
    db_path: str = DB_PATH,
    start_run: int = 1
):
    """
    Run survey-only simulation for multiple real learner sessions.
    
    Args:
        session_ids: List of session IDs (if None, uses all real learners)
        model_name: LLM model to use (if None, uses MODEL_TIERS)
        runs: Number of runs per session (for test-retest, use runs=3)
        db_path: Path to database
    """
    
    if session_ids is None:
        # Get all real learner sessions with 10 steps AND completed survey
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id
            FROM study_sessions s
            JOIN users u ON s.user_id = u.id
            JOIN prolific_demographics pd ON u.username = pd.prolific_participant_id
            WHERE (s.notes IS NULL OR s.notes NOT LIKE '%LLM%')
            AND pd.education_level IS NOT NULL
            AND s.id IN (
                SELECT session_id
                FROM learning_steps
                GROUP BY session_id
                HAVING COUNT(DISTINCT step_index) = 10
            )
            AND s.id IN (
                SELECT session_id
                FROM survey_responses
            )
            ORDER BY s.id
        """)
        session_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
    
    print(f"\n{'='*70}")
    print(f"BATCH SURVEY-ONLY SIMULATION")
    print(f"{'='*70}")
    print(f"Sessions: {len(session_ids)}")
    print(f"Model: {model_name or 'MODEL_TIERS (education-based)'}")
    print(f"Runs per session: {runs}")
    print(f"Total LLM calls: {len(session_ids) * runs}\n")
    
    end_run = start_run + runs - 1
    for i, session_id in enumerate(session_ids, 1):
        for run in range(start_run, end_run + 1):
            run_str = f" (run {run})" if runs > 1 or start_run > 1 else ""
            print(f"\n[{i}/{len(session_ids)}] Processing session {session_id}{run_str}")
            try:
                simulate_survey_only(session_id, model_name, run, db_path)
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()


def run_test_retest(
    model_name: str,
    profiles_per_level: int = 9,
    runs: int = 3,
    db_path: str = DB_PATH
):
    """
    Run test-retest analysis: 9 profiles per education level, 3 runs each.
    
    Args:
        model_name: LLM model to use
        profiles_per_level: Number of profiles per education level
        runs: Number of runs per profile
    """
    print(f"\n{'='*70}")
    print(f"TEST-RETEST SIMULATION")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Profiles per level: {profiles_per_level}")
    print(f"Runs per profile: {runs}")
    
    profiles = select_test_retest_profiles(db_path, profiles_per_level)
    
    total_sessions = sum(len(ids) for ids in profiles.values())
    print(f"Total profiles: {total_sessions}")
    print(f"Total LLM calls: {total_sessions * runs}\n")
    
    for edu_level, session_ids in profiles.items():
        print(f"\n--- {edu_level.upper()} ({len(session_ids)} profiles) ---")
        print(f"Session IDs: {session_ids}")
        
        for session_id in session_ids:
            for run in range(1, runs + 1):
                print(f"\n  Session {session_id}, Run {run}/{runs}")
                try:
                    simulate_survey_only(session_id, model_name, run, db_path)
                except Exception as e:
                    print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Run survey-only simulation')
    parser.add_argument('--session-id', type=int, help='Single session ID to process')
    parser.add_argument('--session-ids', type=str, help='Comma-separated list of session IDs to process')
    parser.add_argument('--batch', action='store_true', help='Process all sessions in batch mode')
    parser.add_argument('--model', type=str, help='LLM model name (e.g., llama-3-8b, gemma-2-9b-it, gpt-5-2025-08-07)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per session (for test-retest use 3)')
    parser.add_argument('--start-run', type=int, default=1, help='Run number to start from (default: 1)')
    parser.add_argument('--test-retest', action='store_true', help='Run test-retest: 9 profiles per education level')
    parser.add_argument('--profiles-per-level', type=int, default=9, help='Profiles per education level for test-retest')
    
    args = parser.parse_args()
    
    if args.test_retest:
        if not args.model:
            print("Error: --model required for test-retest mode")
            print("Example: python -m llm_user_study.survey_only_simulation --test-retest --model llama-3-8b")
            sys.exit(1)
        run_test_retest(args.model, args.profiles_per_level, args.runs or 3)
    elif args.session_ids:
        ids = [int(x.strip()) for x in args.session_ids.split(',')]
        batch_simulate_surveys(model_name=args.model, runs=args.runs, session_ids=ids, start_run=args.start_run)
    elif args.session_id:
        simulate_survey_only(args.session_id, args.model, args.runs if args.runs > 1 else None)
    else:
        batch_simulate_surveys(model_name=args.model, runs=args.runs, start_run=args.start_run)
