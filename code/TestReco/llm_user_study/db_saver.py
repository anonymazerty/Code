import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

from llm_user_study.config import DB_PATH


class SimulationDatabaseSaver:
    """Save simulation results to database matching real user study schema."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
    
    def save_simulation(self, persona, result) -> int:
        """
        Save a single simulation run to database.
        Returns session_id.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 1. Create or get LLM user
            user_id = self._get_or_create_llm_user(cursor, persona)
            
            # 2. Create study session
            session_id = self._create_session(cursor, user_id, persona, result)
            
            # 3. Save pretest attempts
            self._save_pretest_attempts(cursor, session_id, persona, result)
            
            # 4. Save learning steps and attempts
            self._save_learning_steps(cursor, session_id, result)
            
            # 5. Save orchestrator calls
            self._save_orchestrator_calls(cursor, session_id, result)

            # 6. Save survey response
            self._save_survey(cursor, session_id, result)

            # 7. Save LLM student call records (tokens + latency)
            self._save_llm_student_calls(cursor, session_id, result)

            # 8. Update session final stats
            self._update_session_final(cursor, session_id, result)
            
            conn.commit()
            print(f"✓ Saved simulation to database: session_id={session_id}")
            return session_id
            
        except Exception as e:
            conn.rollback()
            print(f"✗ Database save failed: {e}")
            raise
        finally:
            conn.close()
    
    def _get_or_create_llm_user(self, cursor, persona) -> int:
        """Get or create user for LLM simulations."""
        username = f"llm_{persona.education_level}_persona{persona.persona_id}"
        
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        # Create new LLM user (no password needed)
        cursor.execute("""
            INSERT INTO users (username, password_hash, created_at)
            VALUES (?, ?, ?)
        """, (username, "llm_simulation", datetime.now(timezone.utc)))
        
        return cursor.lastrowid
    
    def _create_session(self, cursor, user_id: int, persona, result) -> int:
        """Create study_sessions entry."""
        # Get current model from MODEL_TIERS configuration
        from llm_user_study.config import MODEL_TIERS
        current_model = MODEL_TIERS[persona.education_level][0]
        
        cursor.execute("""
            INSERT INTO study_sessions (
                user_id, topic, benchmark, orchestrator_type, model_name,
                status, started_at, ended_at,
                pretest_num_questions, pretest_correct, pretest_mastery_init,
                total_steps_planned, questions_per_step,
                notes, simulated_session_id, prompt_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            "Fundamental Mathematics",
            "math_bench",
            result.orchestrator_type if hasattr(result, 'orchestrator_type') else "tool_call",
            current_model,
            "completed",
            result.start_time if hasattr(result, 'start_time') else datetime.now(timezone.utc),
            result.end_time if hasattr(result, 'end_time') else datetime.now(timezone.utc),
            len(persona.pretest_questions),
            persona.qualification_score,
            persona.pre_qualification_score,
            10,  # total_steps_planned
            5,   # questions_per_step
            f"LLM simulation - session_id: {persona.session_id}, run: {result.run_number if hasattr(result, 'run_number') else 0}",
            persona.session_id,  # Track which real learner is being simulated
            result.prompt_type if hasattr(result, 'prompt_type') else "detailed"
        ))
        
        return cursor.lastrowid
    
    def _save_pretest_attempts(self, cursor, session_id: int, persona, result):
        """Save pretest attempts."""
        for i, (q_idx, is_correct) in enumerate(zip(persona.pretest_questions, persona.pretest_correctness)):
            cursor.execute("""
                INSERT INTO attempts (
                    session_id, phase, step_index,
                    question_index, dataset_question_id, topic,
                    original_difficulty, scaled_difficulty,
                    chosen_option_index, correct_option_index, is_correct,
                    started_at, answered_at, response_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, "pretest", None,
                q_idx, f"q_{q_idx}", "Fundamental Mathematics",
                None, None,  # difficulty info not in persona
                1 if is_correct else 0,  # chosen
                1 if is_correct else 0,  # correct
                is_correct,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
                5000  # dummy response time
            ))
    
    def _save_learning_steps(self, cursor, session_id: int, result):
        """Save learning steps and their attempts."""
        if not hasattr(result, 'learning_steps_data'):
            return
        
        for step_idx, step_data in enumerate(result.learning_steps_data):
            # Extract question indices (simulation uses 'selected_qids')
            question_indices = step_data.get('selected_qids', step_data.get('question_indices', []))
            
            # Extract rewards (simulation uses nested dict)
            rewards = step_data.get('rewards', {})
            reward_perf = float(rewards.get('performance', 0.0))
            reward_gap = float(rewards.get('gap', 0.0))
            reward_apt = float(rewards.get('aptitude', 0.0))
            
            # Save learning_steps entry
            cursor.execute("""
                INSERT INTO learning_steps (
                    session_id, step_index, action,
                    selected_question_indices_json,
                    selected_question_dataset_ids_json,
                    mastery_before, mastery_after,
                    rolling_accuracy,
                    reward_perf, reward_gap, reward_apt,
                    started_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, step_idx,
                step_data.get('action', 0),
                json.dumps(question_indices),
                json.dumps([f"q_{i}" for i in question_indices]),
                step_data.get('mastery_before', 0.0),
                step_data.get('mastery_after', 0.0),
                step_data.get('rolling_accuracy', 0.0),
                reward_perf,
                reward_gap,
                reward_apt,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            ))
            
            # Save attempts for this step
            for q_idx, is_correct in zip(
                question_indices,
                step_data.get('correctness', [])
            ):
                cursor.execute("""
                    INSERT INTO attempts (
                        session_id, phase, step_index,
                        question_index, dataset_question_id, topic,
                        original_difficulty, scaled_difficulty,
                        chosen_option_index, correct_option_index, is_correct,
                        started_at, answered_at, response_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, "learning", step_idx,
                    q_idx, f"q_{q_idx}", "Fundamental Mathematics",
                    step_data.get('difficulty', None),
                    step_data.get('scaled_difficulty', None),
                    1 if is_correct else 0,
                    1 if is_correct else 0,
                    is_correct,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                    step_data.get('response_time_ms', 5000)
                ))
    
    def _save_orchestrator_calls(self, cursor, session_id: int, result):
        """Save orchestrator call logs."""
        if not hasattr(result, 'orchestrator_calls'):
            return
        
        for call in result.orchestrator_calls:
            cursor.execute("""
                INSERT INTO orchestrator_calls (
                    session_id, step_index,
                    state_obs_json, selected_strategy, action,
                    latency_s, input_tokens, output_tokens, total_tokens,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                call.get('step_index', None),
                json.dumps(call.get('observation', {})),
                call.get('strategy', None),
                call.get('action', None),
                call.get('latency_s', None),
                call.get('input_tokens', None),
                call.get('output_tokens', None),
                call.get('total_tokens', None),
                datetime.now(timezone.utc)
            ))
    
    def _save_survey(self, cursor, session_id: int, result):
        """Save survey response."""
        if not hasattr(result, 'survey_responses'):
            return
        
        survey = result.survey_responses
        cursor.execute("""
            INSERT INTO survey_responses (
                session_id,
                accomplishment, effort_required, mental_demand,
                perceived_controllability, temporal_demand,
                frustration, trust, would_use_again, free_text,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            session_id,
            survey.get('accomplishment', None),
            survey.get('effort_required', None),
            survey.get('mental_demand', None),
            survey.get('perceived_controllability', None),
            survey.get('temporal_demand', None),
            survey.get('frustration', None),
            survey.get('trust', None),
            survey.get('would_use_again', None),
            survey.get('free_text', '')
        ))
    
    def _save_llm_student_calls(self, cursor, session_id: int, result):
        """Create table if needed and save per-call token/latency records."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_student_calls (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    INTEGER NOT NULL,
                call_type     VARCHAR(16) NOT NULL,
                step_index    INTEGER,
                question_index INTEGER,
                latency_s     FLOAT,
                input_tokens  INTEGER,
                output_tokens INTEGER,
                total_tokens  INTEGER,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        calls = getattr(result, 'llm_student_calls', None) or []
        for call in calls:
            inp = call.get('input_tokens') or 0
            out = call.get('output_tokens') or 0
            cursor.execute("""
                INSERT INTO llm_student_calls (
                    session_id, call_type, step_index, question_index,
                    latency_s, input_tokens, output_tokens, total_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                call.get('call_type'),
                call.get('step_index'),
                call.get('question_index'),
                call.get('latency_s'),
                inp,
                out,
                inp + out,
                datetime.now(timezone.utc),
            ))

    def _update_session_final(self, cursor, session_id: int, result):
        """Update session with final mastery and accuracy."""
        cursor.execute("""
            UPDATE study_sessions
            SET final_mastery = ?, final_accuracy = ?
            WHERE id = ?
        """, (
            result.final_mastery if hasattr(result, 'final_mastery') else None,
            result.final_accuracy if hasattr(result, 'final_accuracy') else None,
            session_id
        ))
