import sqlite3
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

from quizcomp_llm_study.config import DB_PATH
from quizcomp_llm_study.teacher_persona import TeacherPersona


class SimulationDatabaseSaver:
    """Saves simulation results to SQLite database."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simulations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id INTEGER NOT NULL,
                run_number INTEGER NOT NULL,
                session_id TEXT NOT NULL UNIQUE,
                
                num_iterations INTEGER,
                final_quiz_size INTEGER,
                final_match_quality REAL,
                
                duration_seconds REAL,
                timestamp TEXT,
                prompt_type TEXT,
                model_name TEXT,
                
                total_tokens INTEGER DEFAULT 0,
                total_llm_calls INTEGER DEFAULT 0,
                
                UNIQUE(persona_id, run_number, prompt_type, model_name)
            )
        """)
        
        # Composition attempts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS composition_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER NOT NULL,
                iteration INTEGER NOT NULL,
                
                action TEXT,
                mode TEXT,
                topic_distribution_json TEXT,
                difficulty_distribution_json TEXT,
                reasoning TEXT,
                
                num_mcqs INTEGER,
                target_match REAL,
                generation_time_s REAL,
                
                timestamp TEXT,
                
                FOREIGN KEY(simulation_id) REFERENCES simulations(id)
            )
        """)
        
        # Survey responses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_survey_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER NOT NULL,
                
                q1_accomplishment INTEGER,
                q2_effort INTEGER,
                q3_mental_demand INTEGER,
                q4_controllability INTEGER,
                q5_temporal_demand INTEGER,
                q6_satisfaction INTEGER,
                
                timestamp TEXT,
                
                FOREIGN KEY(simulation_id) REFERENCES simulations(id)
            )
        """)
        
        # Teacher personas table (for reference)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teacher_personas (
                persona_id INTEGER PRIMARY KEY,
                session_id INTEGER,
                username TEXT,
                prolific_pid TEXT,
                
                total_time_s REAL,
                num_compose_attempts INTEGER,
                
                real_survey_q1 INTEGER,
                real_survey_q2 INTEGER,
                real_survey_q3 INTEGER,
                real_survey_q4 INTEGER,
                real_survey_q5 INTEGER,
                real_survey_q6 INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_simulation(self, result, persona: TeacherPersona):
        """Save simulation result to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert simulation
            cursor.execute("""
                INSERT OR REPLACE INTO simulations (
                    persona_id, run_number, session_id,
                    num_iterations, final_quiz_size, final_match_quality,
                    duration_seconds, timestamp, prompt_type, model_name,
                    total_tokens, total_llm_calls
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.persona_id,
                result.run_number,
                result.session_id,
                result.num_iterations,
                result.final_quiz_size,
                result.final_match_quality,
                result.duration_seconds,
                result.timestamp,
                result.prompt_type,
                result.model_name,
                result.total_tokens,
                result.total_llm_calls
            ))
            
            simulation_id = cursor.lastrowid
            
            # Insert composition attempts
            for attempt in result.composition_attempts:
                request = attempt['request']
                quiz = attempt['generated_quiz']
                
                cursor.execute("""
                    INSERT INTO composition_attempts (
                        simulation_id, iteration,
                        action, mode,
                        topic_distribution_json, difficulty_distribution_json,
                        reasoning,
                        num_mcqs, target_match, generation_time_s,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    simulation_id,
                    attempt['iteration'],
                    request.get('action', 'compose'),
                    request.get('mode', 'fresh'),
                    json.dumps(request.get('topic_distribution', [])),
                    json.dumps(request.get('difficulty_distribution', [])),
                    request.get('reasoning', ''),
                    quiz.get('num_mcqs', 0),
                    quiz.get('target_match', 0.0),
                    quiz.get('generation_time_s', 0.0),
                    attempt.get('timestamp', datetime.now().isoformat())
                ))
            
            # Insert survey responses
            cursor.execute("""
                INSERT INTO llm_survey_responses (
                    simulation_id,
                    q1_accomplishment, q2_effort, q3_mental_demand,
                    q4_controllability, q5_temporal_demand, q6_satisfaction,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                simulation_id,
                result.survey_responses.get('accomplishment', 3),
                result.survey_responses.get('effort', 3),
                result.survey_responses.get('mental_demand', 3),
                result.survey_responses.get('controllability', 3),
                result.survey_responses.get('temporal_demand', 3),
                result.survey_responses.get('satisfaction', 3),
                result.timestamp
            ))
            
            # Insert or update persona
            cursor.execute("""
                INSERT OR REPLACE INTO teacher_personas (
                    persona_id, session_id, username, prolific_pid,
                    total_time_s, num_compose_attempts,
                    real_survey_q1, real_survey_q2, real_survey_q3,
                    real_survey_q4, real_survey_q5, real_survey_q6
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                persona.persona_id,
                persona.session_id,
                persona.username,
                persona.prolific_pid,
                persona.total_time_s,
                persona.num_compose_attempts,
                persona.survey_q1_accomplishment,
                persona.survey_q2_effort,
                persona.survey_q3_mental_demand,
                persona.survey_q4_controllability,
                persona.survey_q5_temporal_demand,
                persona.survey_q6_satisfaction
            ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        
        finally:
            conn.close()
    
    def save_interactive_simulation(self, result, persona: TeacherPersona = None):
        """
        Save interactive simulation result to database.
        
        Interactive simulations have actual API interactions, not simulated ones.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert simulation
            cursor.execute("""
                INSERT OR REPLACE INTO simulations (
                    persona_id, run_number, session_id,
                    num_iterations, final_quiz_size, final_match_quality,
                    duration_seconds, timestamp, prompt_type, model_name,
                    total_tokens, total_llm_calls
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.persona_id,
                result.run_number,
                result.session_id,
                result.num_iterations,
                result.final_quiz_size,
                result.final_match_quality,
                result.duration_seconds,
                result.timestamp,
                "interactive",  # prompt_type for interactive simulations
                result.model_name,
                result.total_tokens,
                result.total_llm_calls
            ))
            
            simulation_id = cursor.lastrowid
            
            # Insert composition attempts 
            for attempt in result.composition_attempts:
                cursor.execute("""
                    INSERT INTO composition_attempts (
                        simulation_id, iteration,
                        action, mode,
                        topic_distribution_json, difficulty_distribution_json,
                        reasoning,
                        num_mcqs, target_match, generation_time_s,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    simulation_id,
                    attempt['iteration'],
                    'compose',  # All API calls are compose actions
                    attempt['mode'],
                    json.dumps(attempt['topic_distribution']),
                    json.dumps(attempt['difficulty_distribution']),
                    '',  # Reasoning is in agent decisions, not API calls
                    attempt.get('quiz_data', {}).get('num_mcqs', result.initial_num_mcqs),
                    attempt['target_match'],
                    attempt['api_time_s'],
                    datetime.now().isoformat()
                ))
            
            # Insert survey responses
            cursor.execute("""
                INSERT INTO llm_survey_responses (
                    simulation_id,
                    q1_accomplishment, q2_effort, q3_mental_demand,
                    q4_controllability, q5_temporal_demand, q6_satisfaction,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                simulation_id,
                result.survey_responses.get('accomplishment', 3),
                result.survey_responses.get('effort', 3),
                result.survey_responses.get('mental_demand', 3),
                result.survey_responses.get('controllability', 3),
                result.survey_responses.get('temporal_demand', 3),
                result.survey_responses.get('satisfaction', 3),
                result.timestamp
            ))
            
            # Insert or update persona if provided
            if persona:
                cursor.execute("""
                    INSERT OR REPLACE INTO teacher_personas (
                        persona_id, session_id, username, prolific_pid,
                        total_time_s, num_compose_attempts,
                        real_survey_q1, real_survey_q2, real_survey_q3,
                        real_survey_q4, real_survey_q5, real_survey_q6
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    persona.persona_id,
                    persona.session_id,
                    persona.username,
                    persona.prolific_pid,
                    persona.total_time_s,
                    persona.num_compose_attempts,
                    persona.survey_q1_accomplishment,
                    persona.survey_q2_effort,
                    persona.survey_q3_mental_demand,
                    persona.survey_q4_controllability,
                    persona.survey_q5_temporal_demand,
                    persona.survey_q6_satisfaction
                ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        
        finally:
            conn.close()
    
    def get_completed_simulations(self, prompt_type: str, model_name: str) -> set:
        """Get set of (persona_id, run_number) tuples for completed simulations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT persona_id, run_number
            FROM simulations
            WHERE prompt_type = ? AND model_name = ?
        """, (prompt_type, model_name))
        
        completed = {(row[0], row[1]) for row in cursor.fetchall()}
        conn.close()
        
        return completed
