import json
import sqlite3
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from llm_user_study.config import DB_PATH, MODEL_TIERS, get_model_for_persona, TOPIC
from llm_user_study.prompt import (
    # PROMPT_TEMPLATE_DETAILED, PROMPT_TEMPLATE_GENERAL,
    PROMPT_TEMPLATE_SURVEY, EDUCATION_EXPLANATIONS, simple_system_prompt, PROMPT_TEMPLATE_GENERAL, PROMPT_TEMPLATE_DETAILED
)
from real_user_study.loader import load_topic_questions_with_difficulties


@dataclass
class LLMPersona:
    """LLM-based learner persona created directly from database."""
    persona_id: int
    session_id: int
    education_level: str
    model_name: str
    
    # From database
    qualification_score: int
    pre_qualification_score: float
    final_mastery: float
    pretest_questions: List[int]
    pretest_correctness: List[bool]
    pretest_chosen_options: List[int]
    
    # Questions dict (not serialized)
    questions_dict: Dict[int, Dict] = None
    
    def to_dict(self) -> Dict:
        # Don't include questions_dict in serialization
        data = asdict(self)
        data.pop('questions_dict', None)
        return data
    
    def set_questions_dict(self, questions_dict: Dict[int, Dict]):
        """Set the questions dictionary after loading from JSON."""
        self.questions_dict = questions_dict
    
    def get_system_prompt(self, prompt_type: str = "", learning_trajectory: str = "") -> str:
        """Generate system prompt."""
        if prompt_type == "detailed":
            template = PROMPT_TEMPLATE_DETAILED
        if prompt_type == "general":
            template = PROMPT_TEMPLATE_GENERAL
        elif prompt_type == "survey":
            template = PROMPT_TEMPLATE_SURVEY
        elif prompt_type == "simple":
            template = simple_system_prompt
        else:
            template = simple_system_prompt
        
        education_explanations = EDUCATION_EXPLANATIONS.get(
            self.education_level,
            EDUCATION_EXPLANATIONS["undergraduate"]
        )
        
        # Build pretest questions display
        pretest_display = self._build_pretest_display()
        
        return template.format(
            education_level=self.education_level.replace('_', ' ').title(),
            education_explanations=education_explanations,
            initial_mastery=self.pre_qualification_score,
            qualification_score=self.qualification_score,
            pretest_questions=pretest_display,
            learning_trajectory=learning_trajectory
        )
    
    def _build_pretest_display(self) -> str:
        """Build a formatted display of pretest questions and results."""
        if not self.questions_dict:
            return "(Pretest questions not available)"
        
        # Separate into correct and incorrect
        correct_questions = []
        incorrect_questions = []
        
        for q_id, is_correct, chosen_opt in zip(
            self.pretest_questions, 
            self.pretest_correctness,
            self.pretest_chosen_options
        ):
            if q_id in self.questions_dict:
                q = self.questions_dict[q_id]
                item = (q_id, q, chosen_opt)
                if is_correct:
                    correct_questions.append(item)
                else:
                    incorrect_questions.append(item)
        
        result = ["\nDuring the prequalification test, here are the actual questions you encountered:\n"]
        
        # Format correct questions
        result.append("\nQuestions you answered CORRECTLY:")
        if correct_questions:
            result.append(self._format_questions(correct_questions))
        else:
            result.append("  (None)")
        
        # Format incorrect questions
        result.append("\nQuestions you answered INCORRECTLY:")
        if incorrect_questions:
            result.append(self._format_questions(incorrect_questions))
        else:
            result.append("  (None)")
        
        return "\n".join(result)
    
    def _format_questions(self, question_items: List[tuple]) -> str:
        """Format a list of (question_id, question_dict, chosen_option) tuples."""
        lines = []
        for q_id, q, chosen_opt in question_items:
            q_text = q.get('text', q.get('question', 'Question text not available'))
            options = q.get('options', [])
            correct_opt = q.get('answer', 0)
            
            # Question text
            lines.append(f"\n  Question {q_id}: {q_text}")
            lines.append("  Options:")
            
            # Display all options with markers
            for i, opt_text in enumerate(options):
                markers = []
                if i == chosen_opt:
                    markers.append("← YOUR CHOICE")
                if i == correct_opt:
                    markers.append("✓ CORRECT")
                
                marker_str = " " + " ".join(markers) if markers else ""
                lines.append(f"    {i}. {opt_text}{marker_str}")
        
        return "\n".join(lines)


class PersonaBuilder:
    """Build LLM personas directly from database."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.questions, self.difficulties = load_topic_questions_with_difficulties(TOPIC)
        print(f"Loaded {len(self.questions)} questions for {TOPIC}")
    
    def get_completed_sessions(self) -> List[int]:
        """Get session IDs that completed 10 learning steps + survey."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id
            FROM study_sessions s
            INNER JOIN survey_responses sr ON s.id = sr.session_id
            AND (s.notes IS NULL OR s.notes NOT LIKE '%LLM%')
            AND s.id IN (
                SELECT session_id 
                FROM learning_steps 
                GROUP BY session_id 
                HAVING COUNT(DISTINCT step_index) = 10
            )
            ORDER BY s.id
        """)
        session_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        print(f"Found {len(session_ids)} sessions with 10 steps + survey")
        return session_ids
    
    def build_persona(self, session_id: int, persona_id: int) -> Optional[LLMPersona]:
        """Build persona from a single session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get session data
        cursor.execute("""
            SELECT s.*, u.username, p.education_level
            FROM study_sessions s
            JOIN users u ON s.user_id = u.id
            LEFT JOIN prolific_demographics p ON u.username = p.prolific_participant_id
            WHERE s.id = ?
        """, (session_id,))
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            return None
        
        # Get pretest attempts
        cursor.execute("""
            SELECT question_index, is_correct, chosen_option_index
            FROM attempts
            WHERE session_id = ? AND phase = 'pretest'
            ORDER BY id
        """, (session_id,))
        pretest_attempts = cursor.fetchall()
        
        conn.close()
        
        # Extract data
        pretest_questions = [a['question_index'] for a in pretest_attempts]
        pretest_correctness = [bool(a['is_correct']) for a in pretest_attempts]
        pretest_chosen_options = [a['chosen_option_index'] for a in pretest_attempts]
        qualification_score = sum(pretest_correctness)
        
        # Get education level
        education_level = session['education_level']
        if not education_level:
            return None
        education_level = education_level.lower().replace(' ', '_')
        
        # Select model
        model_name = get_model_for_persona(education_level, persona_id)
        
        # Build questions dict for this persona
        questions_dict = {i: q for i, q in enumerate(self.questions)}
        
        return LLMPersona(
            persona_id=persona_id,
            session_id=session_id,
            education_level=education_level,
            model_name=model_name,
            qualification_score=qualification_score,
            pre_qualification_score=0.4,  # Default starting mastery for LLM learning
            final_mastery=session['final_mastery'] or 0,
            pretest_questions=pretest_questions,
            pretest_correctness=pretest_correctness,
            pretest_chosen_options=pretest_chosen_options,
            questions_dict=questions_dict
        )
    
    def build_all_personas(self, max_personas: int = None) -> List[LLMPersona]:
        """Build personas from all completed sessions."""
        session_ids = self.get_completed_sessions()
        if max_personas:
            session_ids = session_ids[:max_personas]
        
        personas = []
        for idx, session_id in enumerate(session_ids):
            persona = self.build_persona(session_id, persona_id=idx)
            if persona:
                personas.append(persona)
        
        print(f"Built {len(personas)} personas from {len(session_ids)} sessions")
        return personas
    
    def save_personas(self, personas: List[LLMPersona], output_path: str):
        """Save personas to JSON."""
        data = {
            'total_personas': len(personas),
            'education_distribution': self._count_by_education(personas),
            'personas': [p.to_dict() for p in personas]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(personas)} personas to {output_path}")
    
    def _count_by_education(self, personas: List[LLMPersona]) -> Dict[str, int]:
        """Count personas by education level."""
        counts = {}
        for p in personas:
            counts[p.education_level] = counts.get(p.education_level, 0) + 1
        return counts


if __name__ == "__main__":
    builder = PersonaBuilder()
    personas = builder.build_all_personas(max_personas=None)  # Get all valid personas
    builder.save_personas(personas, "llm_user_study/personas.json")
