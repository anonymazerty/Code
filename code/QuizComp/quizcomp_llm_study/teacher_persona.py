from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class TeacherPersona:
    persona_id: int
    session_id: int
    username: str
    prolific_pid: str
    
    # Session metadata
    total_time_s: float
    num_compose_attempts: int
    
    # Compose history 
    compose_attempts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Survey responses (Q1-Q6)
    survey_q1_accomplishment: int = 0
    survey_q2_effort: int = 0
    survey_q3_mental_demand: int = 0
    survey_q4_controllability: int = 0
    survey_q5_temporal_demand: int = 0
    survey_q6_satisfaction: int = 0
    
    def get_first_attempt_topics(self) -> Optional[List[float]]:
        """Get topic distribution from first compose attempt."""
        if self.compose_attempts:
            return self.compose_attempts[0].get('teacher_topic_json')
        return None
    
    def get_first_attempt_difficulty(self) -> Optional[List[float]]:
        """Get difficulty distribution from first compose attempt."""
        if self.compose_attempts:
            return self.compose_attempts[0].get('teacher_level_json')
        return None
    
    def get_composition_trajectory_summary(self) -> str:
        lines = []
        lines.append(f"QUIZ COMPOSITION SESSION SUMMARY")
        lines.append(f"Total time: {self.total_time_s:.1f} seconds ({self.total_time_s/60:.1f} minutes)")
        lines.append(f"Number of composition attempts: {self.num_compose_attempts}")
        lines.append("")
        
        for i, attempt in enumerate(self.compose_attempts, 1):
            lines.append(f"--- Attempt {i} ({attempt['mode']}) ---")
            
            if attempt['teacher_topic_json']:
                lines.append(f"Topic targets: {attempt['teacher_topic_json']}")
            if attempt['teacher_level_json']:
                lines.append(f"Difficulty targets: {attempt['teacher_level_json']}")
            
            if attempt['num_mcqs']:
                lines.append(f"Quiz size: {attempt['num_mcqs']} questions")
            
            if attempt['target_match'] is not None:
                lines.append(f"Match quality: {attempt['target_match']:.2%}")
            
            if attempt['api_time_s']:
                lines.append(f"Generation time: {attempt['api_time_s']:.2f}s")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary."""
        return {
            'persona_id': self.persona_id,
            'session_id': self.session_id,
            'username': self.username,
            'prolific_pid': self.prolific_pid,
            'total_time_s': self.total_time_s,
            'num_compose_attempts': self.num_compose_attempts,
            'compose_attempts': self.compose_attempts,
            'survey_q1_accomplishment': self.survey_q1_accomplishment,
            'survey_q2_effort': self.survey_q2_effort,
            'survey_q3_mental_demand': self.survey_q3_mental_demand,
            'survey_q4_controllability': self.survey_q4_controllability,
            'survey_q5_temporal_demand': self.survey_q5_temporal_demand,
            'survey_q6_satisfaction': self.survey_q6_satisfaction,
        }


class PersonaBuilder:
    """Builds teacher personas from user study data."""
    
    @staticmethod
    def from_json_file(filepath: str) -> List[TeacherPersona]:
        """Load personas from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        personas = []
        for p in data['personas']:
            persona = TeacherPersona(**p)
            personas.append(persona)
        
        return personas
    
    @staticmethod
    def to_json_file(personas: List[TeacherPersona], filepath: str):
        """Save personas to JSON file."""
        data = {
            'metadata': {
                'num_personas': len(personas),
                'description': 'Teacher personas for QuizComp LLM simulation'
            },
            'personas': [p.to_dict() for p in personas]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_teacher_personas(filepath: str = None) -> List[TeacherPersona]:
    if filepath is None:
        from pathlib import Path
        filepath = Path(__file__).parent / "teacher_personas.json"
    
    return PersonaBuilder.from_json_file(filepath)
