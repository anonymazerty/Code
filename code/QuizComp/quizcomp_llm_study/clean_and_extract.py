import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Database path 
DB_PATH = Path("..") / "quizcomp_study.sqlite"
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)

def clean_and_extract_profiles():
    """Clean database and extract teacher profiles."""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get completed sessions with survey responses
    query = """
    SELECT 
        s.id as session_id,
        s.user_id,
        u.username,
        s.prolific_pid,
        s.status,
        s.started_at,
        s.ended_at,
        s.total_time_s,
        COUNT(DISTINCT c.id) as num_compose_attempts,
        sr.q1_accomplishment,
        sr.q2_effort,
        sr.q3_mental_demand,
        sr.q4_controllability,
        sr.q5_temporal_demand,
        sr.q6_satisfaction_trust,
        sr.q7_would_use_again,
        sr.comments
    FROM study_sessions s
    JOIN users u ON s.user_id = u.id
    JOIN compose_attempts c ON s.id = c.session_id
    JOIN survey_responses sr ON s.id = sr.session_id
    WHERE s.status = 'completed'
    GROUP BY s.id
    HAVING num_compose_attempts > 0
    ORDER BY s.id
    """
    
    cursor.execute(query)
    sessions = cursor.fetchall()
    
    print(f"Found {len(sessions)} valid completed sessions")
    
    # Extract personas
    personas = []
    for idx, session in enumerate(sessions):
        session_id = session['session_id']
        
        # Get compose attempts for this session
        cursor.execute("""
            SELECT 
                mode,
                data_uuid,
                teacher_topic_json,
                teacher_level_json,
                num_mcqs,
                mcq_ids_json,
                topic_coverage_json,
                difficulty_coverage_json,
                target_match,
                api_time_s,
                ts
            FROM compose_attempts
            WHERE session_id = ?
            ORDER BY id
        """, (session_id,))
        
        compose_attempts = [dict(row) for row in cursor.fetchall()]
        
        # Parse JSON fields
        for attempt in compose_attempts:
            for json_field in ['teacher_topic_json', 'teacher_level_json', 
                              'mcq_ids_json', 'topic_coverage_json', 
                              'difficulty_coverage_json']:
                if attempt[json_field]:
                    attempt[json_field] = json.loads(attempt[json_field])
        
        # Create persona
        persona = {
            'persona_id': idx,
            'session_id': session_id,
            'username': session['username'],
            'prolific_pid': session['prolific_pid'],
            
            # Session metadata
            'total_time_s': session['total_time_s'],
            'num_compose_attempts': session['num_compose_attempts'],
            
            # Compose history
            'compose_attempts': compose_attempts,
            
            # Survey responses (Q1-Q6 only, no Q7 or comments)
            'survey_q1_accomplishment': session['q1_accomplishment'],
            'survey_q2_effort': session['q2_effort'],
            'survey_q3_mental_demand': session['q3_mental_demand'],
            'survey_q4_controllability': session['q4_controllability'],
            'survey_q5_temporal_demand': session['q5_temporal_demand'],
            'survey_q6_satisfaction': session['q6_satisfaction_trust'],
        }
        
        personas.append(persona)
        
        print(f"  Persona {idx}: session {session_id}, {session['num_compose_attempts']} attempts, "
              f"username={session['username']}")
    
    # Save personas to JSON
    output_file = OUTPUT_DIR / "teacher_personas.json"
    output_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'source_db': str(DB_PATH),
            'num_personas': len(personas),
            'description': 'Teacher personas extracted from QuizComp user study'
        },
        'personas': personas
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n Saved {len(personas)} teacher personas to {output_file}")
    
    conn.close()
    
    return personas


def clean_database_tables():
    import shutil
    
    # Create backup
    backup_path = DB_PATH.with_suffix('.backup.sqlite')
    print(f"Creating backup: {backup_path}")
    shutil.copy2(DB_PATH, backup_path)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get IDs of valid sessions
    cursor.execute("""
        SELECT s.id
        FROM study_sessions s
        JOIN users u ON s.user_id = u.id
        JOIN survey_responses sr ON s.id = sr.session_id
        WHERE s.status = 'completed'
        AND u.username NOT IN ('test_pid_001')
        AND u.username NOT LIKE '%riamoro%'
        AND EXISTS (SELECT 1 FROM compose_attempts c WHERE c.session_id = s.id)
    """)
    
    valid_session_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"\nCleaning database tables...")
    print(f"Keeping {len(valid_session_ids)} valid sessions: {valid_session_ids}")
    
    # Delete invalid sessions
    if valid_session_ids:
        placeholders = ','.join('?' * len(valid_session_ids))
        
        # Delete related data first
        cursor.execute(f"DELETE FROM event_logs WHERE session_id NOT IN ({placeholders})", 
                      valid_session_ids)
        print(f"  Deleted {cursor.rowcount} invalid event_logs")
        
        cursor.execute(f"DELETE FROM compose_attempts WHERE session_id NOT IN ({placeholders})", 
                      valid_session_ids)
        print(f"  Deleted {cursor.rowcount} invalid compose_attempts")
        
        cursor.execute(f"DELETE FROM survey_responses WHERE session_id NOT IN ({placeholders})", 
                      valid_session_ids)
        print(f"  Deleted {cursor.rowcount} invalid survey_responses")
        
        cursor.execute(f"DELETE FROM study_sessions WHERE id NOT IN ({placeholders})", 
                      valid_session_ids)
        print(f"  Deleted {cursor.rowcount} invalid study_sessions")
        
        # Delete users with no sessions
        cursor.execute("DELETE FROM users WHERE id NOT IN (SELECT DISTINCT user_id FROM study_sessions)")
        print(f"  Deleted {cursor.rowcount} invalid users")
    
    conn.commit()
    conn.close()
    
    print(f"\n Database cleaned successfully")
    print(f" Backup saved at: {backup_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean database and extract teacher profiles")
    parser.add_argument("--clean-db", action="store_true", 
                       help="Clean database tables (creates backup first)")
    parser.add_argument("--extract", action="store_true", 
                       help="Extract teacher personas to JSON")
    
    args = parser.parse_args()
    
    if not (args.clean_db or args.extract):
        # Default: do both
        args.clean_db = True
        args.extract = True
    
    if args.clean_db:
        clean_database_tables()
    
    if args.extract:
        personas = clean_and_extract_profiles()
        print(f"\n Extraction complete. Found {len(personas)} teacher personas.")
