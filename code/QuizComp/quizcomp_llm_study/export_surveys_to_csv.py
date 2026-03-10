import sqlite3
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_llm_surveys():
    """Export LLM survey responses from llm_results.db to CSV."""
    
    llm_db_path = Path(__file__).parent / "llm_results.db"
    
    if not llm_db_path.exists():
        print(f"ERROR: {llm_db_path} does not exist!")
        print("Run survey_only_simulation.py first to generate LLM responses.")
        return None
    
    print(f"Exporting LLM survey responses from {llm_db_path}...")
    
    conn = sqlite3.connect(llm_db_path)
    
    # Query LLM survey responses 
    query = """
    SELECT 
        s.persona_id,
        s.session_id,
        s.model_name,
        s.prompt_type,
        sr.q1_accomplishment as accomplishment,
        sr.q2_effort as effort,
        sr.q3_mental_demand as mental_demand,
        sr.q4_controllability as controllability,
        sr.q5_temporal_demand as temporal_demand,
        sr.q6_satisfaction as satisfaction,
        sr.timestamp
    FROM llm_survey_responses sr
    JOIN simulations s ON sr.simulation_id = s.id
    ORDER BY s.persona_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Save to CSV
    output_path = Path(__file__).parent / "llm_survey_responses.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Exported {len(df)} LLM survey responses to: {output_path}")
    return df


def export_real_surveys():
    """Export real teacher survey responses from quizcomp_study.sqlite to CSV."""
    
    real_db_path = Path(__file__).parent.parent / "quizcomp_study.sqlite"
    
    if not real_db_path.exists():
        print(f"ERROR: {real_db_path} does not exist!")
        return None
    
    print(f"Exporting real teacher survey responses from {real_db_path}...")
    
    conn = sqlite3.connect(real_db_path)
    
    
    query = """
    SELECT 
        s.id as session_id,
        u.username,
        s.prolific_pid,
        sr.q1_accomplishment as accomplishment,
        sr.q2_effort as effort,
        sr.q3_mental_demand as mental_demand,
        sr.q4_controllability as controllability,
        sr.q5_temporal_demand as temporal_demand,
        sr.q6_satisfaction_trust as satisfaction,
        sr.ts as timestamp
    FROM survey_responses sr
    JOIN study_sessions s ON sr.session_id = s.id
    JOIN users u ON s.user_id = u.id
    WHERE s.prolific_pid IS NOT NULL
    ORDER BY s.id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Save to CSV
    output_path = Path(__file__).parent / "real_survey_responses.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Exported {len(df)} real teacher survey responses to: {output_path}")
    return df


def create_comparison_csv(llm_df, real_df):
    """Create a side-by-side comparison CSV."""
    
    if llm_df is None or real_df is None:
        print("Cannot create comparison - missing data")
        return
    
    print("\nCreating comparison CSV...")
    
    # Merge on session_id
    # First, get session_id from persona mapping
    personas_path = Path(__file__).parent / "teacher_personas.json"
    if personas_path.exists():
        import json
        with open(personas_path) as f:
            personas_data = json.load(f)
        
        # Create mapping: persona_id -> session_id
        persona_to_session = {
            p['persona_id']: p['session_id'] 
            for p in personas_data['personas']
        }
        
        # Add session_id to LLM df
        llm_df['real_session_id'] = llm_df['persona_id'].map(persona_to_session)
    
    # Merge LLM and real responses
    comparison = pd.merge(
        llm_df,
        real_df,
        left_on='real_session_id',
        right_on='session_id',
        how='inner',
        suffixes=('_llm', '_real')
    )
    
    
    comparison_cols = {
        'persona_id': 'persona_id',
        'real_session_id': 'session_id',
        'prolific_pid': 'prolific_pid',
        'model_name': 'llm_model',
        'accomplishment_llm': 'accomplishment_llm',
        'accomplishment_real': 'accomplishment_real',
        'effort_llm': 'effort_llm',
        'effort_real': 'effort_real',
        'mental_demand_llm': 'mental_demand_llm',
        'mental_demand_real': 'mental_demand_real',
        'controllability_llm': 'controllability_llm',
        'controllability_real': 'controllability_real',
        'temporal_demand_llm': 'temporal_demand_llm',
        'temporal_demand_real': 'temporal_demand_real',
        'satisfaction_llm': 'satisfaction_llm',
        'satisfaction_real': 'satisfaction_real',
    }
    
    comparison = comparison[list(comparison_cols.keys())].rename(columns=comparison_cols)
    
    # Calculate differences
    questions = ['accomplishment', 'effort', 'mental_demand', 'controllability', 'temporal_demand', 'satisfaction']
    for q in questions:
        comparison[f'{q}_diff'] = comparison[f'{q}_llm'] - comparison[f'{q}_real']
        comparison[f'{q}_abs_diff'] = abs(comparison[f'{q}_diff'])
    
    # Save comparison
    output_path = Path(__file__).parent / "survey_comparison.csv"
    comparison.to_csv(output_path, index=False)
    
    print(f" Created comparison CSV: {output_path}")
    print(f"\nComparison Summary:")
    print(f"  Matched responses: {len(comparison)}")
    
    # Print average differences
    for q in questions:
        mae = comparison[f'{q}_abs_diff'].mean()
        print(f"  {q}: MAE = {mae:.2f}")
    
    overall_mae = comparison[[f'{q}_abs_diff' for q in questions]].mean().mean()
    print(f"\n  Overall MAE: {overall_mae:.2f}")


if __name__ == "__main__":
    print("="*70)
    print("EXPORTING SURVEY RESPONSES TO CSV")
    print("="*70)
    print()
    
    # Export LLM surveys
    llm_df = export_llm_surveys()
    print()
    
    # Export real surveys
    real_df = export_real_surveys()
    print()
    
    # Create comparison
    if llm_df is not None and real_df is not None:
        create_comparison_csv(llm_df, real_df)
    
    print()
    print("="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("   llm_survey_responses.csv")
    print("   real_survey_responses.csv")
    print("   survey_comparison.csv ")
