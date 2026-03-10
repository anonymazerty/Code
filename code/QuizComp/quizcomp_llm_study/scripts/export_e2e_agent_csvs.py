import sqlite3
import csv
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent  # quizcomp_llm_study/
OUTPUT_ROOT = BASE_DIR / "Results" / "e2eAgent" / "Results_as_csvs"

# Mapping the results to their folders
PROMPT_TYPES = {
    "Calibrated": ("Calibrated results", "detailed"),
    "Role-Anchored": ("Role-Anchored results", "inbetween"),
    "Task-Constrained": ("Task-Constrained results", "general"),
}

#Mapping the LLM names 
LLMS = {
    "GPT5": "gpt",
    "llama8B": "llama",
    "Mistral": "mistral",
}

QUESTION_COLS = [
    "Q1_Accomplishment",
    "Q2_Effort",
    "Q3_Mental_Demand",
    "Q4_Controllability",
    "Q5_Temporal_Demand",
    "Q6_Satisfaction",
]


# CROWD export
def export_crowd_csv(crowd_db: Path, out_dir: Path):
    """Export CROWD survey responses: one row per completed participant."""
    conn = sqlite3.connect(crowd_db)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            s.id AS session_id,
            s.prolific_pid,
            sr.q1_accomplishment,
            sr.q2_effort,
            sr.q3_mental_demand,
            sr.q4_controllability,
            sr.q5_temporal_demand,
            sr.q6_satisfaction_trust
        FROM survey_responses sr
        JOIN study_sessions s ON sr.session_id = s.id
        WHERE s.status = 'completed'
        ORDER BY s.id
    """)
    rows = cursor.fetchall()
    conn.close()

    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "crowd_survey.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "prolific_pid"] + QUESTION_COLS)
        writer.writerows(rows)
    print(f"  Wrote {len(rows):>3} rows → {out_path}")


# LLM export 
def export_llm_csvs(llm_key: str, db_path: Path, out_dir: Path):
    """Export per-run, average, and median CSVs for one LLM."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            s.persona_id,
            s.run_number,
            sr.q1_accomplishment,
            sr.q2_effort,
            sr.q3_mental_demand,
            sr.q4_controllability,
            sr.q5_temporal_demand,
            sr.q6_satisfaction
        FROM llm_survey_responses sr
        JOIN simulations s ON sr.simulation_id = s.id
        ORDER BY s.persona_id, s.run_number
    """)
    rows = cursor.fetchall()
    conn.close()

    llm_dir = out_dir / llm_key
    os.makedirs(llm_dir, exist_ok=True)

    # Group by run and by persona 
    by_run = defaultdict(list)       # run_number → [(persona_id, q1…q6)]
    by_persona = defaultdict(dict)   # persona_id → {run_number: [q1…q6]}

    for persona_id, run_number, *answers in rows:
        by_run[run_number].append((persona_id, *answers))
        by_persona[persona_id][run_number] = answers

    # Per-run CSVs 
    sorted_runs = sorted(by_run)
    for idx, run_number in enumerate(sorted_runs):
        label = idx + 1  # normalise to 1-based filename
        out_path = llm_dir / f"run_{label}_survey.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["persona_id"] + QUESTION_COLS)
            for row in sorted(by_run[run_number]):
                writer.writerow(row)
        print(f"  Wrote {len(by_run[run_number]):>3} rows → {out_path.name}")

    # Average & median CSVs 
    avg_rows, med_rows = [], []

    for persona_id in sorted(by_persona):
        runs_data = by_persona[persona_id]
        if not runs_data:
            continue
        matrix = np.array(list(runs_data.values()), dtype=float)
        avg_vals = np.mean(matrix, axis=0)
        med_vals = np.median(matrix, axis=0)
        avg_rows.append([persona_id] + [round(v, 2) for v in avg_vals])
        med_rows.append([persona_id] + [round(v, 2) for v in med_vals])

    avg_path = llm_dir / "average_survey.csv"
    with open(avg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona_id"] + QUESTION_COLS)
        writer.writerows(avg_rows)
    print(f"  Wrote {len(avg_rows):>3} rows → {avg_path.name}")

    med_path = llm_dir / "median_survey.csv"
    with open(med_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona_id"] + QUESTION_COLS)
        writer.writerows(med_rows)
    print(f"  Wrote {len(med_rows):>3} rows → {med_path.name}")


# main 
def main():
    for prompt_label, (db_folder, db_suffix) in PROMPT_TYPES.items():
        db_dir = BASE_DIR / db_folder
        out_dir = OUTPUT_ROOT / prompt_label

        print(f"\n{'='*60}")
        print(f"  {prompt_label}  (source: {db_folder}/)")
        print(f"{'='*60}")

        # CROWD
        crowd_db = db_dir / "quizcomp_study.sqlite"
        print(f"\n[CROWD]")
        export_crowd_csv(crowd_db, out_dir)

        # LLMs
        for llm_key, llm_frag in LLMS.items():
            db_path = db_dir / f"llm_results_{llm_frag}_{db_suffix}.db"
            print(f"\n[{llm_key}]  ← {db_path.name}")
            export_llm_csvs(llm_key, db_path, out_dir)

    print(f"\nDone — all CSVs written under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
