import sqlite3
import csv
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent  # quizcomp_llm_study/
DB_DIR = BASE_DIR / "AgentJUDGEresults"
OUTPUT_DIR = BASE_DIR / "Results" / "JudgeAgent" / "Results_as_csvs"

CROWD_DB = DB_DIR / "quizcomp_study.sqlite"

LLM_DBS = {
    "GPT5": DB_DIR / "llm_results_GPT5.db",
    "llama8B": DB_DIR / "llm_results_llama.db",
    "Mistral": DB_DIR / "llm_results_Mistral.db",
}

QUESTION_COLS = [
    "Q1_Accomplishment",
    "Q2_Effort",
    "Q3_Mental_Demand",
    "Q4_Controllability",
    "Q5_Temporal_Demand",
    "Q6_Satisfaction",
]


def export_crowd_csv():
    """Export CROWD survey responses: one row per participant."""
    conn = sqlite3.connect(CROWD_DB)
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

    out_path = OUTPUT_DIR / "crowd_survey.csv"
    os.makedirs(out_path.parent, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "prolific_pid"] + QUESTION_COLS)
        for row in rows:
            writer.writerow(row)

    print(f"  Wrote {len(rows)} rows → {out_path}")


def export_llm_csvs(llm_name, db_path):
    """Export per-run CSVs, average CSV, and median CSV for one LLM."""
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

    llm_dir = OUTPUT_DIR / llm_name
    os.makedirs(llm_dir, exist_ok=True)

    # Group by run_number and by persona 
    by_run = defaultdict(list)        # run_number → list of (persona_id, q1..q6)
    by_persona = defaultdict(dict)    # persona_id → {run_number: [q1..q6]}

    for persona_id, run_number, *answers in rows:
        by_run[run_number].append((persona_id, *answers))
        by_persona[persona_id][run_number] = answers

    # Per-run CSVs
    for run_number in sorted(by_run):
        out_path = llm_dir / f"run_{run_number}_survey.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["persona_id"] + QUESTION_COLS)
            for row in sorted(by_run[run_number]):
                writer.writerow(row)
        print(f"  Wrote {len(by_run[run_number]):>3} rows → {out_path.name}")

    # Average & median CSVs
    avg_rows = []
    med_rows = []

    for persona_id in sorted(by_persona):
        runs_data = by_persona[persona_id]  # {run_number: [q1..q6]}
        if not runs_data:
            continue

        # Stack answers across runs: shape (n_runs, 6)
        matrix = np.array(list(runs_data.values()), dtype=float)

        avg_vals = np.mean(matrix, axis=0)
        med_vals = np.median(matrix, axis=0)

        avg_rows.append([persona_id] + [round(v, 2) for v in avg_vals])
        med_rows.append([persona_id] + [round(v, 2) for v in med_vals])

    # average CSV
    avg_path = llm_dir / "average_survey.csv"
    with open(avg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona_id"] + QUESTION_COLS)
        writer.writerows(avg_rows)
    print(f"  Wrote {len(avg_rows):>3} rows → {avg_path.name}")

    # median CSV
    med_path = llm_dir / "median_survey.csv"
    with open(med_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona_id"] + QUESTION_COLS)
        writer.writerows(med_rows)
    print(f"  Wrote {len(med_rows):>3} rows → {med_path.name}")


def main():
    print("="*80)
    print("Exporting JudgeAgent results to CSVs")
    print("="*80)

    print("\n[CROWD]")
    export_crowd_csv()

    for llm_name, db_path in LLM_DBS.items():
        print(f"\n[{llm_name}]")
        if not db_path.exists():
            print(f"  WARNING: {db_path} not found, skipping.")
            continue
        export_llm_csvs(llm_name, db_path)

    print("\n" + "="*80)
    print("Done.")
    print("="*80)


if __name__ == "__main__":
    main()
