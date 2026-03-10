import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.environ.get("REALUSER_DB_PATH", "./real-user-study/TestReco/results.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../../llm_user_study/results/JudgeAgent/results_as_csv")
OUT_DIR = os.path.normpath(OUT_DIR)

SURVEY_COLS = [
    "accomplishment", "effort_required", "mental_demand",
    "perceived_controllability", "temporal_demand", "trust"
]

MODEL_LABELS = {
    "llama-3-8b":           "llama",
    "mistral-24b-instruct": "mistral",
    "gpt-5-2025-08-07":     "gpt",
}

RUNS_TO_EXPORT = [1, 2, 3, 4, 5]


def get_conn():
    return sqlite3.connect(DB_PATH)


def export_crowd():
    """Export real learners who completed 10 steps and submitted the survey."""
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            ss.id AS session_id,
            pd.education_level,
            sr.accomplishment,
            sr.effort_required,
            sr.mental_demand,
            sr.perceived_controllability,
            sr.temporal_demand,
            sr.trust
        FROM study_sessions ss
        JOIN survey_responses sr ON sr.session_id = ss.id
        JOIN users u ON ss.user_id = u.id
        JOIN prolific_demographics pd ON u.username = pd.prolific_participant_id
        WHERE ss.simulated_session_id IS NULL
          AND pd.education_level IS NOT NULL
          AND ss.id IN (
              SELECT session_id FROM learning_steps
              GROUP BY session_id
              HAVING COUNT(DISTINCT step_index) = 10
          )
        ORDER BY ss.id
    """, conn)
    conn.close()
    path = os.path.join(OUT_DIR, "crowd_survey.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows -> crowd_survey.csv")


def export_llm(model_name: str, label: str):
    """Export per-run and averaged survey results from survey_test_retest."""
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            id,
            session_id,
            education_level,
            run_number,
            accomplishment,
            effort_required,
            mental_demand,
            perceived_controllability,
            temporal_demand,
            trust
        FROM survey_test_retest
        WHERE model_name = ?
        ORDER BY id DESC
    """, conn, params=(model_name,))
    conn.close()

    if df.empty:
        print(f"  [{label}] No data in survey_test_retest, skipping.")
        return

    # Keep only the 55 most recent rows per run_number (by insertion order = id)
    df = (
        df.sort_values("id", ascending=False)
          .groupby("run_number", group_keys=False)
          .head(55)
          .sort_values(["run_number", "session_id"])
          .drop(columns=["id"])
          .reset_index(drop=True)
    )

    model_dir = os.path.join(OUT_DIR, label)
    os.makedirs(model_dir, exist_ok=True)

    # Per-run exports
    for run in RUNS_TO_EXPORT:
        run_df = df[df["run_number"] == run].drop(columns=["run_number"]).reset_index(drop=True)
        path = os.path.join(model_dir, f"run{run}_survey.csv")
        run_df.to_csv(path, index=False)
        print(f"  Saved {len(run_df)} rows -> {label}/run{run}_survey.csv")

    # Median over all runs per session
    med_df = (
        df.groupby(["session_id", "education_level"])[SURVEY_COLS]
        .median()
        .round(4)
        .reset_index()
    )
    path = os.path.join(model_dir, "med_survey.csv")
    med_df.to_csv(path, index=False)
    print(f"  Saved {len(med_df)} rows -> {label}/med_survey.csv")


def compute_stats(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Compute median and IQR for each survey metric in a dataframe."""
    rows = []
    for col in SURVEY_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        rows.append({
            "source":  source,
            "metric":  col,
            "n":       len(vals),
            "mean":    round(vals.mean(), 4),
            "median":  round(vals.median(), 4),
            "iqr":     round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4),
            "q25":     round(float(np.percentile(vals, 25)), 4),
            "q75":     round(float(np.percentile(vals, 75)), 4),
        })
    return pd.DataFrame(rows)


def export_stats_summary():
    """Compute median+IQR for every exported CSV and save a single summary."""
    all_stats = []

    # Crowd
    crowd_path = os.path.join(OUT_DIR, "crowd_survey.csv")
    if os.path.exists(crowd_path):
        df = pd.read_csv(crowd_path)
        all_stats.append(compute_stats(df, "crowd"))

    # LLM models
    for label in MODEL_LABELS.values():
        model_dir = os.path.join(OUT_DIR, label)

        for run in RUNS_TO_EXPORT:
            path = os.path.join(model_dir, f"run{run}_survey.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_stats.append(compute_stats(df, f"{label}_run{run}"))

        med_path = os.path.join(model_dir, "med_survey.csv")
        if os.path.exists(med_path):
            df = pd.read_csv(med_path)
            all_stats.append(compute_stats(df, f"{label}_med"))

    summary = pd.concat(all_stats, ignore_index=True)
    path = os.path.join(OUT_DIR, "stats_summary.csv")
    summary.to_csv(path, index=False)
    print(f"  Saved {len(summary)} rows -> stats_summary.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n--- Crowd (real learners: 10 steps + survey completed) ---")
    export_crowd()

    for model_name, label in MODEL_LABELS.items():
        print(f"\n--- {label} ({model_name}) ---")
        export_llm(model_name, label)

    print("\n--- Stats summary (median + IQR) ---")
    export_stats_summary()

    print(f"\nDone. Files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
