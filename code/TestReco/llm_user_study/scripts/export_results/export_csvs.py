import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = os.environ.get(
    "REALUSER_DB_PATH",
    "./real-user-study/TestReco/results.db",
)

RESULTS_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../llm_user_study/results")
)

SURVEY_COLS = [
    "accomplishment", "effort_required", "mental_demand",
    "perceived_controllability", "temporal_demand", "trust",
]

MODEL_LABELS = {
    "llama-3-8b":           "llama",
    "mistral-24b-instruct": "mistral",
    "gpt-5-2025-08-07":     "gpt",
}

PROMPT_SLUG = {
    "detailed": "calibrated",
    "general":  "task-constrained",
    "simple":   "role-anchored",
}

RUNS_TO_EXPORT = [1, 2, 3, 4, 5]


def get_conn():
    return sqlite3.connect(DB_PATH)


#  crowd (shared) 

def export_crowd(out_dir: str):
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            ss.id AS session_id,
            pd.education_level,
            sr.accomplishment, sr.effort_required, sr.mental_demand,
            sr.perceived_controllability, sr.temporal_demand,
            sr.trust
        FROM study_sessions ss
        JOIN survey_responses sr ON sr.session_id = ss.id
        JOIN users u ON ss.user_id = u.id
        JOIN prolific_demographics pd ON u.username = pd.prolific_participant_id
        WHERE ss.simulated_session_id IS NULL
          AND pd.education_level IS NOT NULL
          AND ss.id IN (
              SELECT session_id FROM learning_steps
              GROUP BY session_id HAVING COUNT(DISTINCT step_index) = 10
          )
        ORDER BY ss.id
    """, conn)
    conn.close()
    path = os.path.join(out_dir, "crowd_survey.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows -> crowd_survey.csv")


#  judge mode 

def export_llm_judge(model_name: str, label: str, out_dir: str):
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT id, session_id, education_level, run_number,
               accomplishment, effort_required, mental_demand,
               perceived_controllability, temporal_demand, trust
        FROM survey_test_retest
        WHERE model_name = ?
        ORDER BY id DESC
    """, conn, params=(model_name,))
    conn.close()

    if df.empty:
        print(f"  [{label}] No data in survey_test_retest, skipping.")
        return

    # 55 most-recent rows per run_number
    df = (
        df.sort_values("id", ascending=False)
          .groupby("run_number", group_keys=False)
          .head(55)
          .sort_values(["run_number", "session_id"])
          .drop(columns=["id"])
          .reset_index(drop=True)
    )

    model_dir = os.path.join(out_dir, label)
    os.makedirs(model_dir, exist_ok=True)

    for run in RUNS_TO_EXPORT:
        run_df = df[df["run_number"] == run].drop(columns=["run_number"]).reset_index(drop=True)
        path = os.path.join(model_dir, f"run{run}_survey.csv")
        run_df.to_csv(path, index=False)
        print(f"  Saved {len(run_df)} rows -> {label}/run{run}_survey.csv")

    med_df = (
        df.groupby(["session_id", "education_level"])[SURVEY_COLS]
        .median().round(4).reset_index()
    )
    med_df.to_csv(os.path.join(model_dir, "med_survey.csv"), index=False)
    print(f"  Saved {len(med_df)} rows -> {label}/med_survey.csv")


#  e2e mode 

def export_llm_e2e(model_name: str, label: str, prompt_type: str, out_dir: str):
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            ss_sim.id            AS session_id,
            ss_sim.simulated_session_id AS real_session_id,
            pd.education_level,
            sr.accomplishment, sr.effort_required, sr.mental_demand,
            sr.perceived_controllability, sr.temporal_demand,
            sr.trust
        FROM study_sessions ss_sim
        JOIN survey_responses sr   ON sr.session_id  = ss_sim.id
        JOIN study_sessions ss_real ON ss_real.id    = ss_sim.simulated_session_id
        JOIN users u_real           ON ss_real.user_id = u_real.id
        JOIN prolific_demographics pd ON u_real.username = pd.prolific_participant_id
        WHERE ss_sim.simulated_session_id IS NOT NULL
          AND ss_sim.model_name  = ?
          AND ss_sim.prompt_type = ?
          AND pd.education_level IS NOT NULL
        ORDER BY ss_sim.simulated_session_id, ss_sim.id
    """, conn, params=(model_name, prompt_type))
    conn.close()

    if df.empty:
        print(f"  [{label}] No e2e data for prompt_type='{prompt_type}', skipping.")
        return

    # Assign run numbers (1-based) per real_session_id ordered by session_id
    df["run_number"] = (
        df.groupby("real_session_id")["session_id"]
        .rank(method="first").astype(int)
    )

    # Keep 55 most-recent sessions per run_number
    df = (
        df.sort_values("session_id", ascending=False)
          .groupby("run_number", group_keys=False)
          .head(55)
          .sort_values(["run_number", "real_session_id"])
          .reset_index(drop=True)
    )

    model_dir = os.path.join(out_dir, label)
    os.makedirs(model_dir, exist_ok=True)

    for run in RUNS_TO_EXPORT:
        run_df = (
            df[df["run_number"] == run]
            .drop(columns=["run_number"])
            .reset_index(drop=True)
        )
        path = os.path.join(model_dir, f"run{run}_survey.csv")
        run_df.to_csv(path, index=False)
        print(f"  Saved {len(run_df)} rows -> {label}/run{run}_survey.csv")

    med_df = (
        df.groupby(["real_session_id", "education_level"])[SURVEY_COLS]
        .median().round(4).reset_index()
        .rename(columns={"real_session_id": "session_id"})
    )
    med_df.to_csv(os.path.join(model_dir, "med_survey.csv"), index=False)
    print(f"  Saved {len(med_df)} rows -> {label}/med_survey.csv")


#  stats summary (shared) 

def compute_stats(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for col in SURVEY_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "source": source, "metric": col, "n": len(vals),
            "mean":   round(vals.mean(), 4),
            "median": round(vals.median(), 4),
            "iqr":    round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4),
            "q25":    round(float(np.percentile(vals, 25)), 4),
            "q75":    round(float(np.percentile(vals, 75)), 4),
        })
    return pd.DataFrame(rows)


def export_stats_summary(out_dir: str):
    all_stats = []
    crowd_path = os.path.join(out_dir, "crowd_survey.csv")
    if os.path.exists(crowd_path):
        all_stats.append(compute_stats(pd.read_csv(crowd_path), "crowd"))

    for label in MODEL_LABELS.values():
        model_dir = os.path.join(out_dir, label)
        for run in RUNS_TO_EXPORT:
            p = os.path.join(model_dir, f"run{run}_survey.csv")
            if os.path.exists(p):
                all_stats.append(compute_stats(pd.read_csv(p), f"{label}_run{run}"))
        med_p = os.path.join(model_dir, "med_survey.csv")
        if os.path.exists(med_p):
            all_stats.append(compute_stats(pd.read_csv(med_p), f"{label}_med"))

    if not all_stats:
        print("  No CSVs found for stats summary, skipping.")
        return
    summary = pd.concat(all_stats, ignore_index=True)
    path = os.path.join(out_dir, "stats_summary.csv")
    summary.to_csv(path, index=False)
    print(f"  Saved {len(summary)} rows -> stats_summary.csv")


#  main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["judge", "e2e"], required=True)
    parser.add_argument("--prompt-type", choices=["detailed", "general", "simple"],
                        help="Required for --agent e2e")
    args = parser.parse_args()

    if args.agent == "e2e" and not args.prompt_type:
        parser.error("--prompt-type is required for --agent e2e")

    # Resolve output directory
    if args.agent == "judge":
        out_dir = os.path.join(RESULTS_BASE, "JudgeAgent", "results_as_csv")
    else:
        pt_slug = PROMPT_SLUG.get(args.prompt_type, args.prompt_type)
        out_dir = os.path.join(RESULTS_BASE, "e2eAgent", "results_as_csv", pt_slug)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- Crowd (real learners) ---")
    export_crowd(out_dir)

    for model_name, label in MODEL_LABELS.items():
        print(f"\n--- {label} ({model_name}) ---")
        if args.agent == "judge":
            export_llm_judge(model_name, label, out_dir)
        else:
            export_llm_e2e(model_name, label, args.prompt_type, out_dir)

    print(f"\n--- Stats summary ---")
    export_stats_summary(out_dir)

    print(f"\nDone. Files saved to: {out_dir}")


if __name__ == "__main__":
    main()
