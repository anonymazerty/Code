from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

#  config 

DB_PATH = os.environ.get(
    "REALUSER_DB_PATH",
    "./real-user-study/TestReco/results.db",
)

RESULTS_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../llm_user_study/results")
)

MODEL_LABELS = {
    "llama-3-8b":           ("llama",   "LLaMA-8B"),
    "mistral-24b-instruct": ("mistral", "Mistral-24B"),
    "gpt-5-2025-08-07":     ("gpt",     "GPT-5"),
}

PROMPT_TYPES = {
    "general":  "Task-Constrained",
    "simple":   "Role-Anchored",
    "detailed": "Calibrated",
}

STEPS = 11  # points 0..10


#  data loaders (same logic as plot_trajectories.py) 

def load_crowd_steps() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT ls.session_id, ls.step_index, ls.mastery_before, ls.mastery_after
        FROM learning_steps ls
        JOIN study_sessions ss ON ls.session_id = ss.id
        JOIN users u ON ss.user_id = u.id
        JOIN prolific_demographics pd ON u.username = pd.prolific_participant_id
        JOIN survey_responses sr ON sr.session_id = ss.id
        WHERE ss.simulated_session_id IS NULL
          AND pd.education_level IS NOT NULL
          AND ss.id IN (
              SELECT session_id FROM learning_steps
              GROUP BY session_id HAVING COUNT(DISTINCT step_index) = 10
          )
        ORDER BY ls.session_id, ls.step_index
    """, conn)
    conn.close()
    return df


def load_llm_steps(model_name: str, prompt_type: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            ss.simulated_session_id AS real_session_id,
            ss.id AS sim_id,
            ls.step_index,
            ls.mastery_before,
            ls.mastery_after
        FROM learning_steps ls
        JOIN study_sessions ss ON ls.session_id = ss.id
        WHERE ss.simulated_session_id IS NOT NULL
          AND ss.model_name  = ?
          AND ss.prompt_type = ?
        ORDER BY ss.simulated_session_id, ss.id DESC, ls.step_index
    """, conn, params=(model_name, prompt_type))
    conn.close()

    if df.empty:
        return df

    recent = (
        df[["real_session_id", "sim_id"]].drop_duplicates()
        .sort_values("sim_id", ascending=False)
        .groupby("real_session_id").head(5)
    )
    df5 = df[df["sim_id"].isin(recent["sim_id"])]

    avg = (
        df5.groupby(["real_session_id", "step_index"])[["mastery_before", "mastery_after"]]
        .mean()
        .reset_index()
    )
    return avg


#  matrix / stats 

def build_matrix(df: pd.DataFrame, session_col: str = "session_id") -> np.ndarray:
    sessions = sorted(df[session_col].unique())
    mat = np.full((len(sessions), STEPS), np.nan)
    for i, sid in enumerate(sessions):
        rows = df[df[session_col] == sid].sort_values("step_index")
        if rows.empty:
            continue
        mat[i, 0] = rows.iloc[0]["mastery_before"]
        for _, row in rows.iterrows():
            mat[i, int(row["step_index"]) + 1] = row["mastery_after"]
    return mat


def mean_trajectory(mat: np.ndarray) -> np.ndarray:
    return np.nanmean(mat, axis=0)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


#  main 

def main():
    print("=" * 70)
    print("LEARNING-TRAJECTORY RMSE — E2E Agent vs Crowd (full precision)")
    print("=" * 70)

    # Crowd
    crowd_df = load_crowd_steps()
    n_crowd = crowd_df["session_id"].nunique()
    crowd_mat = build_matrix(crowd_df)
    crowd_mean = mean_trajectory(crowd_mat)
    print(f"\nCrowd sessions: {n_crowd}")

    rows = []

    for pt_key, pt_display in PROMPT_TYPES.items():
        for model_name, (slug, display) in MODEL_LABELS.items():
            llm_df = load_llm_steps(model_name, pt_key)
            if llm_df.empty:
                print(f"  [{pt_display} / {display}] no data, skipping")
                continue
            n_llm = llm_df["real_session_id"].nunique()
            llm_mat = build_matrix(llm_df, session_col="real_session_id")
            llm_mean = mean_trajectory(llm_mat)
            r = rmse(crowd_mean, llm_mean)
            p = pearson(crowd_mean, llm_mean)
            rows.append({
                "prompt": pt_display,
                "model": display,
                "n_crowd": n_crowd,
                "n_llm": n_llm,
                "RMSE": r,
                "Pearson": p,
            })

    result = pd.DataFrame(rows)

    # Console output — full precision
    print("\n" + "-" * 70)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", lambda x: f"{x}")
    print(result.to_string(index=False))
    print("-" * 70)

    # Save CSV (no rounding)
    out_dir = os.path.join(RESULTS_BASE, "e2eAgent", "learning_trajectories")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "trajectory_rmse.csv")
    result.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
