from __future__ import annotations

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DB_PATH = os.environ.get(
    "REALUSER_DB_PATH",
    "./real-user-study/TestReco/results.db",
)

RESULTS_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../llm_user_study/results")
)

# Okabe-Ito colorblind-safe palette + distinct markers
MODEL_LABELS = {
    "llama-3-8b":           ("llama",   "LLaMA",   "#D55E00", "s"),   # vermilion, square
    "mistral-24b-instruct": ("mistral", "Mistral", "#E69F00", "^"),   # orange,    triangle
    "gpt-5-2025-08-07":     ("gpt",     "GPT-5",   "#009E73", "D"),   # green,     diamond
}

PROMPT_DISPLAY = {
    "detailed": "Calibrated",
    "general":  "Task-Constrained",
    "simple":   "Role-Anchored",
}

COLOR_CROWD  = "#0072B2"   # Okabe-Ito blue
MARKER_CROWD = "o"
STEPS = 11   # points 0..10
Y_MIN = 0.30  # start y-axis here (mastery rarely falls below 0.3)


#  data loaders 

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
    """LLM trajectories for e2e: one row per (simulated_session_id, step_index)
    averaged over the 5 most recent runs per session."""
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

    # Keep only the 5 most recent sim_ids per real session
    recent = (
        df[["real_session_id", "sim_id"]].drop_duplicates()
        .sort_values("sim_id", ascending=False)
        .groupby("real_session_id").head(5)
    )
    df5 = df[df["sim_id"].isin(recent["sim_id"])]

    # Average mastery_before/after across the 5 runs per (real_session_id, step_index)
    avg = (
        df5.groupby(["real_session_id", "step_index"])[["mastery_before", "mastery_after"]]
        .mean()
        .reset_index()
    )
    return avg


#  matrix builder 

def build_matrix(df: pd.DataFrame, session_col: str = "session_id") -> tuple[list, np.ndarray]:
    """(n_sessions × 11) matrix: col 0 = initial mastery, cols 1–10 = mastery_after."""
    sessions = sorted(df[session_col].unique())
    mat = np.full((len(sessions), STEPS), np.nan)
    for i, sid in enumerate(sessions):
        rows = df[df[session_col] == sid].sort_values("step_index")
        if rows.empty:
            continue
        mat[i, 0] = rows.iloc[0]["mastery_before"]
        for _, row in rows.iterrows():
            mat[i, int(row["step_index"]) + 1] = row["mastery_after"]
    return sessions, mat


def summarise(mat: np.ndarray) -> dict:
    return {
        "mean": np.nanmean(mat, axis=0),
        "std":  np.nanstd(mat, axis=0),
        "q25":  np.nanpercentile(mat, 25, axis=0),
        "q75":  np.nanpercentile(mat, 75, axis=0),
        "n":    mat.shape[0],
    }


#  CSV export 

def save_traj_csv(stats: dict, name: str, out_dir: str):
    df = pd.DataFrame({
        "step":   range(STEPS),
        "mean":   np.round(stats["mean"], 4),
        "std":    np.round(stats["std"], 4),
        "q25":    np.round(stats["q25"], 4),
        "q75":    np.round(stats["q75"], 4),
    })
    path = os.path.join(out_dir, f"{name}_avg_trajectory.csv")
    df.to_csv(path, index=False)
    print(f"  Saved CSV -> {name}_avg_trajectory.csv  (n={stats['n']})")
    return df


#  plotting 

def _add_traj(ax, stats: dict, color: str, label: str, marker: str = "o"):
    s = np.arange(STEPS)
    ax.fill_between(s, stats["q25"], stats["q75"],
                    alpha=0.18, color=color)
    ax.plot(s, stats["mean"], color=color, linewidth=3.5, marker=marker,
            markersize=10, markeredgewidth=1.2, markeredgecolor="white",
            label=label)


def plot_judge(crowd_stats: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(14, 8))
    _add_traj(ax, crowd_stats, COLOR_CROWD, "Crowd", MARKER_CROWD)
    _style_traj_ax(ax, "Crowd — Average Learning Trajectory")
    fig.savefig(os.path.join(out_dir, "crowd_avg_trajectory.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved PNG -> crowd_avg_trajectory.png")


def plot_e2e(crowd_stats: dict, llm_stats: dict, prompt_type: str, out_dir: str):
    display_name = PROMPT_DISPLAY.get(prompt_type, prompt_type)
    fig, ax = plt.subplots(figsize=(14, 8))
    _add_traj(ax, crowd_stats, COLOR_CROWD, "Crowd", MARKER_CROWD)
    for _, (_, display, color, marker) in MODEL_LABELS.items():
        if display not in llm_stats:
            continue
        _add_traj(ax, llm_stats[display], color, display, marker)
    title = f"Average Learning Trajectory — {display_name}"
    _style_traj_ax(ax, title)
    fname = f"avg_trajectories_{display_name.lower().replace(' ', '-')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PNG -> {fname}")


def _style_traj_ax(ax, title: str):
    ax.set_xlabel("Learning Step", fontsize=30)
    ax.set_ylabel("Mastery", fontsize=30)
    ax.set_title(title, fontsize=32, fontweight="bold")
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(Y_MIN, 1.02)
    ax.set_xticks(range(STEPS))
    ax.set_xticklabels([str(i) for i in range(STEPS)])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=22)
    ax.legend(fontsize=22, frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()


#  main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["judge", "e2e"], required=True)
    parser.add_argument("--prompt-type", choices=["detailed", "general", "simple"],
                        help="Required for --agent e2e")
    args = parser.parse_args()

    if args.agent == "e2e" and not args.prompt_type:
        parser.error("--prompt-type is required for --agent e2e")

    # Output directory
    agent_name = "JudgeAgent" if args.agent == "judge" else "e2eAgent"
    out_dir = os.path.join(RESULTS_BASE, agent_name, "learning_trajectories")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- Trajectories ({args.agent}) ---")

    # Crowd
    crowd_df = load_crowd_steps()
    print(f"  Crowd: {crowd_df['session_id'].nunique()} sessions")
    _, crowd_mat = build_matrix(crowd_df)
    crowd_stats = summarise(crowd_mat)
    save_traj_csv(crowd_stats, "crowd", out_dir)

    if args.agent == "judge":
        plot_judge(crowd_stats, out_dir)

    else:
        pt_slug = PROMPT_DISPLAY.get(args.prompt_type, args.prompt_type).lower().replace(" ", "-")
        llm_stats = {}
        for model_name, (slug, display, color, marker) in MODEL_LABELS.items():
            df = load_llm_steps(model_name, args.prompt_type)
            if df.empty:
                print(f"  [{display}] no data for prompt_type='{args.prompt_type}', skipping.")
                continue
            n_sess = df["real_session_id"].nunique()
            print(f"  {display}: {n_sess} sessions (avg over 5 most recent runs)")
            _, mat = build_matrix(df, session_col="real_session_id")
            stats = summarise(mat)
            save_traj_csv(stats, f"{slug}_{pt_slug}", out_dir)
            llm_stats[display] = stats

        plot_e2e(crowd_stats, llm_stats, args.prompt_type, out_dir)

    print(f"\nFiles saved to: {out_dir}")


if __name__ == "__main__":
    main()
