import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DB_PATH = os.environ.get(
    "REALUSER_DB_PATH",
    "./real-user-study/TestReco/results.db"
)

OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../llm_user_study/results/JudgeAgent/learning_trajectories")
)


def load_trajectories() -> pd.DataFrame:
    """
    Return one row per (session_id, point) where point 0 = initial mastery
    (mastery_before at step 0) and points 1..10 = mastery_after at steps 0..9.
    Only includes the 55 valid crowd sessions.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            ls.session_id,
            ls.step_index,
            ls.mastery_before,
            ls.mastery_after
        FROM learning_steps ls
        JOIN study_sessions ss ON ls.session_id = ss.id
        JOIN users u ON ss.user_id = u.id
        JOIN prolific_demographics pd ON u.username = pd.prolific_participant_id
        JOIN survey_responses sr ON sr.session_id = ss.id
        WHERE ss.simulated_session_id IS NULL
          AND pd.education_level IS NOT NULL
          AND ss.id IN (
              SELECT session_id FROM learning_steps
              GROUP BY session_id
              HAVING COUNT(DISTINCT step_index) = 10
          )
        ORDER BY ls.session_id, ls.step_index
    """, conn)
    conn.close()
    return df


def build_matrix(df: pd.DataFrame):
    """
    Returns (sessions x 11) matrix where column i = mastery at point i.
    Point 0 = initial, points 1-10 = after steps 0-9.
    """
    sessions = sorted(df["session_id"].unique())
    mat = np.full((len(sessions), 11), np.nan)
    for i, sid in enumerate(sessions):
        rows = df[df["session_id"] == sid].sort_values("step_index")
        # point 0: mastery before step 0
        mat[i, 0] = rows.iloc[0]["mastery_before"]
        for _, row in rows.iterrows():
            mat[i, int(row["step_index"]) + 1] = row["mastery_after"]
    return sessions, mat


def save_csv(sessions, mat, out_dir):
    steps = list(range(11))
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    q25  = np.nanpercentile(mat, 25, axis=0)
    q75  = np.nanpercentile(mat, 75, axis=0)

    summary = pd.DataFrame({
        "step":   steps,
        "mean":   np.round(mean, 4),
        "std":    np.round(std, 4),
        "q25":    np.round(q25, 4),
        "q75":    np.round(q75, 4),
    })
    path = os.path.join(out_dir, "crowd_avg_trajectory.csv")
    summary.to_csv(path, index=False)
    print(f"  Saved -> crowd_avg_trajectory.csv  ({len(sessions)} learners)")
    return summary


def save_plot(summary: pd.DataFrame, n_learners: int, out_dir: str):
    steps = summary["step"].values
    mean  = summary["mean"].values
    q25   = summary["q25"].values
    q75   = summary["q75"].values

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.fill_between(steps, q25, q75, alpha=0.18, color="#2E86AB", label="IQR (Q25–Q75)")
    ax.plot(steps, mean, color="#2E86AB", linewidth=2.2, marker="o",
            markersize=5, label=f"Mean (n={n_learners})")

    ax.set_xlabel("Learning Step", fontsize=12)
    ax.set_ylabel("Mastery", fontsize=12)
    ax.set_title("Crowd — Average Learning Trajectory", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(range(11))
    ax.set_xticklabels([str(i) for i in range(11)])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "crowd_avg_trajectory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> crowd_avg_trajectory.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n--- Crowd average learning trajectory ---")

    df = load_trajectories()
    print(f"  Loaded {df['session_id'].nunique()} sessions, {len(df)} step rows")

    sessions, mat = build_matrix(df)
    summary = save_csv(sessions, mat, OUT_DIR)
    save_plot(summary, len(sessions), OUT_DIR)

    print(f"\nFiles saved to: {OUT_DIR}")
    print("\nStep-level summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
