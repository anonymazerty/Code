from __future__ import annotations

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = os.environ.get(
    "REALUSER_DB_PATH",
    "./real-user-study/TestReco/results.db",
)

RESULTS_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../llm_user_study/results")
)

QUESTIONS = [
    ("accomplishment",            "Q1 Accomplishment"),
    ("effort_required",           "Q2 Effort Required"),
    ("mental_demand",             "Q3 Mental Demand"),
    ("perceived_controllability", "Q4 Controllability"),
    ("temporal_demand",           "Q5 Temporal Demand"),
    ("trust",                     "Q6 Trust"),
    ("frustration",               "Q7 Frustration"),
]

EDUCATION_GROUPS = [
    ("graduate",      "Graduate"),
    ("undergraduate", "Undergraduate"),
    ("high_school",   "High School"),
]

# (db_key, display_label, color)
SOURCES = [
    ("crowd",                "Crowd",   "#2E86AB"),
    ("llama-3-8b",           "LLaMA",   "#E84855"),
    ("gpt-5-2025-08-07",     "GPT-5",   "#2A9D8F"),
    ("mistral-24b-instruct", "Mistral", "#F4A261"),
]

Q_COLS = [col for col, _ in QUESTIONS]

PROMPT_DISPLAY = {
    "detailed": "Calibrated",
    "general":  "Task-Constrained",
    "simple":   "Role-Anchored",
}


#  data loaders 

def load_crowd() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT sr.session_id, pd.education_level,
               sr.accomplishment, sr.effort_required, sr.mental_demand,
               sr.perceived_controllability, sr.temporal_demand,
               sr.frustration, sr.trust
        FROM survey_responses sr
        JOIN study_sessions ss ON sr.session_id = ss.id
        JOIN users u           ON ss.user_id = u.id
        JOIN prolific_demographics pd ON u.username = pd.prolific_participant_id
        WHERE ss.simulated_session_id IS NULL
          AND pd.education_level IS NOT NULL
          AND ss.id IN (
              SELECT session_id FROM learning_steps
              GROUP BY session_id HAVING COUNT(DISTINCT step_index) = 10
          )
    """, conn)
    conn.close()
    df["education_level"] = df["education_level"].str.lower().str.strip().str.replace(" ", "_")
    return df


def load_llm_judge(model_name: str) -> pd.DataFrame:
    """survey_test_retest: 5 most recent rows per session, averaged."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT t.id, t.session_id, t.education_level,
               t.accomplishment, t.effort_required, t.mental_demand,
               t.perceived_controllability, t.temporal_demand,
               t.frustration, t.trust
        FROM survey_test_retest t
        WHERE t.model_name = ?
        ORDER BY t.session_id, t.id DESC
    """, conn, params=(model_name,))
    conn.close()
    if df.empty:
        return df
    df["education_level"] = df["education_level"].str.lower().str.strip().str.replace(" ", "_")
    df5 = df.groupby("session_id").head(5)
    return (df5.groupby(["session_id", "education_level"])[Q_COLS]
               .mean().reset_index())


def load_llm_e2e(model_name: str, prompt_type: str) -> pd.DataFrame:
    """survey_responses for simulated sessions: 5 most recent runs per real session, averaged."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT ss_sim.id AS sim_id,
               ss_sim.simulated_session_id AS session_id,
               pd.education_level,
               sr.accomplishment, sr.effort_required, sr.mental_demand,
               sr.perceived_controllability, sr.temporal_demand,
               sr.frustration, sr.trust
        FROM study_sessions ss_sim
        JOIN survey_responses sr    ON sr.session_id    = ss_sim.id
        JOIN study_sessions ss_real ON ss_real.id       = ss_sim.simulated_session_id
        JOIN users u_real           ON ss_real.user_id  = u_real.id
        JOIN prolific_demographics pd ON u_real.username = pd.prolific_participant_id
        WHERE ss_sim.simulated_session_id IS NOT NULL
          AND ss_sim.model_name  = ?
          AND ss_sim.prompt_type = ?
          AND pd.education_level IS NOT NULL
        ORDER BY ss_sim.simulated_session_id, ss_sim.id DESC
    """, conn, params=(model_name, prompt_type))
    conn.close()
    if df.empty:
        return df
    df["education_level"] = df["education_level"].str.lower().str.strip().str.replace(" ", "_")
    df5 = df.groupby("session_id").head(5)
    return (df5.groupby(["session_id", "education_level"])[Q_COLS]
               .mean().reset_index())


#  plotting 

def plot_comparison(datasets: dict[str, pd.DataFrame], title: str, output_path: str):
    active = {k: v for k, v in datasets.items() if not v.empty}
    if not active:
        print(f"  ✗ No data for '{title}', skipping.")
        return

    n_s = len(SOURCES)
    bar_w = 0.18
    offsets = np.linspace(-(n_s - 1) / 2, (n_s - 1) / 2, n_s) * bar_w
    x = np.arange(len(QUESTIONS))

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (key, label, color) in enumerate(SOURCES):
        df = active.get(key, pd.DataFrame())
        n = len(df)
        means = [df[c].dropna().mean() if n > 0 else np.nan for c in Q_COLS]
        sds   = [df[c].dropna().std()  if n > 0 else np.nan for c in Q_COLS]

        ax.bar(x + offsets[i], means, width=bar_w, color=color, alpha=0.85,
               label=f"{label} (n={n})", zorder=3)
        ax.errorbar(x + offsets[i], means, yerr=sds,
                    fmt="none", color="black", capsize=3, linewidth=1, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in QUESTIONS], fontsize=22,
                       rotation=15, ha="right")
    ax.set_ylim(1, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Mean Score (1–5)", fontsize=28)
    ax.set_title(title, fontsize=30, fontweight="bold")
    ax.legend(fontsize=19, frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(output_path)}")


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
    if args.agent == "judge":
        sub = ""
        pt_display = ""
    else:
        pt_display = PROMPT_DISPLAY.get(args.prompt_type, args.prompt_type)
        sub = pt_display.lower().replace(" ", "-")
    out_dir = os.path.join(RESULTS_BASE, agent_name, "survey_histograms", sub)
    os.makedirs(out_dir, exist_ok=True)

    pt_label = "" if args.agent == "judge" else f" [{pt_display}]"
    print(f"\n--- Survey histograms ({args.agent}{pt_label}) ---")

    # Load all data
    crowd_df = load_crowd()
    print(f"  Crowd: {len(crowd_df)} sessions")

    llm_dfs = {}
    for key, label, _ in SOURCES[1:]:
        if args.agent == "judge":
            df = load_llm_judge(key)
        else:
            df = load_llm_e2e(key, args.prompt_type)
        llm_dfs[key] = df
        print(f"  {label}: {len(df)} sessions")

    def make_datasets(edu_filter=None):
        d = {"crowd": crowd_df if edu_filter is None
             else crowd_df[crowd_df["education_level"] == edu_filter]}
        for key, df in llm_dfs.items():
            d[key] = df if edu_filter is None else df[df["education_level"] == edu_filter]
        return d

    suffix = f" — e2e{pt_label}" if args.agent == "e2e" else " — JudgeAgent"

    plot_comparison(make_datasets(),
                    f"Survey Responses: Crowd vs LLM Simulators{suffix} — All Learners",
                    os.path.join(out_dir, "overall.png"))

    for edu_key, edu_label in EDUCATION_GROUPS:
        plot_comparison(make_datasets(edu_key),
                        f"Survey Responses: Crowd vs LLM Simulators{suffix} — {edu_label}",
                        os.path.join(out_dir, f"{edu_key}.png"))

    print(f"\nAll files saved to: {out_dir}")


if __name__ == "__main__":
    main()
