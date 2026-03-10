import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent  # quizcomp_llm_study/
DB_DIR = BASE_DIR / "AgentJUDGEresults"
OUTPUT_DIR = BASE_DIR / "Results" / "JudgeAgent" / "survey_histograms"

CROWD_DB = DB_DIR / "quizcomp_study.sqlite"
LLM_DBS = {
    "llama":   DB_DIR / "llm_results_llama.db",
    "gpt5":    DB_DIR / "llm_results_GPT5.db",
    "mistral": DB_DIR / "llm_results_Mistral.db",
}

QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
Q_COLS_CROWD = [
    "q1_accomplishment", "q2_effort", "q3_mental_demand",
    "q4_controllability", "q5_temporal_demand", "q6_satisfaction_trust",
]
Q_COLS_LLM = [
    "q1_accomplishment", "q2_effort", "q3_mental_demand",
    "q4_controllability", "q5_temporal_demand", "q6_satisfaction",
]
Q_LABELS = [
    "Q1:\nAccomplishment",
    "Q2:\nEffort",
    "Q3:\nMental\nDemand",
    "Q4:\nControllability",
    "Q5:\nTemporal\nDemand",
    "Q6:\nSatisfaction",
]

# (key, legend-label, colour)
SOURCES = [
    ("crowd",   "Crowd",   "#2E86AB"),
    ("llama",   "LLaMA",   "#E84855"),
    ("gpt5",    "GPT-5",   "#2A9D8F"),
    ("mistral", "Mistral", "#F4A261"),
]


# Data loading 
def load_crowd() -> pd.DataFrame:
    """Return one row per completed participant with columns Q1–Q6."""
    conn = sqlite3.connect(CROWD_DB)
    df = pd.read_sql_query(
        f"""
        SELECT {', '.join(f'sr.{c}' for c in Q_COLS_CROWD)}
        FROM survey_responses sr
        JOIN study_sessions s ON sr.session_id = s.id
        WHERE s.status = 'completed'
        """,
        conn,
    )
    conn.close()
    df.columns = QUESTIONS
    return df


def load_llm(key: str) -> pd.DataFrame:
    """Return one row per persona (averaged across 5 runs) with columns Q1–Q6."""
    conn = sqlite3.connect(LLM_DBS[key])
    df = pd.read_sql_query(
        f"""
        SELECT s.persona_id,
               {', '.join(f'sr.{c}' for c in Q_COLS_LLM)}
        FROM llm_survey_responses sr
        JOIN simulations s ON sr.simulation_id = s.id
        """,
        conn,
    )
    conn.close()
    q_cols = df.columns[1:] 
    df_avg = df.groupby("persona_id")[q_cols].mean().reset_index()
    df_avg = df_avg.drop(columns=["persona_id"])
    df_avg.columns = QUESTIONS
    return df_avg


# Plotting 
def plot_comparison():
    """Grouped bar chart: mean ± std, one bar per source per question."""
    # Load all data
    frames: dict[str, pd.DataFrame] = {"crowd": load_crowd()}
    for key in ("llama", "gpt5", "mistral"):
        frames[key] = load_llm(key)

    n_questions = len(QUESTIONS)
    n_sources = len(SOURCES)
    bar_w = 0.18
    x = np.arange(n_questions)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (key, label, colour) in enumerate(SOURCES):
        df = frames[key]
        means = df[QUESTIONS].mean().values
        stds = df[QUESTIONS].std().values
        n = len(df)
        offset = (idx - (n_sources - 1) / 2) * bar_w

        ax.bar(
            x + offset,
            means,
            bar_w,
            yerr=stds,
            capsize=3,
            label=f"{label} (n={n})",
            color=colour,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            error_kw=dict(lw=1.2, capthick=1, zorder=4),
        )

    # Styles
    ax.set_xticks(x)
    ax.set_xticklabels(Q_LABELS, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Mean Score (1–5)", fontsize=13)
    ax.set_title(
        "JudgeAgent: CROWD vs LLM Survey Responses (Mean ± Std)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 5.8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / "overall.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {png_path}")

    pdf_path = OUTPUT_DIR / "overall.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved → {pdf_path}")

    plt.close()


if __name__ == "__main__":
    plot_comparison()
