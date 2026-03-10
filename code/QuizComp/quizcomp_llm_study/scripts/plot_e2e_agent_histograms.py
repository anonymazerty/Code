import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#paths 
BASE_DIR = Path(__file__).resolve().parent.parent  # quizcomp_llm_study/
OUTPUT_ROOT = BASE_DIR / "Results" / "e2eAgent" / "survey_histograms"

# prompt types folders
PROMPT_TYPES = {
    "Calibrated":       ("Calibrated results",       "detailed"),
    "Role-Anchored":    ("Role-Anchored results",    "inbetween"),
    "Task-Constrained": ("Task-Constrained results", "general"),
}

# LLM Mapping
LLMS = {
    "GPT5":    ("gpt",     "GPT-5",   "#2A9D8F"),
    "llama8B": ("llama",   "LLaMA",   "#E84855"),
    "Mistral": ("mistral", "Mistral", "#F4A261"),
}

CROWD_COLOUR = "#2E86AB"

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


# data loading 
def load_crowd(db_path: Path) -> pd.DataFrame:
    """One row per completed CROWD participant, columns Q1–Q6."""
    conn = sqlite3.connect(db_path)
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


def load_llm(db_path: Path) -> pd.DataFrame:
    """One row per persona (averaged across 5 runs), columns Q1–Q6."""
    conn = sqlite3.connect(db_path)
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
    q_cols = df.columns[1:]  # skip persona_id
    df_avg = df.groupby("persona_id")[q_cols].mean().reset_index()
    df_avg = df_avg.drop(columns=["persona_id"])
    df_avg.columns = QUESTIONS
    return df_avg


# Plotting
def plot_grouped_bars(
    frames,
    sources,
    title: str,
    out_dir: Path,
    filename: str = "overall",
):
    """

    Parameters
    ----------
    frames : {key: DataFrame} for each source
    sources : [(key, legend_label, colour), …]
    title : plot title
    out_dir : where to save
    filename : stem without extension
    """
    n_questions = len(QUESTIONS)
    n_sources = len(sources)
    bar_w = 0.18
    x = np.arange(n_questions)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (key, label, colour) in enumerate(sources):
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

    # Style
    ax.set_xticks(x)
    ax.set_xticklabels(Q_LABELS, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Mean Score (1–5)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 5.8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {png_path}")

    pdf_path = out_dir / f"{filename}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved → {pdf_path}")

    plt.close()


#  main 
def main():
    for prompt_label, (db_folder, db_suffix) in PROMPT_TYPES.items():
        db_dir = BASE_DIR / db_folder
        out_base = OUTPUT_ROOT / prompt_label

        print(f"\n{'='*60}")
        print(f"  {prompt_label}")
        print(f"{'='*60}")

        crowd_db = db_dir / "quizcomp_study.sqlite"
        crowd_df = load_crowd(crowd_db)

        llm_frames = {}
        for llm_key, (llm_frag, _, _) in LLMS.items():
            db_path = db_dir / f"llm_results_{llm_frag}_{db_suffix}.db"
            llm_frames[llm_key] = load_llm(db_path)

        all_frames = {"crowd": crowd_df, **llm_frames}
        all_sources = [
            ("crowd", "Crowd", CROWD_COLOUR),
        ] + [
            (llm_key, label, colour)
            for llm_key, (_, label, colour) in LLMS.items()
        ]

        print(f"\n[Overall]")
        plot_grouped_bars(
            all_frames,
            all_sources,
            f"e2eAgent ({prompt_label}): CROWD vs LLM Survey (Mean ± Std)",
            out_base,
        )

        for llm_key, (_, label, colour) in LLMS.items():
            pair_frames = {"crowd": crowd_df, llm_key: llm_frames[llm_key]}
            pair_sources = [
                ("crowd", "Crowd", CROWD_COLOUR),
                (llm_key, label, colour),
            ]

            print(f"\n[{llm_key}]")
            plot_grouped_bars(
                pair_frames,
                pair_sources,
                f"e2eAgent ({prompt_label}): Crowd vs {label} (Mean ± Std)",
                out_base / llm_key,
            )

    print(f"\nDone — all plots saved under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
