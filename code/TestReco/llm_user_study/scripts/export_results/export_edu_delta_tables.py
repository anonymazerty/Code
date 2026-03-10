import os
import pandas as pd

RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../llm_user_study/results/JudgeAgent/results_as_csv")
)

SURVEY_COLS = [
    "accomplishment",
    "effort_required",
    "mental_demand",
    "perceived_controllability",
    "temporal_demand",
    "trust",
]

Q_LABELS = {
    "accomplishment":            "Q1: Accomplishment",
    "effort_required":           "Q2: Effort",
    "mental_demand":             "Q3: Mental Demand",
    "perceived_controllability": "Q4: Controllability",
    "temporal_demand":           "Q5: Temporal Demand",
    "trust":                     "Q6: Satisfaction / Trust",
}

EDU_GROUPS = [
    ("high_school",   "HS"),
    ("undergraduate", "UG"),
    ("graduate",      "GR"),
]

MODELS = ["llama", "gpt", "mistral"]


def group_median(df: pd.DataFrame, edu: str, col: str) -> float:
    sub = df[df["education_level"] == edu]
    s = sub[col].dropna()
    return float(s.median()) if len(s) else float("nan")


def build_delta_table(crowd_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    """Single-model table: columns = Metric, HS, UG, GR (exact deltas)."""
    rows = []
    abs_delta_cols = {edu_label: [] for _, edu_label in EDU_GROUPS}

    for col in SURVEY_COLS:
        row = {"Metric": Q_LABELS[col]}
        for edu_key, edu_label in EDU_GROUPS:
            crowd_med = group_median(crowd_df, edu_key, col)
            llm_med   = group_median(llm_df,   edu_key, col)
            delta = llm_med - crowd_med          # exact, no rounding
            row[edu_label] = delta
            abs_delta_cols[edu_label].append(abs(delta))
        rows.append(row)

    # Mean |Δ| summary row
    mean_row = {"Metric": "Mean |Δ| across Qs"}
    for _, edu_label in EDU_GROUPS:
        vals = abs_delta_cols[edu_label]
        mean_row[edu_label] = sum(vals) / len(vals)
    rows.append(mean_row)

    return pd.DataFrame(rows, columns=["Metric"] + [lbl for _, lbl in EDU_GROUPS])


def build_merged_table(crowd_df: pd.DataFrame,
                       model_dfs: dict) -> pd.DataFrame:
    """
    Merged table: columns = Metric,
                            Llama_HS, Llama_UG, Llama_GR,
                            GPT_HS,   GPT_UG,   GPT_GR,
                            Mistral_HS, Mistral_UG, Mistral_GR
    """
    model_label = {"llama": "Llama", "gpt": "GPT", "mistral": "Mistral"}

    col_names = ["Metric"] + [
        f"{model_label[m]}_{edu_label}"
        for m in MODELS
        for _, edu_label in EDU_GROUPS
    ]

    rows = []
    abs_delta_cols = {f"{model_label[m]}_{lbl}": [] for m in MODELS for _, lbl in EDU_GROUPS}

    for col in SURVEY_COLS:
        row = {"Metric": Q_LABELS[col]}
        for m in MODELS:
            llm_df = model_dfs[m]
            for edu_key, edu_label in EDU_GROUPS:
                crowd_med = group_median(crowd_df, edu_key, col)
                llm_med   = group_median(llm_df,   edu_key, col)
                delta = llm_med - crowd_med
                key = f"{model_label[m]}_{edu_label}"
                row[key] = delta
                abs_delta_cols[key].append(abs(delta))
        rows.append(row)

    # Mean |Δ| summary row
    mean_row = {"Metric": "Mean |Δ| across Qs"}
    for key, vals in abs_delta_cols.items():
        mean_row[key] = sum(vals) / len(vals)
    rows.append(mean_row)

    return pd.DataFrame(rows, columns=col_names)


def main():
    crowd_df = pd.read_csv(os.path.join(RESULTS_DIR, "crowd_survey.csv"))
    model_dfs = {}

    for model in MODELS:
        llm_df = pd.read_csv(os.path.join(RESULTS_DIR, model, "med_survey.csv"))
        model_dfs[model] = llm_df

        table = build_delta_table(crowd_df, llm_df)
        out_path = os.path.join(RESULTS_DIR, model, "delta_by_edu.csv")
        table.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        print(table.to_string(index=False))
        print()

    # Merged table
    merged = build_merged_table(crowd_df, model_dfs)
    merged_path = os.path.join(RESULTS_DIR, "delta_by_edu_all_models.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Saved: {merged_path}")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
