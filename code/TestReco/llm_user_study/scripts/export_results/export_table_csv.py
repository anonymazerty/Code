import os
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../llm_user_study/results/JudgeAgent/results_as_csv")
)

SURVEY_COLS = [
    "accomplishment", "effort_required", "mental_demand",
    "perceived_controllability", "temporal_demand", "trust",
]

Q_LABELS = {
    "accomplishment":           "Q1 Accomplishment",
    "effort_required":          "Q2 Effort Required",
    "mental_demand":            "Q3 Mental Demand",
    "perceived_controllability":"Q4 Perceived Controllability",
    "temporal_demand":          "Q5 Temporal Demand",
    "trust":                    "Q6 Trust",
}

MODEL_LABELS = ["llama", "gpt", "mistral"]
RUNS = [1, 2, 3, 4, 5]


#  helpers 

def median_iqr(series: pd.Series):
    vals = series.dropna()
    med = float(vals.median())
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    return med, iqr


def load_summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "stats_summary.csv"))


def summary_row(summary: pd.DataFrame, source: str, metric: str, col: str):
    row = summary[(summary["source"] == source) & (summary["metric"] == metric)]
    return float(row[col].values[0])


#  1. Build delta table 

def build_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in SURVEY_COLS:
        crowd_med = summary_row(summary, "crowd", metric, "median")
        crowd_iqr = summary_row(summary, "crowd", metric, "iqr")

        row = {
            "Metric":        Q_LABELS[metric],
            "Crowd_Med":     crowd_med,
            "Crowd_IQR":     crowd_iqr,
        }

        for label in MODEL_LABELS:
            src = f"{label}_med"
            lm = summary_row(summary, src, metric, "median")
            li = summary_row(summary, src, metric, "iqr")
            row[f"{label.capitalize()}_dMed"] = round(lm - crowd_med, 2)
            row[f"{label.capitalize()}_dIQR"] = round(li - crowd_iqr, 2)

        rows.append(row)

    return pd.DataFrame(rows)


#  2. Coherence check 

def coherence_check(summary: pd.DataFrame):
    """
    Recompute median and IQR directly from the per-learner CSVs and compare
    against stats_summary.csv.  Reports any discrepancy > 0.01.
    """
    print("\n=== Coherence Check: stats_summary vs per-learner CSVs ===\n")
    issues = []

    # Crowd
    crowd_df = pd.read_csv(os.path.join(RESULTS_DIR, "crowd_survey.csv"))
    for metric in SURVEY_COLS:
        med, iqr = median_iqr(crowd_df[metric])
        s_med = summary_row(summary, "crowd", metric, "median")
        s_iqr = summary_row(summary, "crowd", metric, "iqr")
        if abs(med - s_med) > 0.01 or abs(iqr - s_iqr) > 0.01:
            issues.append(f"  crowd / {metric}: recomputed med={med:.4f} iqr={iqr:.4f} | summary med={s_med} iqr={s_iqr}")

    # Per-run files
    for label in MODEL_LABELS:
        model_dir = os.path.join(RESULTS_DIR, label)
        for run in RUNS:
            path = os.path.join(model_dir, f"run{run}_survey.csv")
            if not os.path.exists(path):
                issues.append(f"  MISSING: {label}/run{run}_survey.csv")
                continue
            df = pd.read_csv(path)
            src = f"{label}_run{run}"
            for metric in SURVEY_COLS:
                if metric not in df.columns:
                    continue
                med, iqr = median_iqr(df[metric])
                s_med = summary_row(summary, src, metric, "median")
                s_iqr = summary_row(summary, src, metric, "iqr")
                if abs(med - s_med) > 0.01 or abs(iqr - s_iqr) > 0.01:
                    issues.append(
                        f"  {src} / {metric}: recomputed med={med:.4f} iqr={iqr:.4f}"
                        f" | summary med={s_med} iqr={s_iqr}"
                    )

        # Med file
        med_path = os.path.join(model_dir, "med_survey.csv")
        if not os.path.exists(med_path):
            issues.append(f"  MISSING: {label}/med_survey.csv")
            continue
        med_df = pd.read_csv(med_path)
        src = f"{label}_med"
        for metric in SURVEY_COLS:
            if metric not in med_df.columns:
                continue
            med, iqr = median_iqr(med_df[metric])
            s_med = summary_row(summary, src, metric, "median")
            s_iqr = summary_row(summary, src, metric, "iqr")
            if abs(med - s_med) > 0.01 or abs(iqr - s_iqr) > 0.01:
                issues.append(
                    f"  {src} / {metric}: recomputed med={med:.4f} iqr={iqr:.4f}"
                    f" | summary med={s_med} iqr={s_iqr}"
                )

    if issues:
        print("  DISCREPANCIES FOUND:")
        for i in issues:
            print(i)
    else:
        print("  All values match (tolerance 0.01). stats_summary.csv is coherent.")

    print()


#  main 

def main():
    summary = load_summary()

    # 1. Delta table
    table = build_delta_table(summary)
    out_path = os.path.join(RESULTS_DIR, "delta_table.csv")
    table.to_csv(out_path, index=False)
    print(f"\nDelta table saved -> {out_path}")
    print()
    print(table.to_string(index=False))

    # 2. Coherence check
    coherence_check(summary)


if __name__ == "__main__":
    main()
