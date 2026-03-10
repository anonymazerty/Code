import numpy as np
import pandas as pd
import os
from pathlib import Path

# paths 
BASE_DIR = Path(__file__).resolve().parent.parent          # quizcomp_llm_study/
RESULTS_DIR = BASE_DIR / "Results"

QUESTION_COLS = [
    "Q1_Accomplishment",
    "Q2_Effort",
    "Q3_Mental_Demand",
    "Q4_Controllability",
    "Q5_Temporal_Demand",
    "Q6_Satisfaction",
]
Q_SHORT = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
Q_NAMES = [
    "Accomplishment", "Effort", "Mental Demand",
    "Controllability", "Temporal Demand", "Satisfaction",
]

MODELS = ["GPT5", "llama8B", "Mistral"]
MODEL_LABELS = {"GPT5": "GPT-5", "llama8B": "LLaMA-8B", "Mistral": "Mistral-24B"}

# Configurations to analyse
CONFIGS = []

# JudgeAgent 
_ja = RESULTS_DIR / "JudgeAgent" / "Results_as_csvs"
CONFIGS.append({
    "label":   "JudgeAgent",
    "crowd":   _ja / "crowd_survey.csv",
    "agents":  {m: _ja / m / "median_survey.csv" for m in MODELS},
    "output":  RESULTS_DIR / "JudgeAgent" / "alignment_scores.csv",
})

# e2eAgent 
for prompt in ("Calibrated", "Role-Anchored", "Task-Constrained"):
    _ea = RESULTS_DIR / "e2eAgent" / "Results_as_csvs" / prompt
    CONFIGS.append({
        "label":   f"e2eAgent – {prompt}",
        "crowd":   _ea / "crowd_survey.csv",
        "agents":  {m: _ea / m / "median_survey.csv" for m in MODELS},
        "output":  RESULTS_DIR / "e2eAgent" / f"alignment_{prompt}.csv",
    })

N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
SEED = 42


# ====================================================================
# 1.  BOOTSTRAP HELPERS
# ====================================================================

def bootstrap_ci(values, stat_fn, n_bootstrap=N_BOOTSTRAP,
                 ci_level=CI_LEVEL, seed=SEED):
    """Return (point_estimate, ci_lower, ci_upper)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    point = stat_fn(values)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boots[i] = stat_fn(rng.choice(values, size=n, replace=True))
    alpha = 1 - ci_level
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return point, lo, hi


def iqr(values):
    return np.percentile(values, 75) - np.percentile(values, 25)


# ====================================================================
# 2.  COMPUTE ALIGNMENT TABLE FOR ONE CONFIGURATION
# ====================================================================

def compute_alignment_table(crowd_csv, agent_csvs, role=None):
    """
    Returns a DataFrame with crowd CIs, agent stats, compatibility, and
    alignment scores for all models and questions.
    If role is provided, filter both crowd and agent data to that role.
    """
    crowd = pd.read_csv(crowd_csv)
    if role is not None and 'Current_Job_Role' in crowd.columns:
        crowd = crowd[crowd['Current_Job_Role'] == role]

    # Crowd bootstrap stats 
    crowd_stats = {}
    for q in QUESTION_COLS:
        vals = crowd[q].dropna().values.astype(float)
        med, med_lo, med_hi = bootstrap_ci(vals, np.median, seed=SEED)
        iq, iq_lo, iq_hi   = bootstrap_ci(vals, iqr, seed=SEED + 1)
        crowd_stats[q] = {
            "median": med, "ci_med": (med_lo, med_hi),
            "iqr": iq,     "ci_iqr": (iq_lo, iq_hi),
        }

    # Per-model alignment 
    rows = []
    for model in MODELS:
        agent = pd.read_csv(agent_csvs[model])
        if role is not None and 'Current_Job_Role' in agent.columns:
            agent = agent[agent['Current_Job_Role'] == role]

        for qi, q in enumerate(QUESTION_COLS):
            a_vals = agent[q].dropna().values.astype(float)
            a_med = np.median(a_vals)
            a_iqr = iqr(a_vals)

            c_med = crowd_stats[q]["median"]
            c_iqr = crowd_stats[q]["iqr"]
            ci_m  = crowd_stats[q]["ci_med"]
            ci_i  = crowd_stats[q]["ci_iqr"]

            delta_med = a_med - c_med
            delta_iqr = a_iqr - c_iqr

            # guard against zero IQR
            denom = c_iqr if c_iqr != 0 else 1.0

            d_q = abs(delta_med) / denom + abs(delta_iqr) / denom
            align = np.exp(-d_q)

            # compatibility: agent stat inside crowd bootstrap CI?
            med_compat = ci_m[0] <= a_med <= ci_m[1]
            iqr_compat = ci_i[0] <= a_iqr <= ci_i[1]

            rows.append({
                "Model": MODEL_LABELS[model],
                "Question": Q_SHORT[qi],
                "Question_Name": Q_NAMES[qi],
                "Crowd_Med": c_med,
                "CI_Med_Lo": ci_m[0],
                "CI_Med_Hi": ci_m[1],
                "Crowd_IQR": c_iqr,
                "CI_IQR_Lo": ci_i[0],
                "CI_IQR_Hi": ci_i[1],
                "Agent_Med": round(a_med, 2),
                "Agent_IQR": round(a_iqr, 2),
                "ΔMed": round(delta_med, 2),
                "ΔIQR": round(delta_iqr, 2),
                "d(q)": round(d_q, 4),
                "AlignScore": round(align, 4),
                "Med_Compat": med_compat,
                "IQR_Compat": iqr_compat,
            })

        # Aggregate row for this model
        model_rows = [r for r in rows if r["Model"] == MODEL_LABELS[model]]
        scores = [r["AlignScore"] for r in model_rows]
        d_vals = [r["d(q)"] for r in model_rows]
        n_med = sum(1 for r in model_rows if r["Med_Compat"] is True)
        n_iqr = sum(1 for r in model_rows if r["IQR_Compat"] is True)

        rows.append({
            "Model": MODEL_LABELS[model],
            "Question": "Avg",
            "Question_Name": "Aggregate",
            "Crowd_Med": "",
            "CI_Med_Lo": "",
            "CI_Med_Hi": "",
            "Crowd_IQR": "",
            "CI_IQR_Lo": "",
            "CI_IQR_Hi": "",
            "Agent_Med": "",
            "Agent_IQR": "",
            "ΔMed": "",
            "ΔIQR": "",
            "d(q)": round(np.mean(d_vals), 4),
            "AlignScore": round(np.mean(scores), 4),
            "Med_Compat": f"{n_med}/6",
            "IQR_Compat": f"{n_iqr}/6",
        })

    return pd.DataFrame(rows)


# ====================================================================
# 3.  PRINT TABLES
# ====================================================================

def print_table(label, df):
    """Print a nicely formatted table to stdout."""
    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"{'=' * 100}")

    for model_label in df["Model"].unique():
        mdf = df[df["Model"] == model_label]
        print(f"\n  ── {model_label} {'─' * (80 - len(model_label))}")
        print(f"  {'Q':<6} {'C_Med':>6} {'CI_Med':>10} {'C_IQR':>6} {'CI_IQR':>10}"
              f" {'A_Med':>6} {'A_IQR':>6}"
              f" {'ΔMed':>6} {'ΔIQR':>6} {'d(q)':>8} {'Align':>8}"
              f" {'M✓':>4} {'I✓':>4}")
        print(f"  {'-'*96}")
        for _, r in mdf.iterrows():
            is_agg = r['Question'] == 'Avg'
            c_med = f"{r['Crowd_Med']:.1f}" if not is_agg else ""
            ci_m  = f"[{r['CI_Med_Lo']:.0f},{r['CI_Med_Hi']:.0f}]" if not is_agg else ""
            c_iqr = f"{r['Crowd_IQR']:.1f}" if not is_agg else ""
            ci_i  = f"[{r['CI_IQR_Lo']:.0f},{r['CI_IQR_Hi']:.0f}]" if not is_agg else ""
            a_med = f"{r['Agent_Med']:.2f}" if not is_agg else ""
            a_iqr = f"{r['Agent_IQR']:.2f}" if not is_agg else ""
            d_med = f"{r['ΔMed']:+.2f}" if not is_agg else ""
            d_iqr = f"{r['ΔIQR']:+.2f}" if not is_agg else ""

            if isinstance(r['Med_Compat'], bool):
                mc = "✓" if r['Med_Compat'] else "✗"
                ic = "✓" if r['IQR_Compat'] else "✗"
            else:
                mc = str(r['Med_Compat'])
                ic = str(r['IQR_Compat'])

            print(f"  {r['Question']:<6} {c_med:>6} {ci_m:>10} {c_iqr:>6} {ci_i:>10}"
                  f" {a_med:>6} {a_iqr:>6}"
                  f" {d_med:>6} {d_iqr:>6} {r['d(q)']:>8.4f} {r['AlignScore']:>8.4f}"
                  f" {mc:>4} {ic:>4}")


# ====================================================================
# 4.  MAIN
# ====================================================================

ROLES = ['College Teacher', 'High School Teacher']

def main():
    for cfg in CONFIGS:
        # Validate paths
        if not cfg["crowd"].exists():
            print(f"\n⚠  Skipping {cfg['label']}: crowd CSV not found → {cfg['crowd']}")
            continue
        missing = [m for m, p in cfg["agents"].items() if not p.exists()]
        if missing:
            print(f"\n⚠  Skipping {cfg['label']}: median CSVs missing for {missing}")
            continue

        # ── Overall alignment ────────────────────────────────────────
        df = compute_alignment_table(cfg["crowd"], cfg["agents"])
        print_table(cfg["label"], df)

        out = cfg["output"]
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n  Saved → {out}")

        # ── Per-role alignment ───────────────────────────────────────
        for role in ROLES:
            role_tag = role.replace(' ', '_')
            df_role = compute_alignment_table(cfg["crowd"], cfg["agents"], role=role)

            role_label = f"{cfg['label']} – {role}"
            print_table(role_label, df_role)

            # Save alongside the overall CSV, include config stem to avoid overwrites
            out_stem = out.stem  # e.g. "alignment_Calibrated" or "alignment_scores"
            role_out = out.parent / "by_role" / f"{out_stem}_{role_tag}.csv"
            role_out.parent.mkdir(parents=True, exist_ok=True)
            df_role.to_csv(role_out, index=False)
            print(f"\n  Saved → {role_out}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
