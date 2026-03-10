import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from itertools import combinations

# Paths 
BASE_DIR = Path(__file__).resolve().parent.parent          # quizcomp_llm_study/
CSV_DIR  = BASE_DIR / "Results" / "JudgeAgent" / "Results_as_csvs"
OUT_DIR  = BASE_DIR / "Results" / "JudgeAgent"

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

MODELS     = ["GPT5", "llama8B", "Mistral"]
MODEL_LABELS = {
    "GPT5":    "GPT-5",
    "llama8B": "LLaMA-8B",
    "Mistral": "Mistral-24B",
}
GROUP_NAMES = ["CROWD", "GPT-5", "LLaMA-8B", "Mistral-24B"]

ALPHA = 0.05
N_PAIRS = 6                                    # C(4,2) pairwise comparisons
BONFERRONI_ALPHA = ALPHA / N_PAIRS             # ≈ 0.00833


# ====================================================================
# DATA LOADING
# ====================================================================

def load_data():
    """Return per-group arrays: {group_label: {q_col: np.array}}."""

    # ── CROWD ──────────────────────────────────────────────────────
    crowd_df = pd.read_csv(CSV_DIR / "crowd_survey.csv")
    groups = {"CROWD": {col: crowd_df[col].values.astype(float)
                        for col in QUESTION_COLS}}

    # ── LLM agents (per-persona average → 65 observations each) ───
    for m in MODELS:
        avg_df = pd.read_csv(CSV_DIR / m / "average_survey.csv")
        label = MODEL_LABELS[m]
        groups[label] = {col: avg_df[col].values.astype(float)
                         for col in QUESTION_COLS}

    return groups


def load_raw_runs():
    """Return raw (all 325) responses per LLM for full-sample proportions."""
    raw = {}
    for m in MODELS:
        frames = []
        for r in range(1, 6):
            df = pd.read_csv(CSV_DIR / m / f"run_{r}_survey.csv")
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        label = MODEL_LABELS[m]
        raw[label] = {col: combined[col].values.astype(float)
                      for col in QUESTION_COLS}
    return raw


# ====================================================================
# 1.  PROPORTION OF FAVORABLE RESPONSES  (rating ≥ 4)
# ====================================================================

def compute_favorable_proportions(groups, raw_runs):
    """Compute % ≥ 4 for each group × question."""
    rows = []
    for qi, col in enumerate(QUESTION_COLS):
        row = {"Question": Q_SHORT[qi], "Question_Name": Q_NAMES[qi]}

        # CROWD
        vals = groups["CROWD"][col]
        n_fav = np.sum(vals >= 4)
        row["CROWD_n≥4"]   = int(n_fav)
        row["CROWD_N"]     = len(vals)
        row["CROWD_%≥4"]   = round(100 * n_fav / len(vals), 1)

        # LLMs  (average-based, n=65)
        for m in MODELS:
            label = MODEL_LABELS[m]
            vals = groups[label][col]
            n_fav = np.sum(vals >= 4)
            row[f"{label}_n≥4"]   = int(n_fav)
            row[f"{label}_N"]     = len(vals)
            row[f"{label}_%≥4"]   = round(100 * n_fav / len(vals), 1)

            # Also raw (n=325)
            raw_vals = raw_runs[label][col]
            raw_fav  = np.sum(raw_vals >= 4)
            row[f"{label}_raw_n≥4"] = int(raw_fav)
            row[f"{label}_raw_N"]   = len(raw_vals)
            row[f"{label}_raw_%≥4"] = round(100 * raw_fav / len(raw_vals), 1)

        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================
# 2.  KRUSKAL–WALLIS  (omnibus multi-group test)
# ====================================================================

def kruskal_wallis_tests(groups):
    """Run Kruskal–Wallis across the 4 groups for each question."""
    rows = []
    for qi, col in enumerate(QUESTION_COLS):
        samples = [groups[g][col] for g in GROUP_NAMES]
        H, p = stats.kruskal(*samples)
        rows.append({
            "Question":      Q_SHORT[qi],
            "Question_Name": Q_NAMES[qi],
            "H_statistic":   round(H, 4),
            "p_value":       p,
            "Significant_α005": "Yes" if p < ALPHA else "No",
        })
    return pd.DataFrame(rows)


# ====================================================================
# 3.  WILCOXON RANK-SUM (MANN-WHITNEY U) PAIRWISE
# ====================================================================

def wilcoxon_pairwise(groups):
    """Pairwise Mann-Whitney U tests with Bonferroni correction."""
    pairs = list(combinations(GROUP_NAMES, 2))
    rows = []
    for qi, col in enumerate(QUESTION_COLS):
        for g1, g2 in pairs:
            U, p = stats.mannwhitneyu(groups[g1][col], groups[g2][col],
                                      alternative='two-sided')
            rows.append({
                "Question":      Q_SHORT[qi],
                "Question_Name": Q_NAMES[qi],
                "Group_1":       g1,
                "Group_2":       g2,
                "U_statistic":   round(U, 1),
                "p_value":       p,
                "p_bonferroni":  min(p * N_PAIRS, 1.0),
                "Significant_bonf005": "Yes" if p * N_PAIRS < ALPHA else "No",
            })
    return pd.DataFrame(rows)


# ====================================================================
# REPORT WRITER
# ====================================================================

def write_report(fav_df, kw_df, pw_df, groups, path):
    """Write a readable summary to a text file."""
    lines = []
    lines.append("=" * 80)
    lines.append("  JudgeAgent – Statistical Analysis Report")
    lines.append("  Proportion of Favorable Responses, Kruskal–Wallis, Wilcoxon")
    lines.append("=" * 80)

    # ── Descriptive stats ──────────────────────────────────────────
    lines.append("\n\n1. DESCRIPTIVE STATISTICS (per-persona average, n=65 each)")
    lines.append("-" * 80)
    header = f"{'Question':<20} | {'CROWD':>12} | {'GPT-5':>12} | {'LLaMA-8B':>12} | {'Mistral-24B':>12}"
    lines.append(header)
    lines.append("-" * 80)
    for qi, col in enumerate(QUESTION_COLS):
        vals = {}
        for g in GROUP_NAMES:
            v = groups[g][col]
            vals[g] = f"{np.median(v):.1f} ({np.percentile(v,25):.1f}-{np.percentile(v,75):.1f})"
        lines.append(f"{Q_NAMES[qi]:<20} | {vals['CROWD']:>12} | {vals['GPT-5']:>12} | {vals['LLaMA-8B']:>12} | {vals['Mistral-24B']:>12}")

    # ── Favorable proportions ──────────────────────────────────────
    lines.append("\n\n2. PROPORTION OF FAVORABLE RESPONSES (% rating ≥ 4)")
    lines.append("-" * 80)
    lines.append("   Using per-persona averages (n=65 per group)")
    lines.append("")
    header = f"{'Question':<20} | {'CROWD':>8} | {'GPT-5':>8} | {'LLaMA-8B':>8} | {'Mistral-24B':>8}"
    lines.append(header)
    lines.append("-" * 80)
    for _, row in fav_df.iterrows():
        lines.append(
            f"{row['Question_Name']:<20} | "
            f"{row['CROWD_%≥4']:>7.1f}% | "
            f"{row['GPT-5_%≥4']:>7.1f}% | "
            f"{row['LLaMA-8B_%≥4']:>7.1f}% | "
            f"{row['Mistral-24B_%≥4']:>7.1f}%"
        )

    # Kruskal–Wallis 
    lines.append("\n\n3. KRUSKAL–WALLIS H TEST (4-group omnibus, α=0.05)")
    lines.append("-" * 80)
    lines.append(f"{'Question':<20} | {'H statistic':>12} | {'p-value':>12} | {'Significant':>12}")
    lines.append("-" * 80)
    for _, row in kw_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "n.s."))
        lines.append(
            f"{row['Question_Name']:<20} | "
            f"{row['H_statistic']:>12.4f} | "
            f"{row['p_value']:>12.6f} | "
            f"{sig:>12}"
        )

    # Wilcoxon pairwise 
    lines.append("\n\n4. PAIRWISE WILCOXON RANK-SUM (MANN–WHITNEY U) TESTS")
    lines.append(f"   Bonferroni correction: α' = {ALPHA}/{N_PAIRS} = {BONFERRONI_ALPHA:.4f}")
    lines.append("-" * 80)

    for qi, col in enumerate(QUESTION_COLS):
        sub = pw_df[pw_df["Question"] == Q_SHORT[qi]]
        lines.append(f"\n  {Q_SHORT[qi]} – {Q_NAMES[qi]}")
        lines.append(f"  {'Comparison':<30} | {'U':>8} | {'p-value':>10} | {'p(Bonf)':>10} | {'Sig':>6}")
        for _, row in sub.iterrows():
            pair = f"{row['Group_1']} vs {row['Group_2']}"
            sig = "*" if row["p_bonferroni"] < ALPHA else "n.s."
            lines.append(
                f"  {pair:<30} | "
                f"{row['U_statistic']:>8.1f} | "
                f"{row['p_value']:>10.6f} | "
                f"{row['p_bonferroni']:>10.6f} | "
                f"{sig:>6}"
            )


    with open(path, "w") as f:
        f.write("\n".join(lines))


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("=" * 70)
    print("  JudgeAgent – Statistical Analysis")
    print("  Favorable proportions + Kruskal–Wallis + Wilcoxon pairwise")
    print("=" * 70)

    # Load data
    groups   = load_data()
    raw_runs = load_raw_runs()

    # 1. Favorable proportions
    fav_df = compute_favorable_proportions(groups, raw_runs)
    fav_path = OUT_DIR / "favorable_proportions.csv"
    fav_df.to_csv(fav_path, index=False)
    print(f"\n✓ Favorable proportions → {fav_path}")

    # 2. Kruskal–Wallis
    kw_df = kruskal_wallis_tests(groups)
    kw_path = OUT_DIR / "kruskal_wallis_results.csv"
    kw_df.to_csv(kw_path, index=False)
    print(f"✓ Kruskal–Wallis results → {kw_path}")

    # 3. Wilcoxon pairwise
    pw_df = wilcoxon_pairwise(groups)
    pw_path = OUT_DIR / "wilcoxon_pairwise.csv"
    pw_df.to_csv(pw_path, index=False)
    print(f"✓ Wilcoxon pairwise     → {pw_path}")

    # 4. Human-readable report
    report_path = OUT_DIR / "statistical_tests_report.txt"
    write_report(fav_df, kw_df, pw_df, groups, report_path)
    print(f"✓ Full report           → {report_path}")

    # Print summary to console
    print("\n" + "─" * 70)
    print("  FAVORABLE PROPORTIONS (% ≥ 4, per-persona average, n=65)")
    print("─" * 70)
    for _, row in fav_df.iterrows():
        print(f"  {row['Question_Name']:<18}  CROWD={row['CROWD_%≥4']:5.1f}%  "
              f"GPT-5={row['GPT-5_%≥4']:5.1f}%  "
              f"LLaMA-8B={row['LLaMA-8B_%≥4']:5.1f}%  "
              f"Mistral-24B={row['Mistral-24B_%≥4']:5.1f}%")

    print("\n" + "─" * 70)
    print("  KRUSKAL–WALLIS (α=0.05)")
    print("─" * 70)
    for _, row in kw_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "n.s."))
        print(f"  {row['Question_Name']:<18}  H={row['H_statistic']:8.4f}  "
              f"p={row['p_value']:.6f}  {sig}")

    print("\n" + "─" * 70)
    print(f"  SIGNIFICANT PAIRWISE DIFFERENCES (Bonferroni α'={BONFERRONI_ALPHA:.4f})")
    print("─" * 70)
    sig_pairs = pw_df[pw_df["p_bonferroni"] < ALPHA]
    if sig_pairs.empty:
        print("  No pairwise comparison reached significance after Bonferroni correction.")
    else:
        for _, row in sig_pairs.iterrows():
            print(f"  {row['Question']}: {row['Group_1']} vs {row['Group_2']}  "
                  f"U={row['U_statistic']:.1f}  p={row['p_value']:.6f}  "
                  f"p(Bonf)={row['p_bonferroni']:.6f}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
