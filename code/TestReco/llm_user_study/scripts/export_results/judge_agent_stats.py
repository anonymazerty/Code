import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from itertools import combinations

# Paths 
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # real-user-study/TestReco
CSV_DIR  = BASE_DIR / "llm_user_study" / "results" / "JudgeAgent" / "results_as_csv"
OUT_DIR  = BASE_DIR / "llm_user_study" / "results" / "JudgeAgent" / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Survey question columns 
QUESTION_COLS = [
    "accomplishment",
    "effort_required",
    "mental_demand",
    "perceived_controllability",
    "temporal_demand",
    "trust",
]
Q_SHORT = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
Q_NAMES = [
    "Accomplishment",
    "Effort Required",
    "Mental Demand",
    "Controllability",
    "Temporal Demand",
    "Trust",
]

# Model directories and display labels 
MODELS = ["gpt", "llama", "mistral"]
MODEL_LABELS = {
    "gpt":    "GPT-5",
    "llama":  "LLaMA-8B",
    "mistral":"Mistral-24B",
}
GROUP_NAMES = ["CROWD", "GPT-5", "LLaMA-8B", "Mistral-24B"]

# Multiple-comparison parameters 
ALPHA       = 0.05
N_PAIRS     = 6                          # C(4,2) = 6 pairwise comparisons
BONF_ALPHA  = ALPHA / N_PAIRS            # ≈ 0.00833


# 
# DATA LOADING
# 

def load_groups():
    """
    Return {group_label: {q_col: np.ndarray}} with n=55 observations each.

    CROWD   → one row per real participant (directly from CSV).
    LLM     → 5 run CSVs concatenated, grouped by session_id, MEAN taken.
              Produces one row per persona (n=55), matching CROWD size.
    """
    # CROWD 
    crowd_df = pd.read_csv(CSV_DIR / "crowd_survey.csv")
    groups = {
        "CROWD": {col: crowd_df[col].values.astype(float) for col in QUESTION_COLS}
    }

    # LLM agents 
    for m in MODELS:
        model_dir = CSV_DIR / m
        frames = [pd.read_csv(model_dir / f"run{r}_survey.csv") for r in range(1, 6)]
        all_runs = pd.concat(frames, ignore_index=True)

        # Average the 5 runs per persona → one independent observation per persona
        per_persona = (
            all_runs.groupby("session_id")[QUESTION_COLS]
            .mean()
            .reset_index()
        )

        label = MODEL_LABELS[m]
        groups[label] = {
            col: per_persona[col].values.astype(float) for col in QUESTION_COLS
        }
        print(f"  {label}: {len(per_persona)} personas after averaging 5 runs")

    return groups


# 
# 1. PROPORTION OF FAVORABLE RESPONSES  (rating ≥ 4)
# 

def compute_favorable(groups):
    """
    For each question × group, count how many per-persona averages are ≥ 4
    and report the proportion (%).

    NOTE: A continuous per-persona average can be non-integer (e.g. 3.8, 4.2).
    We apply the same threshold (≥ 4.0) to the averaged value, which is the
    standard practice when per-persona averages are used as the unit of analysis.
    """
    rows = []
    for qi, col in enumerate(QUESTION_COLS):
        row = {"Question": Q_SHORT[qi], "Question_Name": Q_NAMES[qi]}
        for g in GROUP_NAMES:
            vals  = groups[g][col]
            n_fav = int(np.sum(vals >= 4.0))
            n     = len(vals)
            row[f"{g}_n>=4"]  = n_fav
            row[f"{g}_N"]     = n
            row[f"{g}_%>=4"]  = round(100 * n_fav / n, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# 2. KRUSKAL–WALLIS  (omnibus 4-group test)

def kruskal_wallis(groups):
    """
    Kruskal–Wallis H test across the 4 groups for each question.

    Rationale: Kruskal–Wallis is the non-parametric analogue of one-way ANOVA.
    It ranks all observations together and tests whether the rank distributions
    differ across groups.  Appropriate for ordinal Likert data and does not
    assume normality.  A significant result means ≥ 1 group differs from the
    others, but does not say which one(s).
    """
    rows = []
    for qi, col in enumerate(QUESTION_COLS):
        samples = [groups[g][col] for g in GROUP_NAMES]
        H, p = stats.kruskal(*samples)
        rows.append({
            "Question":      Q_SHORT[qi],
            "Question_Name": Q_NAMES[qi],
            "H_statistic":   round(H, 4),
            "p_value":       p,
            "Significant":   "Yes" if p < ALPHA else "No",
        })
    return pd.DataFrame(rows)


# 3. PAIRWISE WILCOXON RANK-SUM (= MANN–WHITNEY U)

def wilcoxon_pairwise(groups):
    """
    All 6 pairwise Mann–Whitney U tests per question, with Bonferroni correction.

    Rationale: When Kruskal–Wallis is significant, we need post-hoc tests to
    identify *which* pairs differ.  Mann–Whitney U is the two-sample equivalent
    of Kruskal–Wallis.  Bonferroni multiplies each raw p-value by the number of
    comparisons (6) to control the family-wise error rate.  Only pairs with
    p_bonferroni < 0.05 are declared significant.
    """
    pairs = list(combinations(GROUP_NAMES, 2))
    rows  = []
    for qi, col in enumerate(QUESTION_COLS):
        for g1, g2 in pairs:
            U, p = stats.mannwhitneyu(
                groups[g1][col], groups[g2][col], alternative="two-sided"
            )
            p_bonf = min(p * N_PAIRS, 1.0)
            rows.append({
                "Question":      Q_SHORT[qi],
                "Question_Name": Q_NAMES[qi],
                "Group_1":       g1,
                "Group_2":       g2,
                "U_statistic":   round(U, 1),
                "p_value":       p,
                "p_bonferroni":  p_bonf,
                "Significant_bonf": "Yes" if p_bonf < ALPHA else "No",
            })
    return pd.DataFrame(rows)


# REPORT
def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def write_report(fav_df, kw_df, pw_df, groups, path):
    lines = []
    W = 80

    lines += [
        "=" * W,
        "  JudgeAgent – Statistical Analysis Report",
        "  Proportion of Favorable Responses | Kruskal–Wallis | Wilcoxon Pairwise",
        "  Method: per-persona AVERAGE across 5 runs (n=55 per group)",
        "=" * W,
    ]

    # 1. Descriptive 
    lines += [
        "\n1. DESCRIPTIVE STATISTICS  (median [Q1–Q3], n=55 per group)",
        "-" * W,
        f"{'Question':<22} {'CROWD':>13} {'GPT-5':>13} {'LLaMA-8B':>13} {'Mistral-24B':>13}",
        "-" * W,
    ]
    for qi, col in enumerate(QUESTION_COLS):
        def fmt(g):
            v = groups[g][col]
            return f"{np.median(v):.2f} [{np.percentile(v,25):.1f}–{np.percentile(v,75):.1f}]"
        lines.append(
            f"{Q_NAMES[qi]:<22} {fmt('CROWD'):>13} {fmt('GPT-5'):>13} "
            f"{fmt('LLaMA-8B'):>13} {fmt('Mistral-24B'):>13}"
        )

    # 2. Favorable proportions 
    lines += [
        "\n\n2. PROPORTION OF FAVORABLE RESPONSES  (% with average ≥ 4, n=55 each)",
        "-" * W,
        f"{'Question':<22} {'CROWD':>9} {'GPT-5':>9} {'LLaMA-8B':>9} {'Mistral-24B':>11}",
        "-" * W,
    ]
    for _, row in fav_df.iterrows():
        lines.append(
            f"{row['Question_Name']:<22} "
            f"{row['CROWD_%>=4']:>8.1f}% "
            f"{row['GPT-5_%>=4']:>8.1f}% "
            f"{row['LLaMA-8B_%>=4']:>8.1f}% "
            f"{row['Mistral-24B_%>=4']:>10.1f}%"
        )

    # 3. Kruskal–Wallis 
    lines += [
        "\n\n3. KRUSKAL–WALLIS H TEST  (omnibus, α=0.05)",
        "-" * W,
        f"{'Question':<22} {'H':>10} {'p-value':>12} {'Sig':>6}",
        "-" * W,
    ]
    for _, row in kw_df.iterrows():
        sig = _sig_stars(row["p_value"])
        lines.append(
            f"{row['Question_Name']:<22} {row['H_statistic']:>10.4f} "
            f"{row['p_value']:>12.6f} {sig:>6}"
        )

    # 4. Pairwise Wilcoxon 
    lines += [
        f"\n\n4. PAIRWISE WILCOXON RANK-SUM (Mann–Whitney U)",
        f"   Bonferroni correction: α' = {ALPHA}/{N_PAIRS} = {BONF_ALPHA:.4f}",
        "-" * W,
    ]
    for qi, col in enumerate(QUESTION_COLS):
        sub = pw_df[pw_df["Question"] == Q_SHORT[qi]]
        lines += [f"\n  {Q_SHORT[qi]} – {Q_NAMES[qi]}",
                  f"  {'Comparison':<32} {'U':>8} {'p-raw':>10} {'p-Bonf':>10} {'Sig':>6}"]
        for _, row in sub.iterrows():
            pair = f"{row['Group_1']} vs {row['Group_2']}"
            sig  = "*" if row["p_bonferroni"] < ALPHA else "n.s."
            lines.append(
                f"  {pair:<32} {row['U_statistic']:>8.1f} "
                f"{row['p_value']:>10.6f} {row['p_bonferroni']:>10.6f} {sig:>6}"
            )

    # Interpretation 
    lines += [
        "\n\n" + "=" * W,
        "  INTERPRETATION GUIDE",
        "=" * W,
        """
Significance codes:  *** p<0.001   ** p<0.01   * p<0.05 (Bonf-corrected)   n.s. not significant

FAVORABLE RESPONSE (≥ 4):
  A rating of 4 or 5 on the 1–5 scale is considered favorable.  For LLM agents,
  we first average the 5 runs per persona, then check whether that average ≥ 4.
  This makes it easy to say "X% of CROWD vs Y% of GPT-5 found this favorable."

KRUSKAL–WALLIS:
  Non-parametric analogue of one-way ANOVA.  Ranks all 4×55 = 220 observations
  together and tests whether rank sums differ across groups.  No normality
  assumption – appropriate for Likert data.  Significant result → at least one
  group is distributed differently, but does not identify which pair(s).

WILCOXON RANK-SUM (= MANN–WHITNEY U):
  Two-sample rank-sum test applied to every pair of groups.  Bonferroni correction
  multiplies each raw p-value by 6 (number of pairs) to guard against inflated
  false-positive rate from multiple testing.  Significant → the two groups'
  distributions are unlikely to be the same.

WHY AVERAGE (NOT MEDIAN) ACROSS RUNS?
  Five runs per persona give us repeated measurements of the same latent
  "agent personality."  Averaging collapses these to one number per persona,
  eliminating pseudo-replication.  The MEAN is preferred over the median here
  because:
    (a) it preserves the full numerical gradient of the 1–5 Likert responses,
    (b) it results in continuous-valued observations that spread more evenly
        across the scale (fewer ties), which improves the power of rank-sum tests,
    (c) it is the standard when the goal is to estimate the expected value of the
        agent's response distribution for a given persona.
  Using the mean vs. the median typically makes little difference when n=5, but
  can slightly shift proportions and test statistics for skewed response patterns.
""",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


# 
# MAIN
# 

def main():
    sep = "" * 70
    print("=" * 70)
    print("  JudgeAgent – Statistical Analysis (per-persona AVERAGE, n=55)")
    print("=" * 70)

    print("\nLoading data …")
    groups = load_groups()
    n_crowd = len(groups["CROWD"]["accomplishment"])
    print(f"  CROWD: {n_crowd} participants")

    # 1. Favorable proportions
    fav_df   = compute_favorable(groups)
    fav_path = OUT_DIR / "favorable_proportions.csv"
    fav_df.to_csv(fav_path, index=False)
    print(f"\n✓ Favorable proportions → {fav_path}")

    # 2. Kruskal–Wallis
    kw_df   = kruskal_wallis(groups)
    kw_path = OUT_DIR / "kruskal_wallis_results.csv"
    kw_df.to_csv(kw_path, index=False)
    print(f"✓ Kruskal–Wallis        → {kw_path}")

    # 3. Wilcoxon pairwise
    pw_df   = wilcoxon_pairwise(groups)
    pw_path = OUT_DIR / "wilcoxon_pairwise.csv"
    pw_df.to_csv(pw_path, index=False)
    print(f"✓ Wilcoxon pairwise     → {pw_path}")

    # 4. Full report
    report_path = OUT_DIR / "statistical_tests_report.txt"
    write_report(fav_df, kw_df, pw_df, groups, report_path)
    print(f"✓ Full report           → {report_path}")

    # Console summary 
    print(f"\n{sep}")
    print("  FAVORABLE PROPORTIONS  (% per-persona average ≥ 4, n=55 per group)")
    print(sep)
    print(f"  {'Question':<22} {'CROWD':>7} {'GPT-5':>7} {'LLaMA-8B':>9} {'Mistral-24B':>12}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*9} {'-'*12}")
    for _, row in fav_df.iterrows():
        print(
            f"  {row['Question_Name']:<22} "
            f"{row['CROWD_%>=4']:>6.1f}% "
            f"{row['GPT-5_%>=4']:>6.1f}% "
            f"{row['LLaMA-8B_%>=4']:>8.1f}% "
            f"{row['Mistral-24B_%>=4']:>11.1f}%"
        )

    print(f"\n{sep}")
    print("  KRUSKAL–WALLIS  (α = 0.05)")
    print(sep)
    for _, row in kw_df.iterrows():
        sig = _sig_stars(row["p_value"])
        print(
            f"  {row['Question_Name']:<22}  H={row['H_statistic']:8.4f}  "
            f"p={row['p_value']:.6f}  {sig}"
        )

    print(f"\n{sep}")
    print(f"  SIGNIFICANT PAIRWISE DIFFERENCES  (Bonferroni α'={BONF_ALPHA:.4f})")
    print(sep)
    sig_rows = pw_df[pw_df["p_bonferroni"] < ALPHA]
    if sig_rows.empty:
        print("  No pairwise comparison reached significance after Bonferroni correction.")
    else:
        for _, row in sig_rows.iterrows():
            print(
                f"  {row['Question']} {row['Group_1']} vs {row['Group_2']:<14}  "
                f"U={row['U_statistic']:.1f}  "
                f"p={row['p_value']:.6f}  p(Bonf)={row['p_bonferroni']:.6f}"
            )

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
