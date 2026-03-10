import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
from pathlib import Path
from itertools import combinations

# Paths
BASE_DIR  = Path(__file__).resolve().parent.parent          # quizcomp_llm_study/
DEMO_CSV  = BASE_DIR.parent / "demographics.csv"            # datagems/demographics.csv
RESULTS   = BASE_DIR / "Results"

# JudgeAgent persona DB (same persona mapping for all 3 LLMs)
JUDGE_DB  = BASE_DIR / "AgentJUDGEresults" / "llm_results_GPT5.db"

# e2eAgent persona DBs (one per prompt folder, same mapping across LLMs)
E2E_DBS = {
    "Calibrated":       BASE_DIR / "Calibrated results"       / "llm_results_gpt_detailed.db",
    "Role-Anchored":    BASE_DIR / "Role-Anchored results"    / "llm_results_gpt_inbetween.db",
    "Task-Constrained": BASE_DIR / "Task-Constrained results" / "llm_results_gpt_general.db",
}

QUESTION_COLS = [
    "Q1_Accomplishment", "Q2_Effort", "Q3_Mental_Demand",
    "Q4_Controllability", "Q5_Temporal_Demand", "Q6_Satisfaction",
]
Q_SHORT = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
Q_NAMES = [
    "Accomplishment", "Effort", "Mental Demand",
    "Controllability", "Temporal Demand", "Satisfaction",
]

MODELS = ["GPT5", "llama8B", "Mistral"]
MODEL_LABELS = {"GPT5": "GPT-5", "llama8B": "LLaMA-8B", "Mistral": "Mistral-24B"}
GROUP_NAMES = ["CROWD", "GPT-5", "LLaMA-8B", "Mistral-24B"]
ROLES = ["College Teacher", "High School Teacher"]

ALPHA = 0.05
N_PAIRS = 6
BONFERRONI_ALPHA = ALPHA / N_PAIRS


# ====================================================================
# 1.  BUILD MAPPINGS
# ====================================================================

def load_demographics():
    """Load prolific_pid → Current job role mapping."""
    demo = pd.read_csv(DEMO_CSV, usecols=["prolific_pid", "Current job role"])
    return dict(zip(demo["prolific_pid"], demo["Current job role"]))


def load_persona_mapping(db_path):
    """Load persona_id → prolific_pid from teacher_personas table."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT persona_id, prolific_pid FROM teacher_personas", conn)
    conn.close()
    return dict(zip(df["persona_id"], df["prolific_pid"]))


# ====================================================================
# 2.  INSERT JOB ROLE INTO CSVs
# ====================================================================

def add_role_to_crowd_csv(csv_path, pid_to_role):
    """Add Current_Job_Role to a crowd_survey.csv (has prolific_pid)."""
    df = pd.read_csv(csv_path)
    if "Current_Job_Role" in df.columns:
        df.drop(columns=["Current_Job_Role"], inplace=True)
    df["Current_Job_Role"] = df["prolific_pid"].map(pid_to_role).fillna("Unknown")
    df.to_csv(csv_path, index=False)
    return df


def add_role_to_llm_csv(csv_path, persona_to_pid, pid_to_role):
    """Add Current_Job_Role to an LLM CSV (has persona_id)."""
    df = pd.read_csv(csv_path)
    if "Current_Job_Role" in df.columns:
        df.drop(columns=["Current_Job_Role"], inplace=True)
    df["Current_Job_Role"] = (
        df["persona_id"]
        .map(persona_to_pid)
        .map(pid_to_role)
        .fillna("Unknown")
    )
    df.to_csv(csv_path, index=False)
    return df


def inject_roles_for_config(csv_dir, persona_to_pid, pid_to_role, label):
    """Inject Current_Job_Role into all CSVs under one config directory."""
    # Crowd CSV
    crowd_csv = csv_dir / "crowd_survey.csv"
    if crowd_csv.exists():
        add_role_to_crowd_csv(crowd_csv, pid_to_role)
        print(f"  ✓ {label} / crowd_survey.csv")

    # LLM CSVs
    for model in MODELS:
        model_dir = csv_dir / model
        if not model_dir.exists():
            continue
        for csv_file in sorted(model_dir.glob("*.csv")):
            add_role_to_llm_csv(csv_file, persona_to_pid, pid_to_role)
            print(f"  ✓ {label} / {model} / {csv_file.name}")


# ====================================================================
# 3.  PER-ROLE STATISTICAL ANALYSIS
# ====================================================================

def load_groups_by_role(csv_dir, persona_to_pid, pid_to_role, role):
    """Load data filtered to a specific role, using per-persona averages for LLMs."""
    groups = {}

    # CROWD
    crowd_df = pd.read_csv(csv_dir / "crowd_survey.csv")
    crowd_df["role"] = crowd_df["prolific_pid"].map(pid_to_role).fillna("Unknown")
    crowd_role = crowd_df[crowd_df["role"] == role]
    groups["CROWD"] = {c: crowd_role[c].values.astype(float) for c in QUESTION_COLS}

    # LLMs (average_survey.csv)
    for m in MODELS:
        avg_path = csv_dir / m / "average_survey.csv"
        if not avg_path.exists():
            continue
        avg_df = pd.read_csv(avg_path)
        avg_df["pid"] = avg_df["persona_id"].map(persona_to_pid)
        avg_df["role"] = avg_df["pid"].map(pid_to_role).fillna("Unknown")
        avg_role = avg_df[avg_df["role"] == role]
        label = MODEL_LABELS[m]
        groups[label] = {c: avg_role[c].values.astype(float) for c in QUESTION_COLS}

    return groups


def compute_stats_for_role(groups, role):
    """Compute favorable proportions, Kruskal-Wallis, Wilcoxon for one role."""
    available_groups = [g for g in GROUP_NAMES if g in groups and len(groups[g][QUESTION_COLS[0]]) > 0]

    # Sample sizes
    n_per_group = {g: len(groups[g][QUESTION_COLS[0]]) for g in available_groups}

    # ── Descriptive + Favorable ──
    fav_rows = []
    for qi, col in enumerate(QUESTION_COLS):
        row = {"Question": Q_SHORT[qi], "Question_Name": Q_NAMES[qi]}
        for g in available_groups:
            vals = groups[g][col]
            row[f"{g}_n"] = len(vals)
            row[f"{g}_median"] = np.median(vals)
            row[f"{g}_IQR"] = f"{np.percentile(vals,25):.1f}-{np.percentile(vals,75):.1f}"
            n_fav = np.sum(vals >= 4)
            row[f"{g}_%≥4"] = round(100 * n_fav / len(vals), 1) if len(vals) > 0 else 0
        fav_rows.append(row)
    fav_df = pd.DataFrame(fav_rows)

    # ── Kruskal-Wallis ──
    kw_rows = []
    for qi, col in enumerate(QUESTION_COLS):
        samples = [groups[g][col] for g in available_groups if len(groups[g][col]) > 0]
        if len(samples) >= 2:
            H, p = stats.kruskal(*samples)
        else:
            H, p = np.nan, np.nan
        kw_rows.append({
            "Question": Q_SHORT[qi], "Question_Name": Q_NAMES[qi],
            "H_statistic": round(H, 4) if not np.isnan(H) else np.nan,
            "p_value": p,
            "Significant": "Yes" if p < ALPHA else "No",
        })
    kw_df = pd.DataFrame(kw_rows)

    # ── Wilcoxon pairwise ──
    pairs = list(combinations(available_groups, 2))
    n_pairs = len(pairs)
    pw_rows = []
    for qi, col in enumerate(QUESTION_COLS):
        for g1, g2 in pairs:
            v1, v2 = groups[g1][col], groups[g2][col]
            if len(v1) > 0 and len(v2) > 0:
                U, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                pb = min(p * n_pairs, 1.0)
            else:
                U, p, pb = np.nan, np.nan, np.nan
            pw_rows.append({
                "Question": Q_SHORT[qi], "Question_Name": Q_NAMES[qi],
                "Group_1": g1, "Group_2": g2,
                "U_statistic": round(U, 1) if not np.isnan(U) else np.nan,
                "p_value": p,
                "p_bonferroni": pb,
                "Significant": "Yes" if pb < ALPHA else "No",
            })
    pw_df = pd.DataFrame(pw_rows)

    return fav_df, kw_df, pw_df, n_per_group


def write_role_report(role, fav_df, kw_df, pw_df, n_per_group, path):
    """Write a human-readable report for one role."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"  Statistical Analysis – {role}")
    lines.append(f"  (per-persona averages across 5 runs)")
    lines.append("=" * 80)

    # Sample sizes
    lines.append(f"\nSample sizes: " + ", ".join(f"{g}={n}" for g, n in n_per_group.items()))

    available_groups = list(n_per_group.keys())

    # Favorable proportions
    lines.append("\n\n1. PROPORTION OF FAVORABLE RESPONSES (% rating ≥ 4)")
    lines.append("-" * 80)
    header_parts = [f"{'Question':<18}"]
    for g in available_groups:
        header_parts.append(f"{g:>10}")
    lines.append(" | ".join(header_parts))
    lines.append("-" * 80)
    for _, row in fav_df.iterrows():
        parts = [f"{row['Question_Name']:<18}"]
        for g in available_groups:
            parts.append(f"{row[f'{g}_%≥4']:>9.1f}%")
        lines.append(" | ".join(parts))

    # Kruskal-Wallis
    lines.append("\n\n2. KRUSKAL–WALLIS H TEST (4-group omnibus, α=0.05)")
    lines.append("-" * 80)
    lines.append(f"{'Question':<18} | {'H statistic':>12} | {'p-value':>12} | {'Significant':>12}")
    lines.append("-" * 80)
    for _, row in kw_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "n.s."))
        lines.append(f"{row['Question_Name']:<18} | {row['H_statistic']:>12.4f} | {row['p_value']:>12.6f} | {sig:>12}")

    # Pairwise
    pairs = list(combinations(available_groups, 2))
    n_pairs = len(pairs)
    bonf = ALPHA / n_pairs if n_pairs > 0 else ALPHA
    lines.append(f"\n\n3. PAIRWISE WILCOXON RANK-SUM (MANN–WHITNEY U)")
    lines.append(f"   Bonferroni correction: α' = {ALPHA}/{n_pairs} = {bonf:.4f}")
    lines.append("-" * 80)

    for qi in range(6):
        sub = pw_df[pw_df["Question"] == Q_SHORT[qi]]
        lines.append(f"\n  {Q_SHORT[qi]} – {Q_NAMES[qi]}")
        lines.append(f"  {'Comparison':<30} | {'U':>8} | {'p-value':>10} | {'p(Bonf)':>10} | {'Sig':>6}")
        for _, row in sub.iterrows():
            pair = f"{row['Group_1']} vs {row['Group_2']}"
            sig = "*" if row["p_bonferroni"] < ALPHA else "n.s."
            lines.append(f"  {pair:<30} | {row['U_statistic']:>8.1f} | {row['p_value']:>10.6f} | {row['p_bonferroni']:>10.6f} | {sig:>6}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def analyse_config(config_label, csv_dir, persona_to_pid, pid_to_role, out_dir):
    """Run per-role analysis for one configuration (JudgeAgent or an e2eAgent prompt)."""
    role_dir = out_dir / "by_role"
    role_dir.mkdir(parents=True, exist_ok=True)

    for role in ROLES:
        safe_role = role.replace(" ", "_")
        groups = load_groups_by_role(csv_dir, persona_to_pid, pid_to_role, role)
        fav_df, kw_df, pw_df, n_per_group = compute_stats_for_role(groups, role)

        fav_df.to_csv(role_dir / f"favorable_{safe_role}.csv", index=False)
        kw_df.to_csv(role_dir / f"kruskal_wallis_{safe_role}.csv", index=False)
        pw_df.to_csv(role_dir / f"wilcoxon_pairwise_{safe_role}.csv", index=False)
        write_role_report(role, fav_df, kw_df, pw_df, n_per_group,
                          role_dir / f"report_{safe_role}.txt")

        print(f"    {role}: n=" + ", ".join(f"{g}={n}" for g, n in n_per_group.items()))


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("=" * 70)
    print("  Add Current_Job_Role & Per-Role Statistical Analysis")
    print("=" * 70)

    pid_to_role = load_demographics()

    # ── STEP 1: Inject roles into all CSVs ────────────────────────
    print("\n[1] Injecting Current_Job_Role into CSVs...")

    # JudgeAgent
    ja_persona_map = load_persona_mapping(JUDGE_DB)
    ja_csv_dir = RESULTS / "JudgeAgent" / "Results_as_csvs"
    inject_roles_for_config(ja_csv_dir, ja_persona_map, pid_to_role, "JudgeAgent")

    # e2eAgent
    for prompt, db_path in E2E_DBS.items():
        e2e_persona_map = load_persona_mapping(db_path)
        e2e_csv_dir = RESULTS / "e2eAgent" / "Results_as_csvs" / prompt
        inject_roles_for_config(e2e_csv_dir, e2e_persona_map, pid_to_role, f"e2eAgent/{prompt}")

    # ── STEP 2: Per-role statistics ───────────────────────────────
    print("\n[2] Computing per-role statistics...")

    # JudgeAgent
    print("\n  JudgeAgent:")
    analyse_config("JudgeAgent", ja_csv_dir, ja_persona_map, pid_to_role,
                    RESULTS / "JudgeAgent")

    # e2eAgent
    for prompt, db_path in E2E_DBS.items():
        e2e_persona_map = load_persona_mapping(db_path)
        e2e_csv_dir = RESULTS / "e2eAgent" / "Results_as_csvs" / prompt
        print(f"\n  e2eAgent – {prompt}:")
        analyse_config(f"e2eAgent-{prompt}", e2e_csv_dir, e2e_persona_map,
                        pid_to_role, RESULTS / "e2eAgent" / prompt)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
