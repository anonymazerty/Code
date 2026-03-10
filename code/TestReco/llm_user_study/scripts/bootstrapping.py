import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from itertools import product


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_ROOT = os.path.join(_SCRIPT_DIR, "..", "llm_user_study", "results")

# E2E Agent results root (contains subfolders: task-constrained/, role-anchored/, calibrated/)
E2E_DIR = os.path.join(_RESULTS_ROOT, "e2eAgent", "results_as_csv")

# JudgeAgent results root (contains subfolders: llama/, gpt/, mistral/, crowd_survey.csv, stats_summary.csv)
JUDGE_DIR = os.path.join(_RESULTS_ROOT, "JudgeAgent", "results_as_csv")

# Output directory for tables
OUTPUT_DIR = os.path.join(_RESULTS_ROOT, "alignment_results")

# Likert metrics (column names in CSV files)
METRICS = [
    "accomplishment",
    "effort_required",
    "mental_demand",
    "perceived_controllability",
    "temporal_demand",
    "trust",
]

# Display labels for the metrics
METRIC_LABELS = {
    "accomplishment": "Q1: Accomplishment",
    "effort_required": "Q2: Effort Required",
    "mental_demand": "Q3: Mental Demand",
    "perceived_controllability": "Q4: Controllability",
    "temporal_demand": "Q5: Temporal Demand",
    "trust": "Q6: Trust",
}

# Model keys (subfolder names) and display names
MODELS = {
    "llama": "LLaMA-8B",
    "gpt": "GPT-5",
    "mistral": "Mistral-24B",
}

# E2E prompt strategies (subfolder names) and display names
E2E_PROMPTS = {
    "task-constrained": "Task-Constrained",
    "role-anchored": "Role-Anchored",
    "calibrated": "Calibrated",
}

# Number of runs per agent configuration
NUM_RUNS = 5

# Bootstrap parameters
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
CI_LOWER = 2.5   # percentile for lower bound
CI_UPPER = 97.5  # percentile for upper bound

# Education levels for per-profile analysis
EDUCATION_LEVELS = ["high_school", "undergraduate", "graduate"]


def load_crowd_survey(base_dir: str) -> pd.DataFrame:
    """Load crowd survey CSV from a results directory."""
    path = os.path.join(base_dir, "crowd_survey.csv")
    df = pd.read_csv(path)
    return df


def load_agent_run(base_dir: str, model_key: str, run: int) -> pd.DataFrame:
    """Load a single agent run CSV. Returns empty DataFrame if file missing or empty."""
    path = os.path.join(base_dir, model_key, f"run{run}_survey.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_agent_median(base_dir: str, model_key: str) -> pd.DataFrame:
    """Load the median-aggregated agent survey CSV."""
    path = os.path.join(base_dir, model_key, "med_survey.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def aggregate_runs(base_dir: str, model_key: str, num_runs: int = NUM_RUNS) -> pd.DataFrame:
    """
    Aggregate multiple runs by computing per-participant median across runs.
    Falls back to med_survey.csv if individual runs are unavailable.
    """
    run_dfs = []
    for r in range(1, num_runs + 1):
        df = load_agent_run(base_dir, model_key, r)
        if not df.empty:
            run_dfs.append(df)


    # Stack all runs and compute per-participant median for each metric
    # Identify the participant key: use session_id (or real_session_id if present)
    if "real_session_id" in run_dfs[0].columns:
        id_col = "real_session_id"
    else:
        id_col = "session_id"

    # Add run index
    for i, df in enumerate(run_dfs):
        df["_run"] = i + 1

    combined = pd.concat(run_dfs, ignore_index=True)

    # Group by participant and compute median per metric
    agg_dict = {m: "median" for m in METRICS if m in combined.columns}
    if "education_level" in combined.columns:
        # Keep education_level (take first)
        grouped = combined.groupby(id_col).agg(
            {**agg_dict, "education_level": "first"}
        ).reset_index()
    else:
        grouped = combined.groupby(id_col).agg(agg_dict).reset_index()

    grouped.rename(columns={id_col: "session_id"}, inplace=True)
    return grouped


#  BOOTSTRAP CONFIDENCE INTERVALS
def bootstrap_ci(values: np.ndarray, stat_func, n_bootstrap=N_BOOTSTRAP,
                 ci_lower=CI_LOWER, ci_upper=CI_UPPER, seed=RANDOM_SEED):
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: 1D array of observations
        stat_func: function that takes an array and returns a scalar
        n_bootstrap: number of resamples
        ci_lower, ci_upper: percentile bounds

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    point = stat_func(values)

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = stat_func(sample)

    ci_lo = np.percentile(boot_stats, ci_lower)
    ci_hi = np.percentile(boot_stats, ci_upper)
    return point, ci_lo, ci_hi


def compute_iqr(values: np.ndarray) -> float:
    """Compute interquartile range."""
    return np.percentile(values, 75) - np.percentile(values, 25)


def compute_crowd_bootstrap(crowd_df: pd.DataFrame, metrics=METRICS,
                             education_level: str = None):
    """
    Compute bootstrap CIs for crowd median and IQR for each metric.

    Args:
        crowd_df: crowd survey DataFrame
        metrics: list of metric column names
        education_level: if provided, filter to this education level

    Returns:
        dict of {metric: {median, ci_med_lo, ci_med_hi, iqr, ci_iqr_lo, ci_iqr_hi}}
    """
    df = crowd_df.copy()
    if education_level is not None:
        df = df[df["education_level"] == education_level]

    results = {}
    for m in metrics:
        if m not in df.columns:
            continue
        vals = df[m].dropna().values
        if len(vals) == 0:
            continue

        med, med_lo, med_hi = bootstrap_ci(vals, np.median)
        iqr_val, iqr_lo, iqr_hi = bootstrap_ci(vals, compute_iqr)

        results[m] = {
            "n": len(vals),
            "median": med,
            "ci_med_lo": med_lo,
            "ci_med_hi": med_hi,
            "iqr": iqr_val,
            "ci_iqr_lo": iqr_lo,
            "ci_iqr_hi": iqr_hi,
        }

    return results


def compute_compatibility(agent_median: float, agent_iqr: float,
                           crowd_ci_med: tuple, crowd_ci_iqr: tuple):
    """
    Check if agent median and IQR fall within crowd bootstrap CIs.

    Returns:
        (median_compatible: bool, iqr_compatible: bool)
    """
    med_ok = crowd_ci_med[0] <= agent_median <= crowd_ci_med[1]
    iqr_ok = crowd_ci_iqr[0] <= agent_iqr <= crowd_ci_iqr[1]
    return med_ok, iqr_ok


def compute_alignment_score(agent_median: float, agent_iqr: float,
                             crowd_median: float, crowd_iqr: float):
    """Compute an alignment score based on distance from crowd median and IQR, normalized by crowd IQR."""
    delta_med = abs(agent_median - crowd_median)
    delta_iqr = abs(agent_iqr - crowd_iqr)

    if crowd_iqr == 0:
        # Fallback: use raw absolute deviation
        d = delta_med + delta_iqr
    else:
        d = delta_med / crowd_iqr + delta_iqr / crowd_iqr

    align_score = np.exp(-d)
    return d, align_score


def analyze_agent_alignment(crowd_bootstrap: dict, agent_df: pd.DataFrame,
                             metrics=METRICS, education_level: str = None):
    """
    Full alignment analysis for one agent configuration vs crowd.

    Args:
        crowd_bootstrap: output of compute_crowd_bootstrap()
        agent_df: agent survey DataFrame (aggregated across runs)
        metrics: list of metric names
        education_level: optional filter

    Returns:
        list of dicts with per-metric results
    """
    df = agent_df.copy()
    if education_level is not None and "education_level" in df.columns:
        df = df[df["education_level"] == education_level]

    results = []
    for m in metrics:
        if m not in crowd_bootstrap or m not in df.columns:
            continue

        cb = crowd_bootstrap[m]
        agent_vals = df[m].dropna().values
        if len(agent_vals) == 0:
            continue

        a_med = np.median(agent_vals)
        a_iqr = compute_iqr(agent_vals)

        # Delta
        delta_med = a_med - cb["median"]
        delta_iqr = a_iqr - cb["iqr"]

        # Compatibility
        med_ok, iqr_ok = compute_compatibility(
            a_med, a_iqr,
            (cb["ci_med_lo"], cb["ci_med_hi"]),
            (cb["ci_iqr_lo"], cb["ci_iqr_hi"]),
        )

        # Alignment score
        d, align = compute_alignment_score(a_med, a_iqr, cb["median"], cb["iqr"])

        results.append({
            "metric": m,
            "metric_label": METRIC_LABELS.get(m, m),
            "crowd_n": cb["n"],
            "crowd_median": cb["median"],
            "crowd_iqr": cb["iqr"],
            "ci_med": f"[{cb['ci_med_lo']:.1f}, {cb['ci_med_hi']:.1f}]",
            "ci_iqr": f"[{cb['ci_iqr_lo']:.1f}, {cb['ci_iqr_hi']:.1f}]",
            "agent_n": len(agent_vals),
            "agent_median": a_med,
            "agent_iqr": a_iqr,
            "delta_med": delta_med,
            "delta_iqr": delta_iqr,
            "med_compatible": med_ok,
            "iqr_compatible": iqr_ok,
            "d_score": d,
            "align_score": align,
        })

    return results


def wilcoxon_test(crowd_df: pd.DataFrame, agent_df: pd.DataFrame,
                  metrics=METRICS, education_level: str = None):
    """
    Pairwise Wilcoxon rank-sum (Mann-Whitney U) test between crowd and agent.

    Returns:
        list of dicts with per-metric test results
    """
    c = crowd_df.copy()
    a = agent_df.copy()
    if education_level is not None:
        if "education_level" in c.columns:
            c = c[c["education_level"] == education_level]
        if "education_level" in a.columns:
            a = a[a["education_level"] == education_level]

    results = []
    for m in metrics:
        if m not in c.columns or m not in a.columns:
            continue
        c_vals = c[m].dropna().values
        a_vals = a[m].dropna().values
        if len(c_vals) < 2 or len(a_vals) < 2:
            continue

        stat, pval = mannwhitneyu(c_vals, a_vals, alternative="two-sided")
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"

        results.append({
            "metric": m,
            "metric_label": METRIC_LABELS.get(m, m),
            "U_stat": stat,
            "p_value": pval,
            "significance": sig,
            "crowd_n": len(c_vals),
            "agent_n": len(a_vals),
        })

    return results



def run_e2e_analysis(e2e_dir: str, output_dir: str):
    """Run full analysis for all E2E Agent configurations."""
    all_results = []
    all_compat = []
    all_wilcoxon = []
    all_deltas = []

    for prompt_key, prompt_label in E2E_PROMPTS.items():
        prompt_dir = os.path.join(e2e_dir, prompt_key)
        if not os.path.isdir(prompt_dir):
            print(f"  [SKIP] {prompt_dir} not found")
            continue

        crowd_df = load_crowd_survey(prompt_dir)
        crowd_boot = compute_crowd_bootstrap(crowd_df)

        for model_key, model_label in MODELS.items():
            print(f"  Analyzing E2E [{prompt_label}] × {model_label}...")

            agent_df = aggregate_runs(prompt_dir, model_key)
            if agent_df.empty:
                print(f"    [SKIP] No data for {model_key}")
                continue

            model_results = []
            model_wilcoxon = []

            # --- Overall analysis ---
            alignment = analyze_agent_alignment(crowd_boot, agent_df)
            for row in alignment:
                row["mode"] = "E2E Agent"
                row["prompt"] = prompt_label
                row["model"] = model_label
                row["profile"] = "All"
            all_results.extend(alignment)
            model_results.extend(alignment)

            # Wilcoxon tests
            wt = wilcoxon_test(crowd_df, agent_df)
            for row in wt:
                row["mode"] = "E2E Agent"
                row["prompt"] = prompt_label
                row["model"] = model_label
                row["profile"] = "All"
            all_wilcoxon.extend(wt)
            model_wilcoxon.extend(wt)

            # --- Per education-level analysis ---
            for level in EDUCATION_LEVELS:
                crowd_boot_level = compute_crowd_bootstrap(crowd_df, education_level=level)
                if not crowd_boot_level:
                    continue
                alignment_level = analyze_agent_alignment(
                    crowd_boot_level, agent_df, education_level=level
                )
                for row in alignment_level:
                    row["mode"] = "E2E Agent"
                    row["prompt"] = prompt_label
                    row["model"] = model_label
                    row["profile"] = level
                all_results.extend(alignment_level)
                model_results.extend(alignment_level)

                wt_level = wilcoxon_test(crowd_df, agent_df, education_level=level)
                for row in wt_level:
                    row["mode"] = "E2E Agent"
                    row["prompt"] = prompt_label
                    row["model"] = model_label
                    row["profile"] = level
                all_wilcoxon.extend(wt_level)
                model_wilcoxon.extend(wt_level)

            # Save per-model results inside the prompt/model folder
            model_dir = os.path.join(prompt_dir, model_key)
            if model_results:
                pd.DataFrame(model_results).to_csv(
                    os.path.join(model_dir, "alignment_results.csv"), index=False
                )
                print(f"    Saved: {model_dir}/alignment_results.csv")
            if model_wilcoxon:
                pd.DataFrame(model_wilcoxon).to_csv(
                    os.path.join(model_dir, "wilcoxon_results.csv"), index=False
                )
                print(f"    Saved: {model_dir}/wilcoxon_results.csv")

    return all_results, all_wilcoxon


def run_judge_analysis(judge_dir: str, output_dir: str):
    """Run full analysis for JudgeAgent configurations."""
    all_results = []
    all_wilcoxon = []

    crowd_path = os.path.join(judge_dir, "crowd_survey.csv")
    if not os.path.exists(crowd_path):
        print(f"  [SKIP] No crowd_survey.csv in {judge_dir}")
        return all_results, all_wilcoxon

    crowd_df = pd.read_csv(crowd_path)
    crowd_boot = compute_crowd_bootstrap(crowd_df)

    for model_key, model_label in MODELS.items():
        print(f"  Analyzing JudgeAgent × {model_label}...")

        model_results = []
        model_wilcoxon = []
        agent_df = aggregate_runs(judge_dir, model_key)
        if agent_df.empty:
            # Try loading from stats_summary.csv as fallback
            stats_path = os.path.join(judge_dir, "stats_summary.csv")
            if os.path.exists(stats_path):
                print(f"    [INFO] No per-participant data; using stats_summary.csv for alignment scores only")
                stats = pd.read_csv(stats_path)
                # Can compute alignment from summary stats but not Wilcoxon
                for m in METRICS:
                    src = f"{model_key}_med"
                    row = stats[(stats["source"] == src) & (stats["metric"] == m)]
                    if len(row) == 0:
                        continue
                    cb = crowd_boot.get(m)
                    if cb is None:
                        continue

                    a_med = row["median"].values[0]
                    a_iqr = row["iqr"].values[0]
                    delta_med = a_med - cb["median"]
                    delta_iqr = a_iqr - cb["iqr"]

                    med_ok, iqr_ok = compute_compatibility(
                        a_med, a_iqr,
                        (cb["ci_med_lo"], cb["ci_med_hi"]),
                        (cb["ci_iqr_lo"], cb["ci_iqr_hi"]),
                    )
                    d, align = compute_alignment_score(a_med, a_iqr, cb["median"], cb["iqr"])

                    result_row = {
                        "metric": m,
                        "metric_label": METRIC_LABELS.get(m, m),
                        "crowd_n": cb["n"],
                        "crowd_median": cb["median"],
                        "crowd_iqr": cb["iqr"],
                        "ci_med": f"[{cb['ci_med_lo']:.1f}, {cb['ci_med_hi']:.1f}]",
                        "ci_iqr": f"[{cb['ci_iqr_lo']:.1f}, {cb['ci_iqr_hi']:.1f}]",
                        "agent_n": int(row["n"].values[0]),
                        "agent_median": a_med,
                        "agent_iqr": a_iqr,
                        "delta_med": delta_med,
                        "delta_iqr": delta_iqr,
                        "med_compatible": med_ok,
                        "iqr_compatible": iqr_ok,
                        "d_score": d,
                        "align_score": align,
                        "mode": "AgentJudge",
                        "prompt": "N/A",
                        "model": model_label,
                        "profile": "All",
                    }
                    all_results.append(result_row)
                    model_results.append(result_row)

                if model_results:
                    model_dir = os.path.join(judge_dir, model_key)
                    pd.DataFrame(model_results).to_csv(
                        os.path.join(model_dir, "alignment_results.csv"), index=False
                    )
                    print(f"    Saved: {model_dir}/alignment_results.csv")
                continue

        # If we have per-participant data, do full analysis
        alignment = analyze_agent_alignment(crowd_boot, agent_df)
        for row in alignment:
            row["mode"] = "AgentJudge"
            row["prompt"] = "N/A"
            row["model"] = model_label
            row["profile"] = "All"
        all_results.extend(alignment)
        model_results.extend(alignment)

        wt = wilcoxon_test(crowd_df, agent_df)
        for row in wt:
            row["mode"] = "AgentJudge"
            row["prompt"] = "N/A"
            row["model"] = model_label
            row["profile"] = "All"
        all_wilcoxon.extend(wt)
        model_wilcoxon.extend(wt)

        # Per education level
        for level in EDUCATION_LEVELS:
            crowd_boot_level = compute_crowd_bootstrap(crowd_df, education_level=level)
            if not crowd_boot_level:
                continue
            alignment_level = analyze_agent_alignment(
                crowd_boot_level, agent_df, education_level=level
            )
            for row in alignment_level:
                row["mode"] = "AgentJudge"
                row["prompt"] = "N/A"
                row["model"] = model_label
                row["profile"] = level
            all_results.extend(alignment_level)
            model_results.extend(alignment_level)

            wt_level = wilcoxon_test(crowd_df, agent_df, education_level=level)
            for row in wt_level:
                row["mode"] = "AgentJudge"
                row["prompt"] = "N/A"
                row["model"] = model_label
                row["profile"] = level
            all_wilcoxon.extend(wt_level)
            model_wilcoxon.extend(wt_level)

        # Save per-model results inside the model folder
        model_dir = os.path.join(judge_dir, model_key)
        if model_results:
            pd.DataFrame(model_results).to_csv(
                os.path.join(model_dir, "alignment_results.csv"), index=False
            )
            print(f"    Saved: {model_dir}/alignment_results.csv")
        if model_wilcoxon:
            pd.DataFrame(model_wilcoxon).to_csv(
                os.path.join(model_dir, "wilcoxon_results.csv"), index=False
            )
            print(f"    Saved: {model_dir}/wilcoxon_results.csv")

    return all_results, all_wilcoxon



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Running Bootstrap & Alignment Analysis")
    print("=" * 60)

    all_results = []
    all_wilcoxon = []

    # --- E2E Agent ---
    if os.path.isdir(E2E_DIR):
        print("\n[E2E Agent]")
        e2e_res, e2e_wt = run_e2e_analysis(E2E_DIR, OUTPUT_DIR)
        all_results.extend(e2e_res)
        all_wilcoxon.extend(e2e_wt)
    else:
        print(f"\n[SKIP] E2E directory not found: {E2E_DIR}")

    # --- JudgeAgent ---
    if os.path.isdir(JUDGE_DIR):
        print("\n[JudgeAgent]")
        judge_res, judge_wt = run_judge_analysis(JUDGE_DIR, OUTPUT_DIR)
        all_results.extend(judge_res)
        all_wilcoxon.extend(judge_wt)
    else:
        print(f"\n[SKIP] JudgeAgent directory not found: {JUDGE_DIR}")

    # --- Save CSVs ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "alignment_results.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/alignment_results.csv ({len(results_df)} rows)")

    if all_wilcoxon:
        wilcoxon_df = pd.DataFrame(all_wilcoxon)
        wilcoxon_df.to_csv(os.path.join(OUTPUT_DIR, "wilcoxon_results.csv"), index=False)
        print(f"Saved: {OUTPUT_DIR}/wilcoxon_results.csv ({len(wilcoxon_df)} rows)")

    # --- Print summary to console ---
    print("\n" + "=" * 60)
    print("ALIGNMENT SCORE SUMMARY (Overall, All profiles)")
    print("=" * 60)

    overall = results_df[results_df["profile"] == "All"]
    for (mode, prompt), group in overall.groupby(["mode", "prompt"]):
        label = f"{mode} [{prompt}]" if prompt != "N/A" else mode
        print(f"\n  {label}:")
        for model_label in MODELS.values():
            mg = group[group["model"] == model_label]
            if not mg.empty:
                mean_align = mg["align_score"].mean()
                n_med_compat = mg["med_compatible"].sum()
                n_iqr_compat = mg["iqr_compatible"].sum()
                n_total = len(mg)
                print(f"    {model_label:15s}  AlignScore={mean_align:.4f}  "
                      f"Compat: Med {n_med_compat}/{n_total}, IQR {n_iqr_compat}/{n_total}")

    print("\nDone.")


if __name__ == "__main__":
    main()