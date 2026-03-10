import sqlite3
import csv
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # quizcomp_llm_study/
OUTPUT_ROOT = BASE_DIR / "Results" / "e2eAgent" / "Results_as_csvs"
PRICING_FILE = BASE_DIR / "pricing_config.json"

# Mapping folder names
PROMPT_TYPES = {
    "Calibrated":       ("Calibrated results",       "detailed"),
    "Role-Anchored":    ("Role-Anchored results",    "inbetween"),
    "Task-Constrained": ("Task-Constrained results", "general"),
}

# Mapping LLMs
LLMS = {
    "GPT5":    ("gpt",     "gpt-5-2025-08-07"),
    "llama8B": ("llama",   "llama3.1:8b"),
    "Mistral": ("mistral", "mistralai/mistral-small-3.2-24b-instruct"),
}


def load_pricing():
    """Load model pricing from JSON config."""
    with open(PRICING_FILE) as f:
        config = json.load(f)
    return config["model_pricing"]


def query_tokens(db_path):
    """
    Return per-run and aggregate token data from a simulation DB.

    Returns list of dicts:
        [{"run_number": int, "num_simulations": int,
          "total_tokens": int, "total_llm_calls": int}, ...]
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT
            run_number,
            COUNT(*)           AS num_simulations,
            SUM(total_tokens)  AS total_tokens,
            SUM(total_llm_calls) AS total_llm_calls
        FROM simulations
        GROUP BY run_number
        ORDER BY run_number
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def write_token_csv(rows, out_path, prompt_type, llm_key, model_name):
    """Write per-run token CSV + an aggregate total row."""
    os.makedirs(out_path.parent, exist_ok=True)

    grand_tokens = sum(r["total_tokens"] for r in rows)
    grand_calls  = sum(r["total_llm_calls"] for r in rows)
    grand_sims   = sum(r["num_simulations"] for r in rows)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_type", "llm", "model_name", "run_number",
            "num_simulations", "total_tokens", "total_llm_calls",
        ])
        for r in rows:
            writer.writerow([
                prompt_type, llm_key, model_name, r["run_number"],
                r["num_simulations"], r["total_tokens"], r["total_llm_calls"],
            ])
        # Aggregate row
        writer.writerow([
            prompt_type, llm_key, model_name, "ALL",
            grand_sims, grand_tokens, grand_calls,
        ])
    print(f"  Wrote {len(rows)+1} rows → {out_path}")
    return {
        "prompt_type": prompt_type,
        "llm": llm_key,
        "model_name": model_name,
        "total_tokens": grand_tokens,
        "total_llm_calls": grand_calls,
        "num_simulations": grand_sims,
    }


def compute_cost(total_tokens, pricing_entry):
    """
    Estimate cost from total tokens.

    The DBs record only total_tokens (no input/output split).
    We estimate cost as:  total_tokens × average(input_rate, output_rate) / 1M
    and also provide lower-bound (all-input) and upper-bound (all-output) costs.
    """
    inp  = pricing_entry["input_per_1m"]
    out  = pricing_entry["output_per_1m"]
    avg  = (inp + out) / 2.0

    factor = total_tokens / 1_000_000.0
    return {
        "cost_avg":        round(factor * avg, 6),
        "cost_lower_input_only":  round(factor * inp, 6),
        "cost_upper_output_only": round(factor * out, 6),
        "rate_input_per_1m":  inp,
        "rate_output_per_1m": out,
        "rate_avg_per_1m":    avg,
    }


def main():
    pricing = load_pricing()
    all_summaries = []   # collected for the aggregate CSVs

    for prompt_label, (db_folder, db_suffix) in PROMPT_TYPES.items():
        print(f"\n{'='*60}")
        print(f"Prompt type: {prompt_label}  (folder: {db_folder})")
        print(f"{'='*60}")

        for llm_key, (db_frag, model_key) in LLMS.items():
            db_name = f"llm_results_{db_frag}_{db_suffix}.db"
            db_path = BASE_DIR / db_folder / db_name

            if not db_path.exists():
                print(f"  [SKIP] DB not found: {db_path}")
                continue

            rows = query_tokens(db_path)
            out_csv = OUTPUT_ROOT / prompt_label / llm_key / "tokens_summary.csv"
            summary = write_token_csv(rows, out_csv, prompt_label, llm_key, model_key)

            # Cost estimation
            if model_key in pricing:
                cost = compute_cost(summary["total_tokens"], pricing[model_key])
                summary.update(cost)
            else:
                print(f"  [WARN] No pricing for model '{model_key}'")

            all_summaries.append(summary)

    # Token summary CSV
    token_csv = OUTPUT_ROOT / "tokens_summary.csv"
    with open(token_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_type", "llm", "model_name",
            "num_simulations", "total_tokens", "total_llm_calls",
        ])
        for s in all_summaries:
            writer.writerow([
                s["prompt_type"], s["llm"], s["model_name"],
                s["num_simulations"], s["total_tokens"], s["total_llm_calls"],
            ])
    print(f"\nWrote aggregate tokens → {token_csv}")

    # Cost summary CSV 
    cost_csv = OUTPUT_ROOT / "cost_summary.csv"
    with open(cost_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_type", "llm", "model_name",
            "total_tokens",
            "rate_input_per_1m", "rate_output_per_1m", "rate_avg_per_1m",
            "cost_lower_input_only", "cost_avg", "cost_upper_output_only",
        ])
        for s in all_summaries:
            writer.writerow([
                s["prompt_type"], s["llm"], s["model_name"],
                s["total_tokens"],
                s.get("rate_input_per_1m", ""),
                s.get("rate_output_per_1m", ""),
                s.get("rate_avg_per_1m", ""),
                s.get("cost_lower_input_only", ""),
                s.get("cost_avg", ""),
                s.get("cost_upper_output_only", ""),
            ])
    print(f"Wrote aggregate costs  → {cost_csv}")

    # Print summary
    print(f"\n{'='*90}")
    print(f"{'Prompt Type':<20} {'LLM':<10} {'Tokens':>12} {'Cost (avg)':>12} {'Cost (low)':>12} {'Cost (high)':>12}")
    print(f"{'-'*90}")
    for s in all_summaries:
        print(f"{s['prompt_type']:<20} {s['llm']:<10} {s['total_tokens']:>12,} "
              f"${s.get('cost_avg', 0):>10.4f} "
              f"${s.get('cost_lower_input_only', 0):>10.4f} "
              f"${s.get('cost_upper_output_only', 0):>10.4f}")
    print(f"{'='*90}")

    # Grand totals per LLM
    print(f"\n{'='*70}")
    print(f"{'LLM':<10} {'Total Tokens':>15} {'Total Cost (avg)':>18} {'Total Cost (low)':>18} {'Total Cost (high)':>18}")
    print(f"{'-'*70}")
    for llm_key in LLMS:
        llm_rows = [s for s in all_summaries if s["llm"] == llm_key]
        tot_tokens = sum(s["total_tokens"] for s in llm_rows)
        tot_avg  = sum(s.get("cost_avg", 0) for s in llm_rows)
        tot_low  = sum(s.get("cost_lower_input_only", 0) for s in llm_rows)
        tot_high = sum(s.get("cost_upper_output_only", 0) for s in llm_rows)
        print(f"{llm_key:<10} {tot_tokens:>15,} ${tot_avg:>16.4f} ${tot_low:>16.4f} ${tot_high:>16.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
