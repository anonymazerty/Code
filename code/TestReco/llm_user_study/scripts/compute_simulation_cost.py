import argparse
import csv
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_user_study.config import DB_PATH, MODEL_PRICING

PROMPT_LABELS = {
    "detailed": "Calibrated",
    "general":  "Task-Constrained",
    "simple":   "Role-Anchored",
}

MODELS = list(MODEL_PRICING.keys())

# 55 most-recent sessions per (run_number × model × prompt_type)
PERSONAS_PER_RUN = 55

RESULTS_BASE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


#  helpers 

def cost(in_tok, out_tok, pricing):
    return (in_tok * pricing["input_per_1m"] + out_tok * pricing["output_per_1m"]) / 1_000_000


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Wrote {path}")


#  e2eAgent 

def _e2e_filtered_session_ids(con):
    """Return session IDs: 55 most-complete sessions per (run_number × model × prompt_type).
    """
    ph = ",".join("?" * len(MODELS))
    rows = con.execute(f"""
        WITH complete_sessions AS (
            -- A session is complete if it has a survey_response row,
            -- matching the same filter used in export_csvs.py.
            SELECT ss.id, ss.model_name, ss.prompt_type, ss.simulated_session_id
            FROM study_sessions ss
            JOIN survey_responses sr ON sr.session_id = ss.id
            WHERE ss.model_name IN ({ph})
              AND ss.prompt_type IS NOT NULL
              AND ss.simulated_session_id IS NOT NULL
        ),
        run_assigned AS (
            SELECT id, model_name, prompt_type,
                   ROW_NUMBER() OVER (
                       PARTITION BY simulated_session_id, model_name, prompt_type
                       ORDER BY id ASC
                   ) AS run_number
            FROM complete_sessions
        ),
        per_run_ranked AS (
            SELECT id, model_name, prompt_type, run_number,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_name, prompt_type, run_number
                       ORDER BY id DESC
                   ) AS rn
            FROM run_assigned
            WHERE run_number <= 5
        )
        SELECT id FROM per_run_ranked WHERE rn <= {PERSONAS_PER_RUN}
    """, MODELS).fetchall()
    return [r[0] for r in rows]


def compute_e2e_costs(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    session_ids = _e2e_filtered_session_ids(con)
    ph = ",".join("?" * len(session_ids))

    session_counts = con.execute(f"""
        SELECT model_name, prompt_type,
               COUNT(DISTINCT id) AS sessions,
               COUNT(DISTINCT simulated_session_id) AS personas
        FROM study_sessions
        WHERE id IN ({ph})
        GROUP BY model_name, prompt_type
        ORDER BY model_name, prompt_type
    """, session_ids).fetchall()

    rows = con.execute(f"""
        SELECT ss.model_name, ss.prompt_type,
               SUM(CASE WHEN lsc.call_type='question' THEN lsc.input_tokens  ELSE 0 END) AS q_in,
               SUM(CASE WHEN lsc.call_type='question' THEN lsc.output_tokens ELSE 0 END) AS q_out,
               SUM(CASE WHEN lsc.call_type='survey'   THEN lsc.input_tokens  ELSE 0 END) AS s_in,
               SUM(CASE WHEN lsc.call_type='survey'   THEN lsc.output_tokens ELSE 0 END) AS s_out,
               SUM(CASE WHEN lsc.call_type='question' THEN 1 ELSE 0 END) AS n_q_calls,
               SUM(CASE WHEN lsc.call_type='survey'   THEN 1 ELSE 0 END) AS n_s_calls
        FROM llm_student_calls lsc
        JOIN study_sessions ss ON ss.id = lsc.session_id
        WHERE ss.id IN ({ph})
        GROUP BY ss.model_name, ss.prompt_type
        ORDER BY ss.model_name, ss.prompt_type
    """, session_ids).fetchall()

    con.close()
    return rows, session_counts


def print_e2e(rows, session_counts):
    print("Student simulation only\n")

    print("Pricing (per 1M tokens):")
    for model, p in MODEL_PRICING.items():
        print(f"    {model:<26}  input=${p['input_per_1m']:.3f}  output=${p['output_per_1m']:.3f}")

    print("Session counts after filtering:")
    for mn, pt, sess, pers in session_counts:
        print(f"    {mn:<26} {PROMPT_LABELS.get(pt, pt):<18}  sessions={sess}  personas={pers}")

    print("Per model & strategy")
    hdr = f"  {'Model':<26} {'Strategy':<18} {'Simul-student $':>16} {'Survey $':>12} {'Total $':>10} {'Q-calls':>9} {'S-calls':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    model_q  = {m: 0.0 for m in MODELS}
    model_s  = {m: 0.0 for m in MODELS}
    prompt_q = {p: 0.0 for p in PROMPT_LABELS}
    prompt_s = {p: 0.0 for p in PROMPT_LABELS}

    for model, pt, qi, qo, si, so, nq, ns in rows:
        p   = MODEL_PRICING[model]
        q_c = cost(qi, qo, p)
        s_c = cost(si, so, p)
        lbl = PROMPT_LABELS.get(pt, pt)
        print(f"  {model:<26} {lbl:<18} ${q_c:>15.4f} ${s_c:>11.4f} ${q_c + s_c:>9.4f} {nq:>9} {ns:>9}")
        model_q[model] += q_c
        model_s[model] += s_c
        if pt in prompt_q:
            prompt_q[pt] += q_c
            prompt_s[pt] += s_c

    print("Per model (all 3 strategies)")
    print(f"  {'Model':<26} {'Simul-student $':>16} {'Survey $':>12} {'Total $':>10}")
    print("  " + "-" * 66)
    grand_q = grand_s = 0.0
    for m in MODELS:
        q_c, s_c = model_q[m], model_s[m]
        print(f"  {m:<26} ${q_c:>15.4f} ${s_c:>11.4f} ${q_c + s_c:>9.4f}")
        grand_q += q_c
        grand_s += s_c
    print(f"  {'TOTAL':<26} ${grand_q:>15.4f} ${grand_s:>11.4f} ${grand_q + grand_s:>9.4f}")

    print("Per prompt strategy (all 3 models)")
    print(f"  {'Strategy':<18} {'Simul-student $':>16} {'Survey $':>12} {'Total $':>10}")
    print("  " + "-" * 58)
    for pt, lbl in PROMPT_LABELS.items():
        q_c, s_c = prompt_q[pt], prompt_s[pt]
        print(f"  {lbl:<18} ${q_c:>15.4f} ${s_c:>11.4f} ${q_c + s_c:>9.4f}")

    print(f"  Simulating student (questions):  ${grand_q:>8.4f}")
    print(f"  Answering survey:                ${grand_s:>8.4f}")
    print(f"  GRAND TOTAL:                     ${grand_q + grand_s:>8.4f}")
    print()

    return model_q, model_s, prompt_q, prompt_s, grand_q, grand_s


def export_e2e_csvs(rows, model_q, model_s, prompt_q, prompt_s, grand_q, grand_s):
    cost_dir    = os.path.join(RESULTS_BASE, "e2eAgent", "results_as_csv", "cost")
    summary_dir = os.path.join(RESULTS_BASE, "e2eAgent", "results_as_csv")

    # cost_by_model_strategy.csv
    _write_csv(
        os.path.join(cost_dir, "cost_by_model_strategy.csv"),
        ["model", "strategy", "simul_student_usd", "survey_usd", "total_usd", "n_q_calls", "n_s_calls"],
        [
            [model, PROMPT_LABELS.get(pt, pt),
             round(cost(qi, qo, MODEL_PRICING[model]), 6),
             round(cost(si, so, MODEL_PRICING[model]), 6),
             round(cost(qi, qo, MODEL_PRICING[model]) + cost(si, so, MODEL_PRICING[model]), 6),
             nq, ns]
            for model, pt, qi, qo, si, so, nq, ns in rows
        ],
    )

    # cost_by_model.csv
    _write_csv(
        os.path.join(cost_dir, "cost_by_model.csv"),
        ["model", "simul_student_usd", "survey_usd", "total_usd"],
        [[m, round(model_q[m], 6), round(model_s[m], 6), round(model_q[m] + model_s[m], 6)]
         for m in MODELS],
    )

    # cost_by_strategy.csv
    _write_csv(
        os.path.join(cost_dir, "cost_by_strategy.csv"),
        ["strategy", "simul_student_usd", "survey_usd", "total_usd"],
        [[lbl, round(prompt_q[pt], 6), round(prompt_s[pt], 6),
          round(prompt_q[pt] + prompt_s[pt], 6)]
         for pt, lbl in PROMPT_LABELS.items()],
    )

    # cost_summary.csv 
    _write_csv(
        os.path.join(summary_dir, "cost_summary.csv"),
        ["category", "simul_student_usd", "survey_usd", "total_usd"],
        [["grand_total", round(grand_q, 6), round(grand_s, 6), round(grand_q + grand_s, 6)]],
    )

def compute_judge_costs(db_path=DB_PATH):
    """Aggregate costs for the 55 most-complete rows per (model × run_number).

    'Complete' = highest total_tokens (most thorough survey response), id DESC as tiebreaker.
    """
    con = sqlite3.connect(db_path)
    rows = con.execute(f"""
        WITH ranked AS (
            SELECT model_name, run_number, input_tokens, output_tokens,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_name, run_number
                       ORDER BY total_tokens DESC, id DESC
                   ) AS completeness_rank
            FROM survey_test_retest
            WHERE model_name IS NOT NULL
              AND total_tokens IS NOT NULL
        )
        SELECT model_name, run_number,
               SUM(input_tokens)  AS total_in,
               SUM(output_tokens) AS total_out,
               COUNT(*)           AS n_calls
        FROM ranked
        WHERE completeness_rank <= {PERSONAS_PER_RUN}
        GROUP BY model_name, run_number
        ORDER BY model_name, run_number
    """).fetchall()
    con.close()
    return rows


def print_judge(rows):
    print("\n=== JUDGE SIMULATION COST  (survey-only, per model × run) ===")
    print("    Orchestrator excluded\n")

    print("  Pricing (per 1M tokens):")
    for model, p in MODEL_PRICING.items():
        print(f"    {model:<26}  input=${p['input_per_1m']:.3f}  output=${p['output_per_1m']:.3f}")

    print("Per model & run")
    hdr = f"  {'Model':<26} {'Run':>4} {'Survey $':>12} {'Calls':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    model_cost = {m: 0.0 for m in MODELS}

    for model, run, ti, to, n in rows:
        if model not in MODEL_PRICING:
            continue
        c = cost(ti, to, MODEL_PRICING[model])
        print(f"  {model:<26} {run:>4} ${c:>11.4f} {n:>7}")
        model_cost[model] += c

    print("Per model (all runs)")
    print(f"  {'Model':<26} {'Survey $':>12}")
    print("  " + "-" * 40)
    grand = 0.0
    for m in MODELS:
        print(f"  {m:<26} ${model_cost[m]:>11.4f}")
        grand += model_cost[m]
    print(f"  {'TOTAL':<26} ${grand:>11.4f}")

    print("Summary")
    print(f"TOTAL (survey only):  ${grand:>8.4f}")
    print()

    return model_cost, grand


def export_judge_csvs(rows, model_cost, grand):
    cost_dir    = os.path.join(RESULTS_BASE, "JudgeAgent", "results_as_csv", "cost")
    summary_dir = os.path.join(RESULTS_BASE, "JudgeAgent", "results_as_csv")

    # cost_by_model_run.csv
    _write_csv(
        os.path.join(cost_dir, "cost_by_model_run.csv"),
        ["model", "run_number", "survey_usd", "n_calls"],
        [
            [model, run, round(cost(ti, to, MODEL_PRICING[model]), 6), n]
            for model, run, ti, to, n in rows
            if model in MODEL_PRICING
        ],
    )

    # cost_by_model.csv
    _write_csv(
        os.path.join(cost_dir, "cost_by_model.csv"),
        ["model", "survey_usd", "total_usd"],
        [[m, round(model_cost[m], 6), round(model_cost[m], 6)] for m in MODELS],
    )

    # cost_summary.csv  (grand — results general)
    _write_csv(
        os.path.join(summary_dir, "cost_summary.csv"),
        ["category", "survey_usd", "total_usd"],
        [["grand_total", round(grand, 6), round(grand, 6)]],
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["e2e", "judge"], required=True,
                        help="Agent mode: e2e (full simulation) or judge (survey-only)")
    args = parser.parse_args()

    if args.agent == "e2e":
        rows, session_counts = compute_e2e_costs()
        model_q, model_s, prompt_q, prompt_s, grand_q, grand_s = print_e2e(rows, session_counts)
        print("CSV export")
        export_e2e_csvs(rows, model_q, model_s, prompt_q, prompt_s, grand_q, grand_s)
    else:
        rows = compute_judge_costs()
        model_cost, grand = print_judge(rows)
        print("CSV export")
        export_judge_csvs(rows, model_cost, grand)


if __name__ == "__main__":
    main()
