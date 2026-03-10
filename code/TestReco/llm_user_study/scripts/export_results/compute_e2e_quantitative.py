import os
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = os.environ.get(
    "REALUSER_DB_PATH",
    os.path.join(os.path.dirname(__file__), "../../results.db"),
)

OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../llm_user_study/results/e2eAgent/results_as_csv")
)

MODELS = ("llama-3-8b", "mistral-24b-instruct", "gpt-5-2025-08-07")


def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ph = ",".join(["?"] * len(MODELS))

    #  Keep only the 5 most-recent runs per (learner, model, prompt) 
    c.execute(
        f"""CREATE TEMP TABLE IF NOT EXISTS valid_sessions AS
        SELECT id FROM (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY simulated_session_id, model_name, prompt_type
                       ORDER BY id DESC
                   ) AS rn
            FROM study_sessions
            WHERE simulated_session_id IS NOT NULL
              AND model_name IN ({ph})
        ) WHERE rn <= 5""",
        MODELS,
    )
    conn.commit()

    # Shorthand filter used in every query below
    VFILT = "s.id IN (SELECT id FROM valid_sessions)"

    rows = []  # (category, measure, formula, value)

    #  1. Participation & Completion 
    c.execute(
        f"SELECT count(*) FROM study_sessions s WHERE {VFILT}",
    )
    total = c.fetchone()[0]

    c.execute(
        f"SELECT count(*) FROM study_sessions s WHERE {VFILT} AND s.status = 'completed'",
    )
    completed = c.fetchone()[0]

    c.execute(
        f"SELECT count(*) FROM study_sessions s WHERE {VFILT} AND s.status = 'abandoned'",
    )
    abandoned = c.fetchone()[0]

    c.execute(
        f"SELECT count(DISTINCT s.id) FROM study_sessions s "
        f"WHERE {VFILT} AND s.status = 'completed' "
        f"AND s.id IN ("
        f"  SELECT session_id FROM learning_steps "
        f"  GROUP BY session_id HAVING COUNT(DISTINCT step_index) = 10"
        f")",
    )
    full = c.fetchone()[0]

    rows.append(("Participation & Completion", "Session count", "Total number of study sessions", str(total)))
    rows.append(("Participation & Completion", "Completion rate", "#completed/#sessions", f"{100*completed/total:.2f}% ({completed}/{total})"))
    rows.append(("Participation & Completion", "Drop-off rate", "#abandoned/#sessions", f"{100*abandoned/total:.2f}% ({abandoned}/{total})"))
    rows.append(("Participation & Completion", "Full completion rate", "#completed(10 steps)/#sessions", f"{100*full/total:.2f}% ({full}/{total})"))

    #  2. Latency 
    # Session duration: sum of all LLM call latencies per session
    c.execute(
        f"SELECT s.id, "
        f"  COALESCE((SELECT sum(latency_s) FROM llm_student_calls WHERE session_id=s.id), 0) + "
        f"  COALESCE((SELECT sum(latency_s) FROM orchestrator_calls WHERE session_id=s.id), 0) "
        f"FROM study_sessions s "
        f"WHERE {VFILT}",
    )
    sess_lats = [r[1] for r in c.fetchall() if r[1] is not None]
    avg_sess_min = np.mean(sess_lats) / 60

    # LLM student call avg latency (≈ step duration proxy)
    c.execute(
        f"SELECT avg(lsc.latency_s) FROM llm_student_calls lsc "
        f"JOIN study_sessions s ON lsc.session_id = s.id "
        f"WHERE {VFILT}",
    )
    student_lat = c.fetchone()[0]

    # Response time from attempts table
    c.execute(
        f"SELECT avg(a.response_time_ms / 1000.0) FROM attempts a "
        f"JOIN study_sessions s ON a.session_id = s.id "
        f"WHERE {VFILT}",
    )
    avg_rt = c.fetchone()[0]

    rows.append(("Latency", "Session duration", "sum(LLM latency) per session", f"{avg_sess_min:.2f} min"))
    rows.append(("Latency", "Learning step duration", "avg llm_student_call latency", f"{student_lat:.2f} sec"))
    rows.append(("Latency", "Response time", "avg(response_time_ms)/1000", f"{avg_rt:.2f} sec"))

    #  3. Performance 
    c.execute(
        f"SELECT avg(is_correct * 1.0) FROM attempts a "
        f"JOIN study_sessions s ON a.session_id = s.id "
        f"WHERE {VFILT} AND a.phase = 'pretest'",
    )
    pretest_acc = c.fetchone()[0]

    c.execute(
        f"SELECT avg(is_correct * 1.0) FROM attempts a "
        f"JOIN study_sessions s ON a.session_id = s.id "
        f"WHERE {VFILT} AND a.phase = 'learning'",
    )
    learning_acc = c.fetchone()[0]

    rows.append(("Performance", "Pretest accuracy", "mean(correct | phase=pretest)", f"{100*pretest_acc:.2f}%"))
    rows.append(("Performance", "Learning accuracy", "mean(correct | phase=learning)", f"{100*learning_acc:.2f}%"))

    #  4. Learning Dynamics 
    c.execute(
        f"SELECT avg(s.final_mastery - s.pretest_mastery_init) FROM study_sessions s "
        f"WHERE {VFILT} AND s.final_mastery IS NOT NULL",
    )
    mastery_gain = c.fetchone()[0]

    c.execute(
        f"SELECT avg(rp), avg(rg), avg(ra) FROM ("
        f"  SELECT s.id, sum(ls.reward_perf) rp, sum(ls.reward_gap) rg, sum(ls.reward_apt) ra "
        f"  FROM learning_steps ls JOIN study_sessions s ON ls.session_id = s.id "
        f"  WHERE {VFILT} "
        f"  GROUP BY s.id"
        f")",
    )
    rr = c.fetchone()

    rows.append(("Learning Dynamics", "Mastery gain per session", "final_mastery - initial_mastery", f"{100*mastery_gain:.2f}%"))
    rows.append(("Learning Dynamics", "Reward components", "sum r_perf, sum r_gap, sum r_apt", f"{rr[0]:.4f}, {rr[1]:.4f}, {rr[2]:.4f}"))

    #  5. Agent Cost & Behavior 
    c.execute(
        f"SELECT avg(oc.latency_s) FROM orchestrator_calls oc "
        f"JOIN study_sessions s ON oc.session_id = s.id "
        f"WHERE {VFILT}",
    )
    orch_lat = c.fetchone()[0]

    # Token usage: orchestrator_calls.total_tokens is NULL for e2e,
    # so we fall back to llm_student_calls.total_tokens.
    c.execute(
        f"SELECT avg(oc.total_tokens) FROM orchestrator_calls oc "
        f"JOIN study_sessions s ON oc.session_id = s.id "
        f"WHERE {VFILT}",
    )
    avg_tokens = c.fetchone()[0]
    if avg_tokens is None:
        c.execute(
            f"SELECT avg(lsc.total_tokens) FROM llm_student_calls lsc "
            f"JOIN study_sessions s ON lsc.session_id = s.id "
            f"WHERE {VFILT}",
        )
        avg_tokens = c.fetchone()[0]

    c.execute(
        f"SELECT oc.selected_strategy, count(*) FROM orchestrator_calls oc "
        f"JOIN study_sessions s ON oc.session_id = s.id "
        f"WHERE {VFILT} "
        f"GROUP BY oc.selected_strategy",
    )
    strat = dict(c.fetchall())
    tot_strat = sum(strat.values())

    # Map strategy keys to canonical names
    perf_pct = gap_pct = apt_pct = 0.0
    for k, cnt in strat.items():
        kl = k.lower() if k else ""
        pct = 100 * cnt / tot_strat
        if "perf" in kl:
            perf_pct = pct
        elif "gap" in kl:
            gap_pct = pct
        elif "apt" in kl or "apti" in kl:
            apt_pct = pct

    rows.append(("Agent Cost & Behavior", "Latency", "avg(latency_s)", f"{orch_lat:.2f} sec"))
    rows.append(("Agent Cost & Behavior", "Token usage per call", "sum(total_tokens)", f"{avg_tokens:.2f}"))
    rows.append(("Agent Cost & Behavior", "Strategy distribution", "f_perf, f_gap, f_apt", f"{perf_pct:.2f}%, {gap_pct:.2f}%, {apt_pct:.2f}%"))

    conn.close()

    #  Print 
    print("=" * 80)
    print("e2eAgent — Quantitative Summary")
    print("=" * 80)
    for cat, measure, formula, value in rows:
        print(f"  {cat:30s} | {measure:25s} | {value}")
    print("=" * 80)

    #  CSV 
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows, columns=["category", "measure", "formula", "e2eAgent_value"])
    csv_path = os.path.join(OUT_DIR, "e2e_quantitative_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
