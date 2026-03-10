from typing import Dict, List, Tuple
import pandas as pd

from utils.benchmark_utils import BenchmarkLoader

MATH_BENCH_QA_PATH = "benchmarks/math_bench/processed/QA.json"
MATH_BENCH_DIFF_CSV = "benchmarks/math_bench/processed/math_bench_QA_scaled_difficulty.csv"


def load_math_bench_full() -> List[dict]:
    # Returns list of questions, each containing at least:
    # {"id", "topic", "level" (or "difficulty"), "options", "correct_answer" (or "answer")}
    return BenchmarkLoader.load_math_bench(MATH_BENCH_QA_PATH, limit=None)


def load_diff_map_from_csv() -> Dict[str, dict]:
    """
    Return mapping: question_id(str) -> {"original_difficulty": float, "scaled_difficulty": float}

    IMPORTANT:
    - With your new CSV, question_id matches QA.json "id" (NOT the full list index).
    """
    df = pd.read_csv(MATH_BENCH_DIFF_CSV)

    # Normalize column types
    # question_id can be int or str depending on csv; we store as str for stable matching
    df["question_id"] = df["question_id"].astype(str)

    id_to_diff: Dict[str, dict] = {}
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        id_to_diff[qid] = {
            "original_difficulty": float(row["original_difficulty"]),
            "scaled_difficulty": float(row["scaled_difficulty"]),
        }
    return id_to_diff


def _qa_original_level(q: dict) -> float:
    """
    QA.json uses 'level' as the discrete difficulty (same as original_difficulty).
    We normalize to float.
    """
    if "level" in q and q["level"] is not None:
        try:
            return float(q["level"])
        except Exception:
            pass
    # fallback for alternative field name
    if "difficulty" in q and q["difficulty"] is not None:
        try:
            return float(q["difficulty"])
        except Exception:
            pass
    return 0.0


def load_topic_questions_with_difficulties(topic: str) -> Tuple[List[dict], Dict[int, dict]]:
    """
    Returns:
      - questions: filtered list (env index = list index)
      - difficulties: dict keyed by env index => {original_difficulty, scaled_difficulty}

    Matching rule:
      - CSV question_id == QA.json "id"
    """
    full_questions = load_math_bench_full()
    id_to_diff = load_diff_map_from_csv()

    # Filter by topic
    questions = [q for q in full_questions if q.get("topic") == topic]

    difficulties: Dict[int, dict] = {}
    for env_idx, q in enumerate(questions):
        dataset_id = str(q.get("id"))

        d = id_to_diff.get(dataset_id)
        if d is None:
            # Fallback: preserve QA level as original difficulty, and use neutral scaled
            d = {
                "original_difficulty": _qa_original_level(q),
                "scaled_difficulty": 0.5,
            }
        else:
            # Ensure original_difficulty is consistent with QA "level" for strict agreement:
            # To enforce equality, uncomment this block.
            qa_od = _qa_original_level(q)
            if qa_od != 0.0:
                d = dict(d)  # copy
                d["original_difficulty"] = qa_od

        difficulties[env_idx] = d

    return questions, difficulties
