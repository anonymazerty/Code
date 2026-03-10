# real_user_study/initial_estimation.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math
import random

# change according to dataset and config
PREQUALIFICATION_QA_IDS: List[str] = [
    "pretest_1", "pretest_2", "pretest_3", "pretest_4",
]


# 
# Helpers: map QA.json "id" -> question index in loaded "questions" list
# 

def _normalize_qid_str(x: Any) -> Optional[str]:
    """Normalize a QA.json-like id to a string (or None if missing)."""
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    # handle int/float ids:
    if isinstance(x, (int, float)):
        # Avoid float like 102.0
        if isinstance(x, float) and not x.is_integer():
            return str(x)
        return str(int(x))
    return str(x).strip() or None


def build_qaid_to_index(questions: List[dict]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, q in enumerate(questions):
        qid_str = _normalize_qid_str(q.get("id"))
        if not qid_str:
            continue
        mapping.setdefault(qid_str, idx)
    return mapping


def get_prequalification_indices(questions: List[dict]) -> List[int]:
    """
    Return the fixed 12 prequalification question indices (into `questions`)
    in the exact order of PREQUALIFICATION_QA_IDS.

    Raises a clear error if any required QA id is missing.
    """
    qaid_to_idx = build_qaid_to_index(questions)
    missing = [qaid for qaid in PREQUALIFICATION_QA_IDS if qaid not in qaid_to_idx]
    if missing:
        raise KeyError(
            "Prequalification test is configured with QA.json ids that were not found "
            f"in the loaded questions list. Missing ids: {missing}. "
            "Check that you are loading the correct QA.json and that ids match."
        )
    return [qaid_to_idx[qaid] for qaid in PREQUALIFICATION_QA_IDS]


# 
# Pretest selection API
# 

def sample_pretest_pattern(
    questions: List[dict],
    difficulties: Dict[int, dict],
    rng: random.Random,
    *,
    n_total: int = 12,
) -> List[int]:
    """
    Deterministic prequalification test.

    Returns the fixed list of 12 question indices.
    """
    qids = get_prequalification_indices(questions)
    if n_total != 12:
        raise ValueError(f"Prequalification test is fixed to 12 questions; got n_total={n_total}.")
    return qids


# 
# Initial mastery = (# correct) / 12
# 

def initial_mastery_from_pretest(
    pretest_qids: List[int],
    correctness_by_qid: Dict[int, bool],
) -> float:
    """
    Compute initial mastery as:
        mastery = (# correct answers) / 12
    """
    if not pretest_qids:
        return 0.0
    correct = sum(1 for qid in pretest_qids if bool(correctness_by_qid.get(qid, False)))
    return float(correct) / float(len(pretest_qids))


# 
# Rasch helpers
# 

def estimate_theta_rasch(scaled_difficulties: List[float], correctness: List[bool]) -> float:
    """
    Rasch-based theta estimation.
    """
    if not scaled_difficulties or not correctness:
        return 0.0
    p = sum(1 for c in correctness if c) / max(1, len(correctness))
    p = min(0.99, max(0.01, p))
    logit = math.log(p / (1.0 - p))
    avg_b = sum(scaled_difficulties) / len(scaled_difficulties)
    return float(logit + (avg_b - 0.5))


def theta_to_mastery(theta: float) -> float:
    """Map theta to [0,1] mastery via sigmoid."""
    return float(1.0 / (1.0 + math.exp(-theta)))



@dataclass
class PretestAnalysis:
    is_guessing: bool
    fast_qids: List[int]           # rt < threshold
    thought_qids: List[int]        # rt >= threshold
    consistent_pairs: int
    total_pairs: int
    thought_correct: int
    debug: Dict[str, Any]


def analyze_pretest_guessing(
    questions: List[dict],
    difficulties: Dict[int, dict],
    qids: List[int],
    correctness_by_qid: Dict[int, bool],
    rt_seconds_by_qid: Dict[int, float],
    *,
    min_seconds_thought: float = 40.0,
    min_consistent_pairs: int = 3,
    fast_questions_threshold: int = 4,
    allow_near_level_pairs: bool = True,
) -> PretestAnalysis:
    """
    Detect guessing based on fast answers and pair consistency.
    """
    fast_qids = [qid for qid in qids if rt_seconds_by_qid.get(qid, 9999.0) < min_seconds_thought]
    thought_qids = [qid for qid in qids if qid not in fast_qids]

    def _question_subtopic(q: dict) -> Optional[str]:
        s = q.get("subtopic")
        if isinstance(s, str) and s.strip():
            return s.strip()
        meta = q.get("metadata", {})
        if isinstance(meta, dict) and isinstance(meta.get("subtopic"), str):
            return meta["subtopic"].strip()
        return None

    def _level_1_to_5(original_difficulty: Any) -> int:
        try:
            lvl = int(round(float(original_difficulty)))
        except Exception:
            lvl = 1
        return max(1, min(5, lvl))

    def sub_and_level(qid: int) -> Tuple[Optional[str], int]:
        sub = _question_subtopic(questions[qid])
        lvl = _level_1_to_5(difficulties.get(qid, {}).get("original_difficulty", 1))
        return sub, lvl

    groups: Dict[Tuple[str, int], List[int]] = {}
    for qid in qids:
        sub, lvl = sub_and_level(qid)
        if sub is None:
            continue
        groups.setdefault((sub, lvl), []).append(qid)

    pairs: List[Tuple[int, int, str]] = []

    # Same-level pairs first
    for (sub, lvl), items in groups.items():
        items_sorted = sorted(items)
        for i in range(0, len(items_sorted) - 1, 2):
            a, b = items_sorted[i], items_sorted[i + 1]
            pairs.append((a, b, f"{sub}|L{lvl}-same"))

    # Near-level pairs (L and L+1) for leftover singletons
    if allow_near_level_pairs:
        paired_set = set([x for p in pairs for x in p[:2]])
        subtopics = sorted({k[0] for k in groups.keys()})
        for sub in subtopics:
            for lvl in range(1, 5):
                A = [qid for qid in groups.get((sub, lvl), []) if qid not in paired_set]
                B = [qid for qid in groups.get((sub, lvl + 1), []) if qid not in paired_set]
                for a, b in zip(sorted(A), sorted(B)):
                    pairs.append((a, b, f"{sub}|L{lvl}-L{lvl+1}-near"))
                    paired_set.add(a)
                    paired_set.add(b)

    consistent_pairs = 0
    evaluated_pairs = 0
    pair_details: List[dict] = []

    for a, b, tag in pairs:
        if a in correctness_by_qid and b in correctness_by_qid:
            evaluated_pairs += 1
            ca = bool(correctness_by_qid[a])
            cb = bool(correctness_by_qid[b])
            ok = (ca == cb)  # TT or FF
            if ok:
                consistent_pairs += 1
            pair_details.append({
                "pair": (a, b),
                "tag": tag,
                "correctness": (ca, cb),
                "rt_s": (rt_seconds_by_qid.get(a), rt_seconds_by_qid.get(b)),
                "consistent": ok,
            })

    too_fast = len(fast_qids) >= fast_questions_threshold
    not_enough_consistency = consistent_pairs < min_consistent_pairs
    is_guessing = bool(too_fast or not_enough_consistency)

    thought_correct = sum(1 for qid in thought_qids if correctness_by_qid.get(qid, False))

    return PretestAnalysis(
        is_guessing=is_guessing,
        fast_qids=fast_qids,
        thought_qids=thought_qids,
        consistent_pairs=consistent_pairs,
        total_pairs=evaluated_pairs,
        thought_correct=thought_correct,
        debug={
            "min_seconds_thought": min_seconds_thought,
            "fast_questions_threshold": fast_questions_threshold,
            "min_consistent_pairs": min_consistent_pairs,
            "allow_near_level_pairs": allow_near_level_pairs,
            "pair_details": pair_details,
        },
    )


