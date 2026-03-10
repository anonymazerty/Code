# real_user_study/app.py

import os
import re
import time
import json
import random
import datetime as dt
import streamlit as st
import streamlit.components.v1 as components
from real_user_study.completion_codes import assign_completion_code
from passlib.hash import pbkdf2_sha256
from generators.model import token_usage_callback
from typing import Optional
from sqlalchemy.exc import IntegrityError


from real_user_study.db import (
    init_db, SessionLocal, now_utc,
    User, StudySession, Attempt, LearningStep, OrchestratorCall, SurveyResponse, to_json
)
from real_user_study.loader import load_topic_questions_with_difficulties
from real_user_study.live_session import RealUserEngine
from real_user_study.settings import (
    TOPIC, BENCHMARK, ORCHESTRATOR_TYPE, MODEL_NAME,
    STEPS_N, QUESTIONS_PER_STEP, POLICY_FOLDERS
)
from real_user_study.ui_components import (
    header,
    landing_overview,
    consent_block,
    qualification_intro,
    qualification_results_block,
    learning_intro,
    learning_completion_block,
    survey_intro,
    study_complete_block,
    likert,
    ms_since,
)

from real_user_study.initial_estimation import (
    sample_pretest_pattern,
    initial_mastery_from_pretest,
    analyze_pretest_guessing,
    PREQUALIFICATION_QA_IDS,
)

st.set_page_config(page_title="Real-User Study", layout="centered")


# -------------------------
# Study gating knobs
# -------------------------
MIN_PRETEST_CORRECT = 0       # must get at least this many correct (0 disables)
MIN_PRETEST_MASTERY = 0.5    # mastery here = correct/12 ratio
PRETEST_TARGET_N = len(PREQUALIFICATION_QA_IDS)
# For mastery-based gating only, set MIN_PRETEST_CORRECT = 0


# -------------------------
# Helpers
# -------------------------
def insert_attempt_if_missing(db, attempt: Attempt) -> bool:
    """
    Avoid duplicate Attempt rows due to Streamlit reruns/double clicks.
    Uniqueness logic:
      - pretest: (session_id, phase, question_index)
      - learning: (session_id, phase, step_index, question_index)
    Returns True if inserted, False if skipped (already exists).
    """
    q = db.query(Attempt).filter(
        Attempt.session_id == attempt.session_id,
        Attempt.phase == attempt.phase,
        Attempt.question_index == attempt.question_index,
    )
    if attempt.phase == "learning":
        q = q.filter(Attempt.step_index == attempt.step_index)

    if q.first() is not None:
        return False

    db.add(attempt)
    return True


def insert_orchestrator_call_if_missing(db, oc: OrchestratorCall) -> bool:
    """
    Avoid duplicate OrchestratorCall per (session_id, step_index).
    Returns True if inserted, False if skipped.
    """
    exists = db.query(OrchestratorCall).filter(
        OrchestratorCall.session_id == oc.session_id,
        OrchestratorCall.step_index == oc.step_index,
    ).first()
    if exists is not None:
        return False

    db.add(oc)
    return True

def db_session():
    return SessionLocal()

def get_step_token_usage(step_start_prompt_tokens: int, step_start_completion_tokens: int) -> dict:
    """
    Compute token usage for this step (delta since step start).
    Uses token_usage_callback counters (non-Claude path, like your main.py).
    """
    step_input_tokens = int(token_usage_callback.total_prompt_tokens - step_start_prompt_tokens)
    step_output_tokens = int(token_usage_callback.total_completion_tokens - step_start_completion_tokens)
    return {
        "input_tokens": step_input_tokens,
        "output_tokens": step_output_tokens,
        "total_tokens": step_input_tokens + step_output_tokens,
    }


def require_state_defaults():
    defaults = {
        "engine": None,
        "questions": None,
        "difficulties": None,

        "study_session_id": None,
        "phase": "landing",
        "survey_started": False,

        "pretest_qids": [],
        "pretest_t0": {},
        "pretest_submitted": False,
        "pretest_mastery": None,
        "pretest_correct": None,
        "pretest_is_eligible": None,
        "pretest_analysis": None,

        "learning_step": 0,
        "learning_current_qids": [],
        "learning_t0": {},
        "step_action": None,
        "step_orch_info": None,
        "step_action_info": None,
        "step_latency": None,
        "step_obs": None,
        "step_started_at": None,

        "orchestrator_attached": False,

        "username": None,
        "completion_code": None,

        # debug guard (prevents printing selection on every rerun)
        "_printed_pretest_selection": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_current_user(db):
    uid = st.session_state.get("user_id")
    if not uid:
        return None
    return db.query(User).filter(User.id == uid).one_or_none()


def login_ui(db):
    st.subheader("Login / Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            u = db.query(User).filter(User.username == username).one_or_none()
            if not u or not pbkdf2_sha256.verify(password, u.password_hash):
                st.error("Invalid username/password")
            else:
                u.last_login_at = now_utc()
                db.commit()
                st.session_state.user_id = u.id
                st.session_state.username = u.username  # <-- add this
                st.success("Logged in")
                st.rerun()

    with tab2:
        username = st.text_input("Username", key="reg_user")
        password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register"):
            if db.query(User).filter(User.username == username).count() > 0:
                st.error("Username already exists")
            else:
                u = User(username=username, password_hash=pbkdf2_sha256.hash(password))
                db.add(u)
                db.commit()
                st.session_state.user_id = u.id
                st.session_state.username = u.username  # <-- add this
                st.success("Registered and logged in")
                st.rerun()


def create_or_load_engine():
    """
    Creates ONLY env + question bank.
    Orchestrator is attached AFTER pretest.
    """
    if st.session_state.engine is None:
        questions, difficulties = load_topic_questions_with_difficulties(TOPIC)
        st.session_state.questions = questions
        st.session_state.difficulties = difficulties

        st.session_state.engine = RealUserEngine(
            topic=TOPIC,
            questions=questions,
            difficulties=difficulties,
            max_steps=STEPS_N,
            questions_per_step=QUESTIONS_PER_STEP,
            objectives=["aptitude", "gap", "performance"],
            seed=42,
            ncc_window=2,
        )

        st.session_state.engine.env.mastery[0] = 0.4

        st.session_state.orchestrator_attached = False


def start_new_study_session(db, user_id: int):
    sess = StudySession(
        user_id=user_id,
        topic=TOPIC,
        benchmark=BENCHMARK,
        orchestrator_type=ORCHESTRATOR_TYPE,
        model_name=MODEL_NAME,
        status="created",
        pretest_num_questions=len(PREQUALIFICATION_QA_IDS),
        total_steps_planned=STEPS_N,
        questions_per_step=QUESTIONS_PER_STEP,
    )
    db.add(sess)
    db.commit()

    # reset all UI state
    st.session_state.study_session_id = sess.id
    st.session_state.phase = "consent"

    st.session_state.pretest_qids = []
    st.session_state.pretest_t0 = {}
    st.session_state.pretest_submitted = False
    st.session_state.pretest_mastery = None
    st.session_state.pretest_correct = None
    st.session_state.pretest_is_eligible = None
    st.session_state.pretest_analysis = None

    st.session_state.learning_step = 0
    st.session_state.learning_current_qids = []
    st.session_state.learning_t0 = {}
    st.session_state.step_action = None
    st.session_state.step_orch_info = None
    st.session_state.step_action_info = None
    st.session_state.step_latency = None
    st.session_state.step_obs = None
    st.session_state.step_started_at = None

    st.session_state.orchestrator_attached = False
    st.session_state._printed_pretest_selection = False


def mark_session_status(db, session_id: int, status: str, abandoned_phase: Optional[str] = None):
    sess = db.query(StudySession).filter(StudySession.id == session_id).one()
    sess.status = status

    if status == "abandoned":
        sess.ended_at = now_utc()
        sess.abandoned_phase = abandoned_phase
        sess.abandoned_at = now_utc()

    if status == "completed":
        sess.ended_at = now_utc()

    db.commit()


_LATEX_REPL = [
    (r"\\mathbb\{R\}", "R"),
    (r"\\mathbb\{Z\}", "Z"),
    (r"\\mathbb\{Q\}", "Q"),
    (r"\\mathbb\{N\}", "N"),
    (r"\\infty", "∞"),
    (r"\\cup", "∪"),
    (r"\\cap", "∩"),
    (r"\\neq", "≠"),
    (r"\\leq", "≤"),
    (r"\\geq", "≥"),
    (r"\\pm", "±"),
    (r"\\times", "×"),
    (r"\\cdot", "·"),
    (r"\\rightarrow", "→"),
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\:", " "),
]


def de_latex(s: str) -> str:
    if not isinstance(s, str):
        return str(s)

    # remove surrounding $...$
    s = re.sub(r"\$(.*?)\$", r"\1", s)

    # replace common commands
    for pat, rep in _LATEX_REPL:
        s = re.sub(pat, rep, s)

    # fractions like \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

    # sqrt \sqrt{3} or \sqrt[3]{x}
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", s)
    s = re.sub(r"\\sqrt\[(\d+)\]\{([^{}]+)\}", r"root\1(\2)", s)

    # remove curly braces leftovers
    s = s.replace("{", "").replace("}", "")

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------
# LaTeX rendering helpers
# -------------------------

def normalize_latex(s: str) -> str:
    """
    Fix common shorthand + convert bracketed LaTeX delimiters to Streamlit-friendly ones.
      - \dfrac23 -> \dfrac{2}{3}
      - \frac23  -> \frac{2}{3}
      - \[ ... \] -> $$ ... $$
      - \( ... \) -> $ ... $
    """
    if not isinstance(s, str):
        return str(s)

    s = s.strip()

    # Convert display/inline delimiters that Streamlit markdown won't parse reliably
    s = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", s, flags=re.DOTALL)  # \[...\] -> $$...$$
    s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)   # \(...\) -> $...$

    # Fix digit/digit shorthand fractions (single digit numerator/denominator)
    s = re.sub(r"\\dfrac\s*([0-9])\s*([0-9])", r"\\dfrac{\1}{\2}", s)
    s = re.sub(r"\\frac\s*([0-9])\s*([0-9])", r"\\frac{\1}{\2}", s)

    return s


def wrap_math_for_markdown(s: str) -> str:
    """
    - If already contains $ or $$, keep as-is.
    - If it's a mixed English sentence (contains words), DO NOT wrap the whole thing.
      Instead, wrap only:
        - standalone single-letter variables (r, v, x...) in the sentence
        - trailing math expression like v=... in $...$
    - If it's mostly math, wrap whole string in $...$.
    """
    if not isinstance(s, str):
        s = str(s)

    s = normalize_latex(s)

    # Already math-delimited
    if "$" in s:
        return s

    # Detect "sentence-ish" text (words like Make/subject/expression etc.)
    has_words = bool(re.search(r"[A-Za-z]{2,}", s))  # at least one 2+ letter word

    if has_words:
        # Wrap standalone single-letter variables: " r " -> "$r$"
        s = re.sub(r"\b([a-zA-Z])\b", r"$\1$", s)

        # Wrap a trailing math expression of the form "v=..." if present
        m = re.search(r"([a-zA-Z]\s*=\s*.*)$", s)
        if m and any(tok in m.group(1) for tok in ["\\", "^", "_", "=", "\\pi", "\\frac", "\\dfrac"]):
            expr = m.group(1)
            s = s[: m.start(1)] + f"${expr}$"

        return s

    # Otherwise: looks like pure math -> wrap
    if any(tok in s for tok in ["\\", "^", "_", "=", "\\frac", "\\dfrac", "\\pi", "\\sqrt"]):
        return f"${s}$"

    return s

# --------------------------
# End session button helper
# --------------------------
def end_session_footer(db, sess):
    left, right = st.columns([8, 2])

    with left:
        st.markdown(':red[Reminer:]  You can end the session anytime you want by clicking on **End session**.')

    with right:
        if st.button("End session", key=f"end_session_{st.session_state.phase}"):
            mark_session_status(
                db,
                sess.id,
                "abandoned",
                abandoned_phase=st.session_state.phase,
            )

            # Clear current session so user can start a new one later.
            st.session_state.study_session_id = None
            st.session_state.engine = None
            st.session_state.phase = "landing"
            st.rerun()




def extract_question_text_raw(q: dict) -> str:
    """
    Like your extract_question_text, but DOES NOT de_latex (keeps LaTeX).
    """
    for k in ("question", "prompt", "stem", "text", "body"):
        if k in q and q[k]:
            return str(q[k])
    if "title" in q and q["title"]:
        return str(q["title"])
    return ""


def extract_options_raw(q: dict) -> list:
    opts = q.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        return [str(x) for x in opts]
    opts = q.get("choices")
    if isinstance(opts, list) and len(opts) > 0:
        return [str(x) for x in opts]
    return []


# -------------------------
# User-facing "Level" display helper
# -------------------------

def user_level_label(qid: int, difficulties: dict) -> str:
    """
    Show the author/gold difficulty level (discrete) to the user.
    Loader provides difficulties[qid]["original_difficulty"].
    Clamp to 1..5.
    """
    d = difficulties.get(qid, {})
    try:
        od = float(d.get("original_difficulty", 0.0))
        lvl = int(round(od))
        lvl = max(1, min(5, lvl))
        return f"Level {lvl}"
    except Exception:
        return "Level ?"


# -------------------------
# Custom option selector (no preselection)
# -------------------------
import inspect

def render_question_no_preselect(q: dict, key_prefix: str):
    qtext_raw = extract_question_text_raw(q)
    if qtext_raw:
        st.markdown(wrap_math_for_markdown(qtext_raw))

    options_raw = extract_options_raw(q)
    options_disp = [wrap_math_for_markdown(x) for x in options_raw]
    correct_idx = int(q.get("correct_answer", q.get("answer", -1)))

    choice_key = f"{key_prefix}_choice_idx"
    if choice_key not in st.session_state:
        st.session_state[choice_key] = -1

    chosen_idx = int(st.session_state[choice_key])
    letters = [chr(ord("A") + i) for i in range(len(options_disp))]

    # Some Streamlit versions support button type= ("primary"/"secondary")
    supports_type = "type" in inspect.signature(st.button).parameters

    for i, opt_md in enumerate(options_disp):
        left, right = st.columns([1, 12])

        btn_type = None
        if chosen_idx == i:
            btn_type = "primary"
        else:
            btn_type = "secondary"

        if supports_type:
            clicked = left.button(
                letters[i],
                key=f"{key_prefix}_pick_{i}",
                type=btn_type,
                use_container_width=True,
            )
        else:
            # Fallback for older Streamlit versions
            clicked = left.button(
                ("● " + letters[i]) if chosen_idx == i else letters[i],
                key=f"{key_prefix}_pick_{i}",
                use_container_width=True,
            )

        if clicked:
            st.session_state[choice_key] = i
            st.rerun()
            #chosen_idx = i

        # Keep selection indicator INLINE (no stacking)
        dot = ":blue[●] " if chosen_idx == i else ""
        right.markdown(f"{opt_md}")

    st.session_state[f"{key_prefix}_choice"] = None if chosen_idx < 0 else (
        options_raw[chosen_idx] if (0 <= chosen_idx < len(options_raw)) else None
    )

    is_correct = (chosen_idx == correct_idx) if chosen_idx >= 0 else False
    return chosen_idx, correct_idx, bool(is_correct)




def safe_choice_index(options, choice):
    if not choice:
        return -1
    try:
        return options.index(choice)
    except ValueError:
        return -1


# -------------------------
# Main
# -------------------------

def main():
    init_db()
    require_state_defaults()

    db = db_session()
    try:
        header()

        user = get_current_user(db)

        # Landing page before login
        if st.session_state.phase == "landing":
            if landing_overview():
                st.session_state.phase = "login"
                st.rerun()
            return

        # Login page
        if not user:
            st.session_state.phase = "login"
            login_ui(db)
            return

        # If user is logged in and still on login phase, move on to consent
        if st.session_state.phase == "login":
            st.session_state.phase = "consent"
            st.rerun()

        # Always load engine/questions once user is logged in
        create_or_load_engine()

        sess = None
        if st.session_state.study_session_id is not None:
            sess = (
                db.query(StudySession)
                .filter(StudySession.id == st.session_state.study_session_id)
                .one_or_none()
            )
            # If deleted from DB etc.
            if sess is None:
                st.session_state.study_session_id = None


        phase = st.session_state.get("phase", "consent")
        # If user is in a study phase but no DB session exists, send them back.
        if phase not in ("landing", "login", "consent") and sess is None:
            st.session_state.phase = "landing"
            st.rerun()

        questions = st.session_state.questions
        difficulties = st.session_state.difficulties
        engine = st.session_state.engine

        # -------- Phase 1: consent ----------
        if phase == "consent":
            ok = consent_block()
            if ok:
                if st.button("Continue to qualification test"):
                    # Create session ONLY here (user explicitly starts)
                    if st.session_state.study_session_id is None:
                        start_new_study_session(db, user.id)
                        sess = db.query(StudySession).filter(
                            StudySession.id == st.session_state.study_session_id
                        ).one()

                    mark_session_status(db, sess.id, "pretest")
                    st.session_state.phase = "pretest_intro"
                    st.rerun()
            else:
                st.info("Please provide consent to proceed.")

            if sess is not None:
                end_session_footer(db, sess)
            return

        # -------- Phase 2a: Pretest intro ----------
        if phase == "pretest_intro":
            if qualification_intro(n_questions=PRETEST_TARGET_N):
                st.session_state.phase = "pretest"
                st.rerun()
            if sess is not None:
                end_session_footer(db, sess)
            return

        # -------- Phase 2: Pretest ----------
        if phase == "pretest":
            st.subheader(f"Qualification test ({PRETEST_TARGET_N} questions)")

            if not st.session_state.pretest_qids:
                rng = random.Random(42 + int(sess.id))
                picked = sample_pretest_pattern(
                    questions=questions,
                    difficulties=difficulties,
                    rng=rng,
                    n_total=PRETEST_TARGET_N,
                )
                st.session_state.pretest_qids = picked
                st.session_state._printed_pretest_selection = False

            # Print selected pretest questions (print once, not on every rerun)
            if not st.session_state._printed_pretest_selection:
                print("\n===== PRETEST QUESTION SELECTION (pattern-based) =====")
                for i, qid in enumerate(st.session_state.pretest_qids):
                    q = questions[qid]
                    diff = difficulties[qid]
                    try:
                        sd = float(diff.get("scaled_difficulty", 0.0))
                    except Exception:
                        sd = 0.0
                    try:
                        od = float(diff.get("original_difficulty", 0.0))
                    except Exception:
                        od = 0.0
                    print(
                        f"Q{i+1}: env_idx={qid}, dataset_id={q.get('id')}, "
                        f"scaled_difficulty={sd:.3f}, original_difficulty={od:.3f}, "
                        f"subtopic={q.get('subtopic')}"
                    )
                print("======================================================\n")
                st.session_state._printed_pretest_selection = True

            qids = st.session_state.pretest_qids
            correctness_list = []
            chosen_idxs = []

            for i, qid in enumerate(qids):
                q = questions[qid]
                level_label = user_level_label(qid, difficulties)

                st.markdown(f"**Q{i+1}**  ({level_label})")

                if qid not in st.session_state.pretest_t0:
                    st.session_state.pretest_t0[qid] = time.time()

                chosen_idx, _, is_correct = render_question_no_preselect(q, f"pre_{qid}")
                chosen_idxs.append(chosen_idx)
                correctness_list.append(bool(is_correct))

                st.markdown("---")

            all_answered = all(idx >= 0 for idx in chosen_idxs)

            if st.button("Submit pretest"):
                if not all_answered:
                    st.error("You must answer all questions before submitting.")
                    return

                # Build correctness + RT maps
                correctness_by_qid: dict[int, bool] = {}
                rt_seconds_by_qid: dict[int, float] = {}
                for qid, is_correct in zip(qids, correctness_list):
                    correctness_by_qid[int(qid)] = bool(is_correct)
                    t0 = st.session_state.pretest_t0[qid]
                    rt_ms = ms_since(t0)
                    rt_seconds_by_qid[int(qid)] = float(rt_ms) / 1000.0

                total_correct = sum(1 for ok in correctness_list if ok)

                # mastery here is simply correct/12 (for gating + display)
                score_ratio = float(
                    initial_mastery_from_pretest(
                        pretest_qids=qids,
                        correctness_by_qid=correctness_by_qid,
                    )
                )

                # guessing detection: pairs consistency + time
                analysis_obj = analyze_pretest_guessing(
                    questions=questions,
                    difficulties=difficulties,
                    qids=qids,
                    correctness_by_qid=correctness_by_qid,
                    rt_seconds_by_qid=rt_seconds_by_qid,
                    min_seconds_thought=40.0,
                    min_consistent_pairs=3,
                    fast_questions_threshold=4,
                    allow_near_level_pairs=True,
                )

                # Eligibility = (not guessing) AND thresholds
                pass_correct = (total_correct >= MIN_PRETEST_CORRECT) if MIN_PRETEST_CORRECT > 0 else True
                pass_mastery = (score_ratio >= MIN_PRETEST_MASTERY)
                is_eligible = (not analysis_obj.is_guessing) and pass_correct and pass_mastery

                # Learning mastery is fixed to 0.4 (not from pretest)
                # learning mastery remains default 0.4
                st.session_state.pretest_submitted = True
                st.session_state.pretest_correct = int(total_correct)
                st.session_state.pretest_mastery = float(score_ratio)  # displayed as "mastery" in results
                st.session_state.pretest_is_eligible = bool(is_eligible)

                st.session_state.pretest_analysis = {
                    "is_guessing": bool(analysis_obj.is_guessing),
                    "consistent_pairs": int(analysis_obj.consistent_pairs),
                    "total_pairs": int(analysis_obj.total_pairs),
                    "fast_count": int(len(analysis_obj.fast_qids)),
                    "thought_count": int(len(analysis_obj.thought_qids)),
                    "correct_thought": int(analysis_obj.thought_correct),
                }

                # Persist attempts (store all 12 attempts for audit)
                for qid, is_correct in zip(qids, correctness_list):
                    q = questions[qid]
                    t0 = st.session_state.pretest_t0[qid]
                    rt_ms = ms_since(t0)

                    chosen = st.session_state.get(f"pre_{qid}_choice")
                    chosen_idx = safe_choice_index(q.get("options", []), chosen)

                    attempt = Attempt(
                        session_id=sess.id,
                        phase="pretest",
                        step_index=None,
                        question_index=qid,
                        dataset_question_id=str(q.get("id", qid)),
                        topic=q.get("topic", TOPIC),
                        original_difficulty=float(difficulties[qid].get("original_difficulty", 0.0)),
                        scaled_difficulty=float(difficulties[qid].get("scaled_difficulty", 0.5)),
                        chosen_option_index=int(chosen_idx),
                        correct_option_index=int(q.get("correct_answer", q.get("answer", -1))),
                        is_correct=bool(is_correct),
                        started_at=dt.datetime.fromtimestamp(t0, tz=dt.timezone.utc),
                        answered_at=now_utc(),
                        response_time_ms=int(rt_ms),
                    )
                    insert_attempt_if_missing(db, attempt)
                
                # Store pretest values in session row
                sess.pretest_num_questions = len(st.session_state.pretest_qids)
                sess.pretest_correct = int(total_correct)
                sess.pretest_mastery_init = float(score_ratio)  # pretest ratio, not learning init mastery
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()

                mark_session_status(db, sess.id, "pretest_done")
                st.session_state.phase = "pretest_result"
                st.rerun()
            if sess is not None:
                end_session_footer(db, sess)
            return

        # -------- Phase 2b: Pretest result + confirmation ----------
        if phase == "pretest_result":
            m0 = float(st.session_state.pretest_mastery or 0.0)  # = correct/12 ratio
            score = int(st.session_state.pretest_correct or 0)
            eligible = bool(st.session_state.pretest_is_eligible)
            analysis = st.session_state.get("pretest_analysis") or {}

            go_next = qualification_results_block(
                passed=eligible,
                score=score,
                total=PRETEST_TARGET_N,
                mastery=m0,
                guessing_info=analysis,
            )

            if not eligible:
                mark_session_status(db, sess.id, "abandoned", abandoned_phase="pretest_result")
                if st.button("End"):
                    st.session_state.study_session_id = None
                    st.session_state.engine = None
                    st.rerun()
                return

            if go_next:
                st.session_state.phase = "learning_intro"
                st.rerun()

            if sess is not None:
                end_session_footer(db, sess)
            return

        # -------- Phase 3a: Learning intro ----------
        if phase == "learning_intro":
            if learning_intro(steps_n=STEPS_N, max_q_per_step=QUESTIONS_PER_STEP):
                # Attach orchestrator NOW (only once)
                if not st.session_state.orchestrator_attached:
                    engine.attach_orchestrator(
                        policy_folders=POLICY_FOLDERS,
                        model_name=MODEL_NAME,
                        objectives=["aptitude", "gap", "performance"],
                        orchestrator_type=ORCHESTRATOR_TYPE,
                        verbose=True,
                    )
                    st.session_state.orchestrator_attached = True

                    # Ensure learning starts at fixed 0.4 (attach/reset might overwrite it)
                    engine.env.mastery[0] = 0.4

                mark_session_status(db, sess.id, "learning")
                st.session_state.phase = "learning"
                st.rerun()
            if sess is not None:
                end_session_footer(db, sess)
            return

        # -------- Phase 3: Learning loop ----------
        if phase == "learning":
            step = int(st.session_state.learning_step)
            st.subheader(f"Learning phase — step {step+1}/{STEPS_N}")

            if engine.orchestrator is None:
                st.error("Orchestrator not attached. Please restart the session.")
                return

            if not st.session_state.learning_current_qids:
                obs = engine.get_state_obs()
                # ) Snapshot token counters BEFORE the orchestrator/LLM call
                step_start_prompt_tokens = int(token_usage_callback.total_prompt_tokens)
                step_start_completion_tokens = int(token_usage_callback.total_completion_tokens)

                # 2) This is the call that triggers LLM usage
                action_info, orch_info, latency = engine.select_action()

                # 3) Compute token deltas AFTER the call
                step_input_tokens = int(token_usage_callback.total_prompt_tokens - step_start_prompt_tokens)
                step_output_tokens = int(token_usage_callback.total_completion_tokens - step_start_completion_tokens)
                step_total_tokens = step_input_tokens + step_output_tokens


                action = int(action_info["action"])
                qids = engine.get_questions_for_action(action)

                st.session_state.learning_current_qids = qids
                st.session_state.step_action = action
                st.session_state.step_orch_info = orch_info
                st.session_state.step_action_info = action_info
                st.session_state.step_latency = latency
                st.session_state.step_obs = obs.tolist()
                st.session_state.step_started_at = time.time()

                oc = OrchestratorCall(
                    session_id=sess.id,
                    step_index=step,
                    state_obs_json=to_json(st.session_state.step_obs),
                    selected_strategy=orch_info.get("selected_strategy"),
                    action=action,
                    latency_s=float(latency),
                    # Fill token columns (these were NULL in the db)
                    input_tokens=step_input_tokens,
                    output_tokens=step_output_tokens,
                    total_tokens=step_total_tokens,
        
                    raw_action_info_json=to_json(action_info),
                    raw_orchestrator_info_json=to_json(orch_info),

                    
                )
                insert_orchestrator_call_if_missing(db, oc)
                db.commit()
                
            qids = st.session_state.learning_current_qids
            action = st.session_state.step_action
            st.info("Tip: take your time, response times are recorded.")

            correctness_list = []
            chosen_idxs = []

            for i, qid in enumerate(qids):
                q = questions[qid]
                level_label = user_level_label(qid, difficulties)
                st.markdown(f"**Q{i+1}**  ({level_label})")

                if qid not in st.session_state.learning_t0:
                    st.session_state.learning_t0[qid] = time.time()

                chosen_idx, _, is_correct = render_question_no_preselect(q, f"learn_{step}_{qid}")
                chosen_idxs.append(chosen_idx)
                correctness_list.append(bool(is_correct))

                st.markdown("---")

            all_answered = all(idx >= 0 for idx in chosen_idxs)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit this step"):
                    if not all_answered:
                        st.error("You must answer all questions in this step.")
                        return

                    # Save attempts
                    for qid, is_correct in zip(qids, correctness_list):
                        q = questions[qid]
                        t0 = st.session_state.learning_t0[qid]
                        rt_ms = ms_since(t0)

                        chosen = st.session_state.get(f"learn_{step}_{qid}_choice")
                        chosen_idx = safe_choice_index(q.get("options", []), chosen)

                        attempt = Attempt(
                            session_id=sess.id,
                            phase="learning",
                            step_index=step,
                            question_index=qid,
                            dataset_question_id=str(q.get("id", qid)),
                            topic=q.get("topic", TOPIC),
                            original_difficulty=float(difficulties[qid].get("original_difficulty", 0.0)),
                            scaled_difficulty=float(difficulties[qid].get("scaled_difficulty", 0.5)),
                            chosen_option_index=int(chosen_idx),
                            correct_option_index=int(q.get("correct_answer", q.get("answer", -1))),
                            is_correct=bool(is_correct),
                            started_at=dt.datetime.fromtimestamp(t0, tz=dt.timezone.utc),
                            answered_at=now_utc(),
                            response_time_ms=int(rt_ms),
                        )
                        insert_attempt_if_missing(db, attempt)
                    try:
                        db.commit()
                    except IntegrityError:
                        db.rollback()
                    mastery_before = float(engine.env.mastery[0])
                    res = engine.apply_learning_batch(action, qids, correctness_list)
                    mastery_after = float(engine.env.mastery[0])

                    step_row = LearningStep(
                        session_id=sess.id,
                        step_index=step,
                        action=int(action),
                        selected_question_indices_json=to_json(qids),
                        selected_question_dataset_ids_json=to_json([str(questions[qid].get("id", qid)) for qid in qids]),
                        mastery_before=mastery_before,
                        mastery_after=mastery_after,
                        rolling_accuracy=float(res.rolling_accuracy),
                        reward_perf=float(res.rewards.get("performance", 0.0)),
                        reward_gap=float(res.rewards.get("gap", 0.0)),
                        reward_apt=float(res.rewards.get("aptitude", 0.0)),
                        started_at=dt.datetime.fromtimestamp(st.session_state.step_started_at, tz=dt.timezone.utc),
                        ended_at=now_utc(),
                    )
                    db.add(step_row)
                    try:
                        db.commit()
                    except IntegrityError:
                        db.rollback()
                        # Step already exists (likely rerun/double click). Continue safely.

                    st.success(
                        f"Step submitted. mastery {mastery_before:.2f} → {mastery_after:.2f}, "
                        f"accuracy={res.rolling_accuracy:.2f}"
                    )

                    # advance
                    st.session_state.learning_step = step + 1
                    st.session_state.learning_current_qids = []
                    st.session_state.learning_t0 = {}
                    st.session_state.step_action = None
                    st.session_state.step_orch_info = None
                    st.session_state.step_action_info = None
                    st.session_state.step_latency = None
                    st.session_state.step_obs = None
                    st.session_state.step_started_at = None

                    if st.session_state.learning_step >= STEPS_N:
                        mark_session_status(db, sess.id, "survey")
                        st.session_state.phase = "learning_done"

                    st.rerun()


            if sess is not None:
                end_session_footer(db, sess)
            return

        # -------- Phase 3b: Learning completion ----------
        if phase == "learning_done":
            rows = (
                db.query(LearningStep)
                .filter(LearningStep.session_id == sess.id)
                .order_by(LearningStep.step_index.asc())
                .all()
            )
            mastery_series = [r.mastery_after for r in rows] if rows else None
            final_mastery = float(engine.env.mastery[0])

            if learning_completion_block(mastery_value=final_mastery, mastery_series=mastery_series):
                st.session_state.survey_started = False
                st.session_state.phase = "survey_intro"
                st.rerun()
            if sess is not None:
                end_session_footer(db, sess)
            return

        # -------- Phase 4a: Survey intro ----------
        if phase == "survey_intro":
            if survey_intro():
                st.session_state.survey_started = True
                st.session_state.phase = "survey"
                st.rerun()
            if sess is not None:
                end_session_footer(db, sess)
            return

        # -------- Phase 4: Survey ----------
        if phase == "survey":
            st.subheader("Post-study survey")

            q1 = likert(
                ":red[Feeling of Accomplishment:] How much do you feel that the recommended questions helped you improve your understanding of the topic?",
                "sv_q1",
            )
            q2 = likert(
                ":red[Effort Required:] How much effort was required to follow the recommendations and work through the questions?",
                "sv_q2",
            )
            q3 = likert(
                ":red[Mental Demand:] How mentally demanding was it to understand and solve the recommended questions?",
                "sv_q3",
            )
            q4 = likert(
                ":red[Perceived Controllability:] How well did the difficulty of the recommended questions match your current level throughout the session?",
                "sv_q4",
            )
            q5 = likert(
                ":red[Temporal Demand:] How time-pressured did you feel while completing the recommended questions within the allotted time?",
                "sv_q5",
            )
            q6 = likert(
                ":red[Frustration:] How frustrated did you feel while working with the recommended questions?",
                "sv_q6",
            )
            q7 = likert(
                ":red[Trust:] How much did you trust the system to recommend appropriate questions for your learning progress?",
                "sv_q7",
            )


            again = st.checkbox("I would use such a system again.", value=True, key="sv_again")
            free_text = st.text_area("Any comments?", key="sv_text")

            if st.button("Submit survey & finish"):
                if any(v is None for v in [q1, q2, q3, q4, q5, q6, q7]):
                    st.error("Please answer all 7 scale questions (1–5) before submitting.")
                    return

                attempts = db.query(Attempt).filter(
                    Attempt.session_id == sess.id,
                    Attempt.phase == "learning"
                ).all()
                final_acc = (sum(1 for a in attempts if a.is_correct) / len(attempts)) if attempts else None

                sess.final_mastery = float(st.session_state.engine.env.mastery[0])
                sess.final_accuracy = float(final_acc) if final_acc is not None else None

                sr = SurveyResponse(
                    session_id=sess.id,
                    accomplishment=int(q1),
                    effort_required=int(q2),
                    mental_demand=int(q3),
                    perceived_controllability=int(q4),
                    temporal_demand=int(q5),
                    frustration=int(q6),
                    trust=int(q7),
                    would_use_again=bool(again),
                    free_text=free_text,
                )
                db.add(sr)
                mark_session_status(db, sess.id, "completed")
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    # Survey already submitted for this session; continue safely.

                # Assign Prolific completion code NOW (so it's fixed immediately)
                if st.session_state.get("completion_code") is None:
                    username_key = st.session_state.get("username") or user.username
                    st.session_state["completion_code"] = assign_completion_code(username_key)

                st.session_state.phase = "completed"
                st.rerun()
            if sess is not None:
                end_session_footer(db, sess)
            return


        # -------- Final: Completed ----------
        if phase == "completed":
            # Assign ONE code per username (persistently)
            if st.session_state.get("completion_code") is None:
                username_key = st.session_state.get("username") or user.username
                st.session_state["completion_code"] = assign_completion_code(username_key)

            study_complete_block()

            st.markdown("### Your Prolific completion code")
            st.code(st.session_state["completion_code"], language=None)
            st.caption("Copy/paste this code into Prolific to receive payment.")
            return

    finally:
        db.close()



if __name__ == "__main__":
    main()
