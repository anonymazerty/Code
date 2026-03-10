# quizcomp_ui/str_app.py
import os
import re
import time
from typing import Optional
from db import User, now_utc

import requests
import pandas as pd
import streamlit as st

from config import (
    API_BASE_DEFAULT,
    TOPICS,
    NUM_DIFFICULTIES,
    DEFAULT_MCQS,
    DEFAULT_NUM_QUIZZES,
    DEFAULT_NUM_TOPICS,
    DEFAULT_TOPIC_MODE,
    DEFAULT_LEVEL_MODE,
    DEFAULT_ORDER_LEVEL,
    DEFAULT_MODEL_PATH,
    DEFAULT_ALFA,
)
from ui_components import (
    topic_selector,
    difficulty_selector,
    render_best_quiz,
    render_coverage,
    render_mcq_list,
)
from db import init_db, SessionLocal, log_event, ComposeAttempt, SurveyResponse, to_json
from auth import create_user, authenticate
from study_ui import (
    study_landing_page,
    consent_block,
    ensure_study_session,
    mark_consented,
    end_session,
)

API_BASE = os.environ.get("QUIZCOMP_API_BASE", API_BASE_DEFAULT)

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# -----------------------------
# Helpers
# -----------------------------
def extract_uuid(s: str) -> Optional[str]:
    if not s:
        return None
    m = UUID_RE.search(s)
    return m.group(0) if m else None


def api_post(path: str, payload: dict, timeout_s: int = 300):
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.post(url, json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"{path} failed ({r.status_code}): {r.text}")
    return r.json()


def parse_generation_response(resp_json):
    if isinstance(resp_json, dict):
        path = resp_json.get("PathToQuizzes") or resp_json.get("pathToQuizzes") or ""
        rid = resp_json.get("RequestID") or resp_json.get("request_id") or ""
        if not rid:
            rid = extract_uuid(path) or ""
        return rid, path
    path = str(resp_json)
    return extract_uuid(path) or "", path


def parse_num_mcqs(x: str) -> Optional[int]:
    x = (x or "").strip()
    if not x:
        return None
    try:
        v = int(x)
    except Exception:
        return None
    if v < 3 or v > 30:
        return None
    return v


def extract_result_vectors(best_json: dict) -> tuple[list[int], list[float], list[float]]:
    if not best_json:
        return [], [0.0] * len(TOPICS), [0.0] * NUM_DIFFICULTIES

    mcq_cols = sorted(
        [k for k in best_json.keys() if k.startswith("mcq_")],
        key=lambda x: int(x.split("_")[1]),
    )
    mcq_ids: list[int] = []
    for c in mcq_cols:
        try:
            mcq_ids.append(int(best_json[c]))
        except Exception:
            pass

    topic_cov: list[float] = []
    for i in range(len(TOPICS)):
        try:
            topic_cov.append(float(best_json.get(f"topic_coverage_{i}", 0.0)))
        except Exception:
            topic_cov.append(0.0)

    diff_cov: list[float] = []
    for i in range(NUM_DIFFICULTIES):
        try:
            diff_cov.append(float(best_json.get(f"difficulty_coverage_{i}", 0.0)))
        except Exception:
            diff_cov.append(0.0)

    return mcq_ids, topic_cov, diff_cov


def load_mcqs_for_best_quiz(best_json: dict, data_uuid: str):
    if not best_json:
        return None
    mcq_cols = sorted(
        [k for k in best_json.keys() if k.startswith("mcq_")],
        key=lambda x: int(x.split("_")[1]),
    )
    mcq_ids: list[int] = []
    for c in mcq_cols:
        try:
            mcq_ids.append(int(best_json[c]))
        except Exception:
            pass
    mcq_resp = api_post("/mcqs/by_ids", {"ids": mcq_ids, "dataUUID": data_uuid})
    return mcq_resp.get("items", [])


def raw_topic_total_pct(chosen_topics: list[str]) -> int:
    total = 0
    for name in chosen_topics or []:
        total += int(st.session_state.get(f"tw_{name}", 0) or 0)
    return int(total)


def raw_difficulty_total_pct(num_difficulties: int) -> int:
    total = 0
    for i in range(num_difficulties):
        lab = f"Level {i+1}"
        total += int(st.session_state.get(f"dw_{lab}", 0) or 0)
    return int(total)


def render_current_params_summary():
    """
    Compact summary (no editing). Uses current session_state values.
    """
    num_mcqs = st.session_state.get("num_mcqs", None)
    chosen_topics = st.session_state.get("selected_topics", []) or []
    teacherTopic = st.session_state.get("last_teacherTopic", None)
    teacherLevel = st.session_state.get("last_teacherLevel", None)

    topic_rows = []
    for name in chosen_topics:
        pct = int(st.session_state.get(f"tw_{name}", 0) or 0)
        if pct > 0:
            topic_rows.append((name, pct))

    diff_rows = []
    for i in range(NUM_DIFFICULTIES):
        lab = f"Level {i+1}"
        pct = int(st.session_state.get(f"dw_{lab}", 0) or 0)
        if pct > 0:
            diff_rows.append((lab, pct))

    st.markdown("### Current parameters (locked)")
    if num_mcqs is not None:
        st.write(f"- **Number of questions:** {int(num_mcqs)}")
    else:
        st.write("- **Number of questions:** —")

    if topic_rows:
        st.write("- **Topics:**")
        for name, pct in topic_rows:
            st.write(f"  - {name}: {pct}%")
    else:
        if isinstance(teacherTopic, list) and len(teacherTopic) == len(TOPICS):
            st.write("- **Topics:**")
            for i, (_tid, tname, _count) in enumerate(TOPICS):
                v = float(teacherTopic[i] or 0.0)
                if v > 0:
                    st.write(f"  - {tname}: {int(round(v * 100))}%")
        else:
            st.write("- **Topics:** —")

    if diff_rows:
        st.write("- **Difficulties:**")
        for lab, pct in diff_rows:
            st.write(f"  - {lab}: {pct}%")
    else:
        if isinstance(teacherLevel, list) and len(teacherLevel) == NUM_DIFFICULTIES:
            st.write("- **Difficulties:**")
            for i in range(NUM_DIFFICULTIES):
                v = float(teacherLevel[i] or 0.0)
                if v > 0:
                    st.write(f"  - Level {i+1}: {int(round(v * 100))}%")
        else:
            st.write("- **Difficulties:** —")


def has_minimum_interaction() -> bool:
    """
    Returns True if the participant has generated and seen at least one quiz.
    This is the minimum requirement to count as a valid study interaction.
    """
    return st.session_state.get("best_quiz_json") is not None

def prolific_entry_error_page():
    st.error("Invalid study entry")

    st.markdown(
        """
        This study must be accessed **directly from Prolific**.

        It looks like the required participant information was not provided.

        ### What to do:
        1. Return to **Prolific**
        2. Open the study again using the **Start Study** button
        3. Do **not** bookmark or manually copy the link

        If you believe this is an error, please contact the researcher via Prolific.
        """
    )

    st.stop()

def prolific_already_participated_page():
    st.error("Study already ended")

    st.markdown(
        """
        Our records show that you have already **ended this study**.

        For data integrity reasons, participants cannot restart the study
        after completing or abandoning it.

        ### What to do
        - Please return to **Prolific**
        - Submit the study if applicable
        - If you believe this is an error, contact the researcher via Prolific
        """
    )
    st.stop()

def get_or_create_user_id(prolific_pid: str) -> int:
    """
    Create exactly one user per Prolific PID and return user.id.
    """
    db = SessionLocal()
    try:
        user = (
            db.query(User)
            .filter(User.username == prolific_pid)
            .one_or_none()
        )

        if user:
            return user.id

        user = User(
            username=prolific_pid,
            password_hash="prolific",  # unused, but NOT NULL
            created_at=now_utc(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        return user.id
    finally:
        db.close()


# Streamlit init

st.set_page_config(page_title="QuizComp Study", layout="centered")
init_db()


# Prolific entry validation

qp = st.query_params

if (
    "PROLIFIC_PID" not in qp
    or "STUDY_ID" not in qp
    or "SESSION_ID" not in qp
):
    prolific_entry_error_page()



# Global session state
st.session_state.setdefault("phase", "landing")  # landing -> auth -> tool -> survey -> done
st.session_state.setdefault("user_id", None)
st.session_state.setdefault("study_session_id", None)
st.session_state.setdefault("consented", False)

# Tool sub-pages
st.session_state.setdefault("tool_page", "main")

# Tool state
st.session_state.setdefault("data_uuid", None)
st.session_state.setdefault("universe_path", None)
st.session_state.setdefault("same_quiz_repeat_count", 0)
st.session_state.setdefault("num_mcqs", None)
st.session_state.setdefault("last_teacherTopic", None)
st.session_state.setdefault("last_teacherLevel", None)
st.session_state.setdefault("best_quiz_json", None)
st.session_state.setdefault("previous_quiz_json", None)
st.session_state.setdefault("last_best_quiz_id", None)
st.session_state.setdefault("best_mcqs", None)
st.session_state.setdefault("selected_topics", [])
st.session_state.setdefault("finish_status", "abandoned")
st.session_state.setdefault("quiz_downloaded", False)
st.session_state.setdefault("quiz_validated", False)

# when True, we hide parameter editing UI on main and show quiz compare directly.
st.session_state.setdefault("lock_params_view", False)



# Phase 1: Landing (overview only)

if st.session_state.phase == "landing":
    study_landing_page()

    col_spacer, col_next = st.columns([3, 1])
    with col_next:
        proceed = st.button("Next", type="primary")

    if proceed:
        st.session_state.phase = "consent"
        st.rerun()


# Phase 2: Consent

if st.session_state.phase == "consent":
    st.title("Consent and Data Use")

    agreed = consent_block()

    col_back, col_next = st.columns([1, 1])
    with col_back:
        back = st.button("Back")
    with col_next:
        proceed = st.button("Continue", type="primary")

    if back:
        st.session_state.phase = "landing"
        st.rerun()

    if proceed:
        if not agreed:
            st.error("Please provide your consent before continuing.")
            st.stop()

        st.session_state.consented = True

        # No login; Prolific identity is handled at session level
        prolific_pid = st.query_params.get("PROLIFIC_PID")

        user_id = get_or_create_user_id(prolific_pid)
        st.session_state.user_id = user_id

        session_id = ensure_study_session(user_id)

        mark_consented(session_id)

        st.session_state.phase = "tool"
        st.session_state.tool_page = "params"
        st.rerun()


# Phase 3: Tool

if st.session_state.phase == "tool":
    st.title("Quiz Composition")

    # 🔒 Session integrity check (refresh / lost state)
    if not st.session_state.get("study_session_id"):
        st.warning(
            "Your session could not be restored. Please return to Prolific and restart the study."
        )
        st.stop()

    if st.session_state.best_quiz_json is None:
        st.caption("Step 1 of 3 • Configure your quiz")
    else:
        st.caption("Step 2 of 3 • Review and refine the quiz")

    if not st.session_state.user_id:
        st.error("Not logged in.")
        st.stop()

    session_id = int(st.session_state.study_session_id) if st.session_state.study_session_id else 0
    if not session_id:
        session_id = ensure_study_session(st.session_state.user_id)

    db = SessionLocal()
    try:
        log_event(db, session_id, "page_view", name=f"tool:{st.session_state.tool_page}")
    finally:
        db.close()

    st.markdown(
        """
<style>
div[data-testid="stRadio"]:has(input[name="satisfied_choice"]) div[role="radiogroup"]{
  display:flex;
  gap: 36px;
  align-items:center;
}
</style>
""",
        unsafe_allow_html=True,
    )

    
    # NEXT STEP
    
    if st.session_state.tool_page == "next_step":
        st.subheader("Next step")

        choice = st.radio(
            "What do you want to do next?",
            ["Keep the same parameters and get another quiz", "Modify the parameters"],
            index=None,
            key="next_step_choice",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            back = st.button("Back", key="btn_next_step_back")
        with col_b:
            go = st.button("Continue", type="primary", key="btn_next_step_go")

        if back:
            st.session_state.tool_page = "main"
            if "satisfied_choice" in st.session_state:
                del st.session_state["satisfied_choice"]
            # back just returns to quiz view
            st.rerun()

        if go:
            if choice is None:
                st.error("Please choose an option to continue.")
                st.stop()

            if choice == "Modify the parameters":
                # Unlock and go back to editable parameters
                st.session_state.lock_params_view = False
                st.session_state.tool_page = "main"
                st.session_state.previous_quiz_json = None
                for k in ("next_step_choice", "satisfied_choice"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

            # Keep same parameters: lock UI and directly improve
            st.session_state.lock_params_view = True

            if st.session_state.data_uuid is None or st.session_state.last_best_quiz_id is None:
                st.error("Cannot improve because no previous quiz exists.")
                st.stop()

            previous_quiz_id = st.session_state.last_best_quiz_id
            st.session_state.previous_quiz_json = st.session_state.best_quiz_json

            compose_payload = {
                "dataUUID": st.session_state.data_uuid,
                "teacherTopic": st.session_state.last_teacherTopic,
                "teacherLevel": st.session_state.last_teacherLevel,
                "pathToModel": DEFAULT_MODEL_PATH,
                "alfaValue": DEFAULT_ALFA,
                "startQuizId": st.session_state.last_best_quiz_id,
            }

            t0 = time.time()
            try:
                with st.spinner("Improving your quiz... Please wait."):
                    resp = api_post("/compose/quiz", compose_payload)
                dt_s = time.time() - t0

                best_json = resp.get("best_quiz") if isinstance(resp, dict) else None
                st.session_state.best_quiz_json = best_json

                best_quiz_id = None
                target_match = None
                if best_json and "quiz_id" in best_json:
                    best_quiz_id = int(best_json["quiz_id"])
                    st.session_state.last_best_quiz_id = best_quiz_id

                if best_json and "targetMatch" in best_json:
                    try:
                        target_match = float(best_json["targetMatch"])
                    except Exception:
                        target_match = None

                st.session_state.best_mcqs = None
                if best_json:
                    st.session_state.best_mcqs = load_mcqs_for_best_quiz(best_json, st.session_state.data_uuid)

                mcq_ids, topic_cov, diff_cov = extract_result_vectors(best_json or {})

                if best_quiz_id == previous_quiz_id:
                    st.session_state.same_quiz_repeat_count = st.session_state.get("same_quiz_repeat_count", 0) + 1
                else:
                    st.session_state.same_quiz_repeat_count = 0

                db = SessionLocal()
                try:
                    a = ComposeAttempt(
                        session_id=session_id,
                        mode="improve",
                        data_uuid=str(st.session_state.data_uuid),
                        start_quiz_id=int(compose_payload["startQuizId"]),
                        teacher_topic_json=to_json(st.session_state.last_teacherTopic),
                        teacher_level_json=to_json(st.session_state.last_teacherLevel),
                        num_mcqs=int(st.session_state.num_mcqs or 0),
                        mcq_ids_json=to_json(mcq_ids),
                        topic_coverage_json=to_json(topic_cov),
                        difficulty_coverage_json=to_json(diff_cov),
                        best_quiz_id=best_quiz_id,
                        target_match=target_match,
                        api_time_s=float(dt_s),
                        succeeded=True,
                        error=None,
                    )
                    db.add(a)
                    db.commit()
                finally:
                    db.close()

                st.session_state.tool_page = "main"
                for k in ("satisfied_choice", "next_step_choice"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

            except Exception as e:
                db = SessionLocal()
                try:
                    log_event(db, session_id, "error", name="compose_quiz_failed_improve", payload={"error": str(e)})
                finally:
                    db.close()
                st.error(f"Composition failed: {e}")
                st.stop()

        st.divider()

        st.markdown(
            '<p style="color: #cc0000; font-weight: bold;">Note: Remember that you can end the study anytime by clicking the <strong>"End study now"</strong> button below.</p>',
            unsafe_allow_html=True,
        )

        abandon = st.button("End study now", key="btn_end_study_now_next_step")
        if abandon:
            if not has_minimum_interaction():
                st.warning(
                    "To complete the study, you need to generate and review at least one quiz "
                    "and complete the final questionnaire."
                )
                st.info(
                    "If you exit now, the study will be recorded as incomplete."
                )
                st.stop()


            db = SessionLocal()
            
            try:
                log_event(db, session_id, "click", name="abandon")
            finally:
                db.close()

            st.session_state.finish_status = "abandoned"
            st.session_state.phase = "survey"

            st.rerun()

        st.stop()

    
    # MAIN
    
    # If locked, DO NOT show parameter editors. Show only summary + quizzes (+ optional "Modify parameters" button).
    if st.session_state.lock_params_view and st.session_state.best_quiz_json:
        with st.expander("Current parameters summary", expanded=True):
            render_current_params_summary()

        # Quick way to unlock (equivalent to choosing "Modify parameters")
        if st.button("Modify parameters", key="btn_unlock_modify_params"):
            st.session_state.lock_params_view = False
            st.session_state.previous_quiz_json = None
            # do not wipe teacher vectors; user may want to start from them
            st.rerun()

        # Jump straight to quiz display section (no parameter UI)
        st.divider()

    else:
        # Editable parameters UI (only when not locked)
        st.subheader("Step 1. Configure your quiz")
        st.caption(
            "Choose the number of questions, topics, and difficulty levels as you would for a real course. "
            "Then click **Compose my quiz**."
        )

        st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Quiz parameters</div>", unsafe_allow_html=True)

        default_num = "" if st.session_state.num_mcqs is None else str(st.session_state.num_mcqs)
        num_str = st.text_input(
            "How many questions do you want your quiz to contain?",
            value=default_num,
            placeholder="Enter a number (3 to 30)",
        )
        requested_num_mcqs = parse_num_mcqs(num_str)

        if (num_str or "").strip() == "":
            st.warning("Please enter the number of questions (minimum 3).")
        elif requested_num_mcqs is None:
            st.warning("Number of questions must be an integer between 3 and 30.")

        with st.expander("Available topics for this dataset"):
            df_topics = pd.DataFrame(TOPICS, columns=["topic_id", "topic_name", "count"])
            df_topics = df_topics[["topic_name", "count"]].rename(
                columns={"topic_name": "Topic", "count": "Number of questions (available)"}
            )
            st.dataframe(df_topics, use_container_width=True, hide_index=True)

        teacherTopic, chosen_topics = topic_selector(TOPICS)
        topic_total = raw_topic_total_pct(chosen_topics)

        if len(chosen_topics) == 0:
            st.warning("Please choose at least one topic.")
        else:
            if topic_total == 0:
                st.warning("Topic percentages are all 0%. Please set topic percentages (total must be 100%).")
            elif topic_total < 100:
                st.warning(f"Topic percentages must sum to 100% (currently {topic_total}%).")
            elif topic_total > 100:
                st.warning(f"Be careful: topic percentages are above 100% (currently {topic_total}%).")

        teacherLevel = difficulty_selector(NUM_DIFFICULTIES)
        diff_total = raw_difficulty_total_pct(NUM_DIFFICULTIES)

        if diff_total == 0:
            st.warning("Please choose at least one difficulty level and set percentages (total must be 100%).")
        else:
            if diff_total < 100:
                st.warning(f"Difficulty percentages must sum to 100% (currently {diff_total}%).")
            elif diff_total > 100:
                st.warning(f"Be careful: difficulty percentages are above 100% (currently {diff_total}%).")

        can_compose = (
            requested_num_mcqs is not None
            and len(chosen_topics) > 0
            and topic_total == 100
            and diff_total == 100
        )

        generate_and_compose = st.button(
            "Compose my quiz",
            type="primary",
            disabled=not can_compose,
        )

        if not can_compose:
            reasons = []
            if requested_num_mcqs is None:
                if (num_str or "").strip() == "":
                    reasons.append("number of questions not set (minimum 3)")
                else:
                    reasons.append("number of questions must be between 3 and 30")
            if len(chosen_topics) == 0:
                reasons.append("no topic selected")
            elif topic_total != 100:
                reasons.append(f"topic distribution is {topic_total}% (must be 100%)")
            if diff_total != 100:
                reasons.append(f"difficulty distribution is {diff_total}% (must be 100%)")
            st.caption(" Compose disabled: " + ", ".join(reasons))

        if generate_and_compose:
            if requested_num_mcqs is None:
                st.error("Please enter a valid number of questions (3 to 30).")
                st.stop()

            if len(chosen_topics) == 0:
                st.error("Please select at least one topic.")
                st.stop()

            if topic_total != 100:
                if topic_total < 100:
                    st.error(f"Topic percentages must sum to 100% (currently {topic_total}%).")
                else:
                    st.error(f"Topic percentages must not exceed 100% (currently {topic_total}%).")
                st.stop()

            if diff_total != 100:
                if diff_total < 100:
                    st.error(f"Difficulty percentages must sum to 100% (currently {diff_total}%).")
                else:
                    st.error(f"Difficulty percentages must not exceed 100% (currently {diff_total}%).")
                st.stop()

            prev_num = st.session_state.num_mcqs
            st.session_state.num_mcqs = int(requested_num_mcqs)

            st.session_state.selected_topics = list(chosen_topics)
            st.session_state.last_teacherTopic = teacherTopic
            st.session_state.last_teacherLevel = teacherLevel

            # When user composes fresh, unlock parameters (normal flow)
            st.session_state.lock_params_view = False

            must_regen = (st.session_state.data_uuid is None) or (prev_num != st.session_state.num_mcqs)

            if must_regen:
                gen_payload = {
                    "MCQs": DEFAULT_MCQS,
                    "numQuizzes": DEFAULT_NUM_QUIZZES,
                    "numMCQs": int(st.session_state.num_mcqs),
                    "listTopics": [],
                    "numTopics": DEFAULT_NUM_TOPICS,
                    "topicMode": DEFAULT_TOPIC_MODE,
                    "levelMode": DEFAULT_LEVEL_MODE,
                    "orderLevel": DEFAULT_ORDER_LEVEL,
                }

                t0 = time.time()
                try:
                    with st.spinner("Generating quiz universe... This may take a minute."):
                        resp = api_post("/gen/quizzes", gen_payload)
                    rid, universe_path = parse_generation_response(resp)
                    dt_s = time.time() - t0

                    db = SessionLocal()
                    try:
                        log_event(db, session_id, "api_call", name="gen_quizzes", payload={"elapsed_s": dt_s})
                    finally:
                        db.close()

                    if not rid:
                        st.error("Generation succeeded but could not extract UUID from response.")
                        st.stop()

                    st.session_state.data_uuid = rid
                    st.session_state.universe_path = universe_path

                    st.session_state.best_quiz_json = None
                    st.session_state.best_mcqs = None
                    st.session_state.last_best_quiz_id = None
                    st.session_state.previous_quiz_json = None

                    for k in ("satisfied_choice", "next_step_choice"):
                        if k in st.session_state:
                            del st.session_state[k]

                except Exception as e:
                    db = SessionLocal()
                    try:
                        log_event(db, session_id, "error", name="gen_quizzes_failed", payload={"error": str(e)})
                    finally:
                        db.close()
                    st.error(f"Generation failed: {e}")
                    st.stop()

            compose_payload = {
                "dataUUID": st.session_state.data_uuid,
                "teacherTopic": st.session_state.last_teacherTopic,
                "teacherLevel": st.session_state.last_teacherLevel,
                "pathToModel": DEFAULT_MODEL_PATH,
                "alfaValue": DEFAULT_ALFA,
                "startQuizId": None,
            }

            t0 = time.time()
            try:
                with st.spinner("Composing your quiz... Please wait."):
                    resp = api_post("/compose/quiz", compose_payload)
                dt_s = time.time() - t0

                best_json = resp.get("best_quiz") if isinstance(resp, dict) else None
                st.session_state.best_quiz_json = best_json

                best_quiz_id = None
                target_match = None
                if best_json and "quiz_id" in best_json:
                    best_quiz_id = int(best_json["quiz_id"])
                    st.session_state.last_best_quiz_id = best_quiz_id

                if best_json and "targetMatch" in best_json:
                    try:
                        target_match = float(best_json["targetMatch"])
                    except Exception:
                        target_match = None

                st.session_state.best_mcqs = None
                if best_json:
                    st.session_state.best_mcqs = load_mcqs_for_best_quiz(best_json, st.session_state.data_uuid)

                mcq_ids, topic_cov, diff_cov = extract_result_vectors(best_json or {})

                db = SessionLocal()
                try:
                    a = ComposeAttempt(
                        session_id=session_id,
                        mode="fresh",
                        data_uuid=str(st.session_state.data_uuid),
                        start_quiz_id=None,
                        teacher_topic_json=to_json(st.session_state.last_teacherTopic),
                        teacher_level_json=to_json(st.session_state.last_teacherLevel),
                        num_mcqs=int(st.session_state.num_mcqs or 0),
                        mcq_ids_json=to_json(mcq_ids),
                        topic_coverage_json=to_json(topic_cov),
                        difficulty_coverage_json=to_json(diff_cov),
                        best_quiz_id=best_quiz_id,
                        target_match=target_match,
                        api_time_s=float(dt_s),
                        succeeded=True,
                        error=None,
                    )
                    db.add(a)
                    db.commit()
                finally:
                    db.close()

                for k in ("satisfied_choice", "next_step_choice"):
                    if k in st.session_state:
                        del st.session_state[k]

                st.session_state.same_quiz_repeat_count = 0
                st.session_state.previous_quiz_json = None
                st.rerun()

            except Exception as e:
                db = SessionLocal()
                try:
                    log_event(db, session_id, "error", name="compose_quiz_failed", payload={"error": str(e)})
                finally:
                    db.close()
                st.error(f"Composition failed: {e}")
                st.stop()

    
    # Quiz display 
    

    if st.session_state.best_quiz_json:
        st.divider()
        st.subheader("Step 2. Examine and refine the quiz")
        st.caption(
            "Review the generated quiz and decide whether it matches your intended topics and difficulty. "
            "You may refine the quiz or generate an improved version."
        )

        if st.session_state.same_quiz_repeat_count >= 2:
            st.warning(
                "**Same quiz returned multiple times**: The system couldn't find a better match with your current parameters. "
                "This is the best quiz available. You can try modifying your parameters to explore other options."
            )

        has_previous_quiz = st.session_state.previous_quiz_json is not None

        if has_previous_quiz:
            st.subheader("Compare quizzes")
            tab_old, tab_new = st.tabs(["Previous quiz", "New quiz"])

            total_q = int(st.session_state.num_mcqs or 0)

            with tab_old:
                st.write("**Previous quiz**")
                old_quiz = st.session_state.previous_quiz_json
                render_best_quiz(old_quiz, TOPICS)
                render_coverage(old_quiz, TOPICS, total_q)

                old_mcq_cols = sorted(
                    [k for k in old_quiz.keys() if k.startswith("mcq_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                if old_mcq_cols:
                    old_mcq_ids = [old_quiz[c] for c in old_mcq_cols]
                    try:
                        old_mcq_resp = api_post(
                            "/mcqs/by_ids",
                            {"ids": old_mcq_ids, "dataUUID": st.session_state.data_uuid},
                        )
                        old_mcqs = old_mcq_resp.get("items", [])
                        if old_mcqs:
                            render_mcq_list(old_mcqs)
                    except Exception:
                        pass

            with tab_new:
                st.write("**New quiz**")
                best = st.session_state.best_quiz_json
                render_best_quiz(best, TOPICS)
                render_coverage(best, TOPICS, total_q)
                if st.session_state.best_mcqs:
                    st.markdown("**Quiz questions**")
                    with st.container(border=True):
                        render_mcq_list(st.session_state.best_mcqs)
        else:
            best = st.session_state.best_quiz_json
            render_best_quiz(best, TOPICS)
            total_q = int(st.session_state.num_mcqs or 0)
            render_coverage(best, TOPICS, total_q)
            if st.session_state.best_mcqs:
                render_mcq_list(st.session_state.best_mcqs)

        st.divider()
        st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Are you satisfied with the quiz?</div>", unsafe_allow_html=True)

        sat = st.radio(
            label="",
            options=["Yes", "No"],
            index=None,
            horizontal=True,
            key="satisfied_choice",
            label_visibility="collapsed",
        )

        if sat == "Yes":
            db = SessionLocal()
            try:
                log_event(db, session_id, "click", name="satisfied_yes")
            finally:
                db.close()

            st.divider()
            st.markdown(
                "<div style='font-size:1.2rem; font-weight:600;'>Confirm your quiz</div>",
                unsafe_allow_html=True
            )

            st.info(
                "If you are satisfied with this quiz, you may confirm it to indicate that "
                "this is the version you are happy with."
)

            st.warning(
                " **Payment reminder**: To receive payment, you must generate at least one quiz "
                "and complete the final questionnaire."
            )

            col_validate, col_download = st.columns(2)
            with col_validate:
                if st.button("Confirm this quiz", key="btn_validate_quiz"):
                    st.success("Quiz confirmed. You may now proceed to the survey.")
                    st.session_state.quiz_validated = True
                    db = SessionLocal()
                    try:
                        log_event(
                            db,
                            session_id,
                            "click",
                            name="quiz_validated",
                            payload={"quiz_id": st.session_state.last_best_quiz_id},
                        )
                    finally:
                        db.close()

            with col_download:
                if st.button("Download quiz", key="btn_download_quiz"):
                    if st.session_state.best_mcqs:
                        import csv
                        import io

                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=st.session_state.best_mcqs[0].keys())
                        writer.writeheader()
                        writer.writerows(st.session_state.best_mcqs)
                        csv_content = output.getvalue()

                        st.download_button(
                            label="Click here to save quiz.csv",
                            data=csv_content,
                            file_name=f"quiz_{st.session_state.last_best_quiz_id}.csv",
                            mime="text/csv",
                            key="download_csv",
                        )
                    st.session_state.quiz_downloaded = True
                    db = SessionLocal()
                    try:
                        log_event(
                            db,
                            session_id,
                            "click",
                            name="quiz_downloaded",
                            payload={"quiz_id": st.session_state.last_best_quiz_id},
                        )
                    finally:
                        db.close()



            if st.button("Proceed to survey →", type="primary", key="btn_proceed_survey"):
                db = SessionLocal()
                try:
                    log_event(
                        db,
                        session_id,
                        "click",
                        name="proceed_to_survey",
                        payload={
                            "quiz_id": st.session_state.last_best_quiz_id,
                            "quiz_validated": st.session_state.get("quiz_validated", False),
                            "quiz_downloaded": st.session_state.get("quiz_downloaded", False),
                        },
                    )
                finally:
                    db.close()
                st.session_state.finish_status = "completed"
                st.session_state.phase = "survey"
                st.rerun()

        elif sat == "No":
            db = SessionLocal()
            try:
                log_event(db, session_id, "click", name="satisfied_no")
            finally:
                db.close()

            st.session_state.tool_page = "next_step"
            if "next_step_choice" in st.session_state:
                del st.session_state["next_step_choice"]
            st.rerun()

    st.divider()

    st.markdown(
        '<p style="color: #cc0000; font-weight: bold;">Note: Remember that you can end the study anytime by clicking the <strong>"End study now"</strong> button below.</p>',
        unsafe_allow_html=True,
    )

    abandon = st.button("End study now", key="btn_end_study_now")
    if abandon:
        if not has_minimum_interaction():
            st.warning(
                "To complete the study, you need to generate and review at least one quiz "
                "and complete the final questionnaire."
            )
            st.info(
                "If you still wish to exit now, the study will be recorded as incomplete."
            )

            #  Explicit confirmation button
            confirm_exit = st.button("Exit study anyway", key="confirm_exit_early")
            if not confirm_exit:
                st.stop()

        #  Either confirmed exit OR minimum interaction met
        db = SessionLocal()
        try:
            log_event(db, session_id, "click", name="abandon")
        finally:
            db.close()

        st.session_state.finish_status = "abandoned"
        st.session_state.phase = "survey"
        st.rerun()



# Phase 4: Survey

if st.session_state.phase == "survey":
    st.title("Post-study survey")
    st.caption("Step 3 of 3 • Post-study survey")

    st.subheader("Step 3. Post-study survey")
    st.caption(
        "Answer a short questionnaire about your experience using the quiz composition system."
    )

    session_id = int(st.session_state.study_session_id)
    status = st.session_state.get("finish_status", "abandoned")

    st.markdown("Select a number from **1 (lowest)** to **5 (highest)** for each statement.")

    def likert(q: str, key: str) -> Optional[int]:
        v = st.radio(q, [1, 2, 3, 4, 5], horizontal=True, index=None, key=key)
        return int(v) if v is not None else None

    q1 = likert(
        "(Q1) Feeling of Accomplishment: How successful do you feel you were in building a quiz that matches your needs (topics and difficulty) during the session?",
        "q1",
    )
    q2 = likert(
        "(Q2) Effort Required: How much effort was required to inspect the proposed quizzes and decide whether to keep or change them?",
        "q2",
    )
    q3 = likert(
        "(Q3) Mental Demand: How mentally demanding was it to read and evaluate the candidate quizzes?",
        "q3",
    )
    q4 = likert(
        "(Q4) Perceived Controllability: How much control did you feel you had over the final quiz?",
        "q4",
    )
    q5 = likert(
        "(Q5) Temporal Demand: How time-pressured did you feel while composing the quiz within the allotted time?",
        "q5",
    )
    q6 = likert(
        "(Q6) Satisfaction: Overall, how satisfied are you with the final quiz you produced?",
        "q6",
    )
    st.divider()

    q7_response = st.radio(
        "(Q7) Would you use this system again?",
        ["Yes", "No"],
        index=None,
        horizontal=True,
        key="q7",
        label_visibility="visible",
    )
    q7 = 1 if q7_response == "Yes" else (0 if q7_response == "No" else None)
    comments = st.text_area("Optional comments (what worked / what didn’t, suggestions)", value="", height=120)

    if st.button("Submit survey", type="primary"):
        missing = [
            name
            for name, v in [
                ("Q1", q1),
                ("Q2", q2),
                ("Q3", q3),
                ("Q4", q4),
                ("Q5", q5),
                ("Q6", q6),
                ("Q7", q7),
            ]
            if v is None
        ]
        if missing:
            st.error(f"Please answer all questions before submitting: {', '.join(missing)}")
            st.stop()

        db = SessionLocal()
        try:
            existing = db.query(SurveyResponse).filter(SurveyResponse.session_id == session_id).one_or_none()
            if existing is not None:
                db.delete(existing)
                db.commit()

            sr = SurveyResponse(
                session_id=session_id,
                q1_accomplishment=int(q1),
                q2_effort=int(q2),
                q3_mental_demand=int(q3),
                q4_controllability=int(q4),
                q5_temporal_demand=int(q5),
                q6_satisfaction_trust=int(q6),
                q7_would_use_again=int(q7),
                comments=comments.strip() or None,
            )
            db.add(sr)
            db.commit()
            log_event(db, session_id, "survey", name="submitted", payload={"status": status})
        finally:
            db.close()

        code = end_session(session_id, status=status)
        st.session_state.completion_code = code
        st.session_state.phase = "done"
        st.rerun()



# Phase 5: Done

if st.session_state.phase == "done":
    st.title("Thank you!")
    code = st.session_state.get("completion_code", "N/A")
    status = st.session_state.get("finish_status", "abandoned")

    if status == "completed":
        st.success("Thank you for completing the study. You are being redirected to Prolific.")
        st.markdown(
            """
            <meta http-equiv="refresh" content="2;url=https://app.prolific.com/submissions/complete?cc=C1GWM6YM">
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <p>If you are not redirected automatically, please click
            <a href="https://app.prolific.com/submissions/complete?cc=C1GWM6YM">
            here to return to Prolific</a>.
            </p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("You ended the study early.")

  