import time
import streamlit as st
from completion_codes_allocator import allocate_code

from db import SessionLocal, StudySession, log_event, now_utc


def get_prolific_params():
    """
    Prolific typically passes these as query parameters:
    PROLIFIC_PID, STUDY_ID, SESSION_ID
    """
    qp = st.query_params
    return {
        "prolific_pid": qp.get("PROLIFIC_PID", None),
        "prolific_study_id": qp.get("STUDY_ID", None),
        "prolific_session_id": qp.get("SESSION_ID", None),
    }



# Landing page (overview only)


def study_landing_page():
    st.title("Interactive Math Quiz Composition")

    st.markdown(
        """
In this study, you will use a tool that automatically generates math quizzes
based on selected topics and difficulty levels. The goal is to understand how
people evaluate and refine generated quizzes.
"""
    )

    st.markdown(
        """
**What you will do**
- Configure quiz parameters  
- Review and optionally refine a generated quiz  
- Complete a short questionnaire  
"""
    )

    st.markdown(
        """
You do not need to solve the questions. You only need to judge whether the quiz
matches the intended topics and difficulty.
"""
    )

    st.markdown(
        """
**Time and payment:** The study takes approximately 20–25 minutes. To receive
compensation, you must generate at least one quiz and complete the final questionnaire.
"""
    )

    st.markdown(
        """
<p style="color:#b91c1c; font-weight:600;">
Participation is voluntary. You may end the study at any time.
</p>
""",
        unsafe_allow_html=True,
    )



# Consent page

def consent_block() -> bool:
    #st.subheader("Consent and data use")

    st.markdown(
        """
This study is conducted for **research purposes** to examine how people interact
with a quiz composition system.

**During the session, we will record:**
- the quiz parameters you choose (e.g., topics and difficulty distributions),
- the quizzes generated and selected by the system,
- basic interaction events required to run the study (e.g., button clicks and phase transitions),
- your responses to a short post-study questionnaire.

**Privacy**
- No personally identifying information is collected beyond your **Prolific ID**.
- Your Prolific ID is used only to manage participation and compensation.
- Your Prolific ID will not be linked to any published results.

**Voluntary participation**
- Your participation is voluntary, and you may stop at any time.
- To be considered a completed study, at least one quiz interaction and the final questionnaire must be completed.
- If you stop early, partial data may still be recorded and the session may be marked as incomplete.
"""
    )

    agreed = st.checkbox(
        "I consent to participate in this study and to the recording of my interactions and responses for research purposes.",
        value=False,
    )
    return bool(agreed)



# Study session management

def ensure_study_session(user_id: int) -> int:
    """
    Ensure exactly ONE StudySession per PROLIFIC_PID.
    - Resume if in_progress
    - Block if abandoned or completed
    - Restore correct phase (consent vs tool)
    """
    # Fast path: already loaded in this Streamlit run
    if st.session_state.get("study_session_id"):
        return int(st.session_state.study_session_id)

    params = get_prolific_params()
    prolific_pid = params["prolific_pid"]

    if not prolific_pid:
        st.error("Invalid study entry. Please return to Prolific.")
        st.stop()

    db = SessionLocal()
    try:
        # Look for most recent session for this participant
        existing = (
            db.query(StudySession)
            .filter(StudySession.prolific_pid == prolific_pid)
            .order_by(StudySession.started_at.desc())
            .first()   # <-- IMPORTANT: not one_or_none()
        )

        if existing:
            # Hard stop if already finished
            if existing.status in ("completed", "abandoned"):
                st.error(
                    "You have already participated in this study. "
                    "Please return to Prolific."
                )
                st.stop()

            # Resume in-progress session
            st.session_state.study_session_id = existing.id
            st.session_state.user_id = existing.user_id
            st.session_state.study_started_perf = time.time()

            # Restore correct phase
            if existing.consented:
                st.session_state.phase = "tool"
            else:
                st.session_state.phase = "consent"

            log_event(
                db,
                existing.id,
                "page_view",
                name="study_session_resumed",
                payload=params,
            )

            return existing.id

        # Create first (and only) session
        ss = StudySession(
            user_id=user_id,
            prolific_pid=prolific_pid,
            prolific_study_id=params["prolific_study_id"],
            prolific_session_id=params["prolific_session_id"],
            consented=False,
            status="in_progress",
            started_at=now_utc(),
        )

        db.add(ss)
        db.commit()
        db.refresh(ss)

        st.session_state.study_session_id = ss.id
        st.session_state.study_started_perf = time.time()

        log_event(
            db,
            ss.id,
            "page_view",
            name="study_session_created",
            payload=params,
        )

        return ss.id

    finally:
        db.close()

def mark_consented(session_id: int):
    db = SessionLocal()
    try:
        ss = db.query(StudySession).filter(StudySession.id == session_id).one()
        ss.consented = True
        db.commit()

        log_event(
            db,
            session_id,
            "state",
            name="consented",
            payload={"consented": True},
        )
    finally:
        db.close()


def end_session(session_id: int, status: str) -> str:
    """
    Ends the study session.

    status: "completed" or "abandoned"
    Returns a completion code.
    """
    db = SessionLocal()
    try:
        ss = db.query(StudySession).filter(StudySession.id == session_id).one()

        ss.status = status
        ss.ended_at = now_utc()

        started = st.session_state.get("study_started_perf", None)
        if started is not None:
            ss.total_time_s = float(time.time() - float(started))

        if not ss.completion_code:
            ss.completion_code = allocate_code(ss.id)

        db.commit()

        log_event(
            db,
            session_id,
            "state",
            name="ended",
            payload={
                "status": status,
                "completion_code": ss.completion_code,
            },
        )

        return ss.completion_code
    finally:
        db.close()