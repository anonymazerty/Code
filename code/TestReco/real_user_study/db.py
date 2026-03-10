import os
import json
import datetime as dt
from typing import Any
from sqlalchemy import text, event


from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DB_PATH = os.environ.get("REALUSER_DB_PATH", "real_user_study.sqlite")
DB_URL = os.environ.get("REALUSER_DB_URL", f"sqlite:///{DB_PATH}")

Base = declarative_base()
engine = create_engine(DB_URL, future=True)
# SQLite does NOT enforce foreign keys unless enabled per connection.
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)

    username = Column(String(64), nullable=False, unique=True)
    password_hash = Column(String(256), nullable=False)

    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    sessions = relationship("StudySession", back_populates="user")


class StudySession(Base):
    __tablename__ = "study_sessions"
    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic = Column(String(128), nullable=False)
    benchmark = Column(String(64), nullable=False)
    orchestrator_type = Column(String(64), nullable=False)  # "tool_call"
    model_name = Column(String(128), nullable=False)  # "chatgpt-4o-mini"

    status = Column(String(32), nullable=False, default="created")  # created|pretest|learning|survey|completed|abandoned
    started_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Calibration/pretest summary
    pretest_num_questions = Column(Integer, default=10, nullable=False)
    pretest_correct = Column(Integer, default=0, nullable=False)
    pretest_mastery_init = Column(Float, nullable=True)

    # Learning loop config
    total_steps_planned = Column(Integer, default=10, nullable=False)
    questions_per_step = Column(Integer, default=5, nullable=False)

    # Final
    final_mastery = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="sessions")
    attempts = relationship("Attempt", back_populates="session")
    steps = relationship("LearningStep", back_populates="session")
    orchestrator_calls = relationship("OrchestratorCall", back_populates="session")
    survey = relationship("SurveyResponse", back_populates="session", uselist=False)

    # if abandoned, store where it happened 
    abandoned_phase = Column(String(32), nullable=True)
    abandoned_at = Column(DateTime(timezone=True), nullable=True)


class LearningStep(Base):
    __tablename__ = "learning_steps"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False)
    step_index = Column(Integer, nullable=False)  # 0..N-1

    action = Column(Integer, nullable=False)  # 0/1/2 from env action space
    selected_question_indices_json = Column(Text, nullable=False)  # env indices as JSON
    selected_question_dataset_ids_json = Column(Text, nullable=False)  # dataset "id" as JSON

    # Snapshot metrics
    mastery_before = Column(Float, nullable=True)
    mastery_after = Column(Float, nullable=True)
    rolling_accuracy = Column(Float, nullable=True)
    reward_perf = Column(Float, nullable=True)
    reward_gap = Column(Float, nullable=True)
    reward_apt = Column(Float, nullable=True)

    started_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    session = relationship("StudySession", back_populates="steps")
    __table_args__ = (UniqueConstraint("session_id", "step_index", name="uq_step_per_session"),)


class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False)

    phase = Column(String(16), nullable=False)  # "pretest" | "learning"
    step_index = Column(Integer, nullable=True)  # null for pretest, else 0..N-1

    question_index = Column(Integer, nullable=False)  # env index
    dataset_question_id = Column(String(64), nullable=False)  # original QA.json "id" string
    topic = Column(String(128), nullable=False)

    original_difficulty = Column(Float, nullable=True)
    scaled_difficulty = Column(Float, nullable=True)

    chosen_option_index = Column(Integer, nullable=False)
    correct_option_index = Column(Integer, nullable=False)
    is_correct = Column(Boolean, nullable=False)

    started_at = Column(DateTime(timezone=True), nullable=False)
    answered_at = Column(DateTime(timezone=True), nullable=False)
    response_time_ms = Column(Integer, nullable=False)

    session = relationship("StudySession", back_populates="attempts")


class OrchestratorCall(Base):
    __tablename__ = "orchestrator_calls"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False)
    step_index = Column(Integer, nullable=True)  # null for pretest, else step index

    state_obs_json = Column(Text, nullable=False)
    selected_strategy = Column(String(128), nullable=True)
    action = Column(Integer, nullable=True)

    latency_s = Column(Float, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    raw_action_info_json = Column(Text, nullable=True)
    raw_orchestrator_info_json = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    session = relationship("StudySession", back_populates="orchestrator_calls")


class SurveyResponse(Base):
    __tablename__ = "survey_responses"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False, unique=True)

    # Survey fields (1..5 Likert)
    accomplishment = Column(Integer, nullable=False)  # Q1: Feeling of Accomplishment
    effort_required = Column(Integer, nullable=False)  # Q2: Effort Required
    mental_demand = Column(Integer, nullable=False)  # Q3: Mental Demand
    perceived_controllability = Column(Integer, nullable=False)  # Q4: Perceived Controllability
    temporal_demand = Column(Integer, nullable=False)  # Q5: Temporal Demand
    frustration = Column(Integer, nullable=False)  # Q6: Frustration
    trust = Column(Integer, nullable=False)  # Q7: Trust

    would_use_again = Column(Boolean, nullable=False)
    free_text = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    session = relationship("StudySession", back_populates="survey")

def _ensure_column(engine, table: str, col: str, coltype_sql: str):
    with engine.begin() as conn:
        cols = [r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]
        if col not in cols:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {coltype_sql}"))

def _ensure_index(engine, index_name: str, create_sql: str):
    """
    Create an index if it does not exist (SQLite-safe).
    create_sql must be a full CREATE INDEX statement using the same index_name.
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))

def init_db():
    Base.metadata.create_all(engine)

    # Migration for older sqlite files (columns)
    _ensure_column(engine, "study_sessions", "abandoned_phase", "VARCHAR(32)")
    _ensure_column(engine, "study_sessions", "abandoned_at", "DATETIME")

    # Migration: indexes (safe on existing DBs)
    # Sessions
    _ensure_index(
        engine,
        "ix_study_sessions_user_id",
        "CREATE INDEX IF NOT EXISTS ix_study_sessions_user_id ON study_sessions(user_id)",
    )
    _ensure_index(
        engine,
        "ix_study_sessions_status",
        "CREATE INDEX IF NOT EXISTS ix_study_sessions_status ON study_sessions(status)",
    )
    _ensure_index(
        engine,
        "ix_study_sessions_started_at",
        "CREATE INDEX IF NOT EXISTS ix_study_sessions_started_at ON study_sessions(started_at)",
    )

    # Attempts
    _ensure_index(
        engine,
        "ix_attempts_session_phase",
        "CREATE INDEX IF NOT EXISTS ix_attempts_session_phase ON attempts(session_id, phase)",
    )
    _ensure_index(
        engine,
        "ix_attempts_session_step",
        "CREATE INDEX IF NOT EXISTS ix_attempts_session_step ON attempts(session_id, step_index)",
    )
    _ensure_index(
        engine,
        "ix_attempts_question",
        "CREATE INDEX IF NOT EXISTS ix_attempts_question ON attempts(question_index)",
    )

    # Learning steps
    _ensure_index(
        engine,
        "ix_learning_steps_session",
        "CREATE INDEX IF NOT EXISTS ix_learning_steps_session ON learning_steps(session_id)",
    )

    # Orchestrator calls
    _ensure_index(
        engine,
        "ix_orchestrator_calls_session_step",
        "CREATE INDEX IF NOT EXISTS ix_orchestrator_calls_session_step ON orchestrator_calls(session_id, step_index)",
    )

    # Survey responses
    _ensure_index(
        engine,
        "ix_survey_responses_session",
        "CREATE INDEX IF NOT EXISTS ix_survey_responses_session ON survey_responses(session_id)",
    )
# -----------------------
# JSON helpers (FIX HERE)
# -----------------------

def _to_jsonable(obj: Any) -> Any:
    """
    Convert common non-JSON-serializable objects (numpy, datetime, sets, bytes, etc.)
    into JSON-friendly Python types.
    """
    # numpy (optional)
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None:
        # ndarray -> list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy scalar -> python scalar
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()

    # datetime -> ISO string
    if isinstance(obj, (dt.datetime, dt.date)):
        # Keep timezone if present
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    # bytes -> decode best-effort
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)

    # dict / list / tuple / set
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_jsonable(v) for v in sorted(obj)]

    # fallback: keep primitives, stringify unknown objects
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # last resort
    return str(obj)


def to_json(obj: Any) -> str:
    """
    Robust JSON serialization for DB storage.
    """
    return json.dumps(_to_jsonable(obj), ensure_ascii=False)
