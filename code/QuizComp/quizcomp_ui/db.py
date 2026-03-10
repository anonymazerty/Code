# quizcomp_ui/db.py
import os
import json
import datetime as dt
from typing import Any, Optional, Dict

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# SQLite path 
DB_PATH = os.environ.get("QUIZCOMP_DB_PATH", "quizcomp_study.sqlite")
DB_URL = os.environ.get("QUIZCOMP_DB_URL", f"sqlite:///{DB_PATH}")

Base = declarative_base()

engine = create_engine(
    DB_URL,
    future=True,
    connect_args={"check_same_thread": False},  # needed for Streamlit
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_json(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return json.dumps(str(x), ensure_ascii=False)



# Tables

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)

    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    sessions = relationship("StudySession", back_populates="user")


class StudySession(Base):
    """
    One participation session (often 1 per Prolific participant).
    """
    __tablename__ = "study_sessions"
    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="sessions")

    prolific_pid = Column(String(120), nullable=True, index=True)
    prolific_study_id = Column(String(120), nullable=True)
    prolific_session_id = Column(String(120), nullable=True)

    consented = Column(Boolean, default=False, nullable=False)

    started_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    status = Column(String(32), default="in_progress", nullable=False)
    # allowed values: in_progress, completed, abandoned

    completion_code = Column(String(64), nullable=True)

    # “high-level” timings (seconds)
    total_time_s = Column(Float, nullable=True)

    events = relationship("EventLog", back_populates="session")
    compose_attempts = relationship("ComposeAttempt", back_populates="session")
    survey = relationship("SurveyResponse", back_populates="session", uselist=False)


class EventLog(Base):
   
    __tablename__ = "event_logs"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False, index=True)
    session = relationship("StudySession", back_populates="events")

    ts = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    event_type = Column(String(64), nullable=False)   # "page_view", "click", "api_call", "error"
    name = Column(String(128), nullable=True)
    payload_json = Column(Text, nullable=True)        # JSON string


class ComposeAttempt(Base):
    
    __tablename__ = "compose_attempts"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False, index=True)
    session = relationship("StudySession", back_populates="compose_attempts")

    ts = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    mode = Column(String(16), nullable=False)  # "fresh" or "improve"

    data_uuid = Column(String(64), nullable=True)  # universe UUID you already use
    start_quiz_id = Column(Integer, nullable=True)

    # Teacher targets (vectors)
    teacher_topic_json = Column(Text, nullable=True)   # JSON list[float]
    teacher_level_json = Column(Text, nullable=True)   # JSON list[float]

    # Returned quiz info (what you want)
    num_mcqs = Column(Integer, nullable=True)
    mcq_ids_json = Column(Text, nullable=True)             # JSON list[int]
    topic_coverage_json = Column(Text, nullable=True)      # JSON list[float]
    difficulty_coverage_json = Column(Text, nullable=True) # JSON list[float]

    best_quiz_id = Column(Integer, nullable=True)
    target_match = Column(Float, nullable=True)

    api_time_s = Column(Float, nullable=True)  # elapsed for compose call
    succeeded = Column(Boolean, default=True, nullable=False)
    error = Column(Text, nullable=True)


class SurveyResponse(Base):
    __tablename__ = "survey_responses"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False, unique=True, index=True)
    session = relationship("StudySession", back_populates="survey")

    ts = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    q1_accomplishment = Column(Integer, nullable=False)
    q2_effort = Column(Integer, nullable=False)
    q3_mental_demand = Column(Integer, nullable=False)
    q4_controllability = Column(Integer, nullable=False)
    q5_temporal_demand = Column(Integer, nullable=False)
    q6_satisfaction_trust = Column(Integer, nullable=False)
    q7_would_use_again = Column(Integer, nullable=False)  # 1 for yes, 0 for no

    comments = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def log_event(
    db,
    session_id: int,
    event_type: str,
    name: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    ev = EventLog(
        session_id=session_id,
        event_type=event_type,
        name=name,
        payload_json=to_json(payload) if payload is not None else None,
    )
    db.add(ev)
    db.commit()
