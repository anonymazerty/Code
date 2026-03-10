# quizcomp_ui/auth.py
from typing import Optional
from passlib.hash import pbkdf2_sha256

from db import SessionLocal, User


def hash_password(pw: str) -> str:
    return pbkdf2_sha256.hash(pw)


def verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return pbkdf2_sha256.verify(pw, pw_hash)
    except Exception:
        return False


def create_user(username: str, password: str) -> User:
    username = (username or "").strip()
    if not username:
        raise ValueError("Username is required.")

    # remove all password constraints
    password = password or ""

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).one_or_none()
        if existing is not None:
            raise ValueError("This username is already taken.")

        u = User(username=username, password_hash=hash_password(password))
        db.add(u)
        db.commit()
        db.refresh(u)
        return u
    finally:
        db.close()


def authenticate(username: str, password: str) -> Optional[User]:
    username = (username or "").strip()
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == username).one_or_none()
        if u is None:
            return None
        if not verify_password(password, u.password_hash):
            return None
        return u
    finally:
        db.close()
