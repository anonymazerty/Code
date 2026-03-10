# quizcomp_ui/completion_code_allocator.py
import os
import json
import datetime as dt
from typing import Optional

CODES_JSON_PATH = os.environ.get(
    "QUIZCOMP_CODES_JSON",
    os.path.join(os.path.dirname(__file__), "completion_codes.json")
)

def _now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def allocate_code(session_id: int) -> str:
    """
    Allocate one unused code from completion_codes.json and mark it as used.
    Returns the allocated code, or "NO-CODE-LEFT" if none available.
    """

    # Read file
    with open(CODES_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    used = data.get("used", {}) or {}

    # Find first unused
    chosen: Optional[str] = None
    for c in codes:
        if c not in used:
            chosen = c
            break

    if chosen is None:
        return "NO-CODE-LEFT"

    # Mark used
    used[chosen] = {"session_id": int(session_id), "ts": _now_iso()}
    data["used"] = used

    # Write back
    with open(CODES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return chosen
