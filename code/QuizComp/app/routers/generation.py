# datagems/app/routers/generation.py
from uuid import uuid4
from fastapi import APIRouter, HTTPException

from app.schemas.generation import QuizGenerationRequest
from app.services.generation import generate_quiz_universe

router = APIRouter(prefix="/gen", tags=["generation"])


@router.post("/quizzes")
def generate_quizzes_endpoint(req: QuizGenerationRequest):
    """
    Generates a quiz universe + quizzes CSV under data/<uuid>/...
    Returns:
      {
        "RequestID": "<uuid>",
        "PathToQuizzes": "data/<uuid>/universe_<uuid>.json"
      }
    """
    try:
        run_uuid = uuid4()
        path = generate_quiz_universe(req, run_uuid)

        return {
            "RequestID": str(run_uuid),
            "PathToQuizzes": str(path),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
