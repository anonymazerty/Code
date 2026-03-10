from fastapi import APIRouter
from app.services.composition import compose_quiz
from app.schemas.composition import QuizCompositionRequest
import uuid

router = APIRouter(tags=["compose"], prefix="/compose")

@router.post("/quiz")
async def compose(req: QuizCompositionRequest):
    request_id = str(uuid.uuid4())
    result = compose_quiz(req, request_id)  # now returns dict

    # Ensure RequestID is always returned
    if isinstance(result, dict):
        result["RequestID"] = request_id
        return result

    # fallback if older return type
    return {"PathToQuiz": result, "RequestID": request_id}
