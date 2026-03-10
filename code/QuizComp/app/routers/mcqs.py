from fastapi import APIRouter
from app.schemas.mcqs import MCQByIdsRequest
from app.services.mcqs import get_mcqs_by_ids

router = APIRouter(tags=["mcqs"], prefix="/mcqs")

@router.post("/by_ids")
async def by_ids(req: MCQByIdsRequest):
    return get_mcqs_by_ids(req)
