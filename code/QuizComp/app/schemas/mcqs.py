from pydantic import BaseModel
from typing import List, Optional

class MCQByIdsRequest(BaseModel):
    ids: List[int]
    dataUUID: str  # where the generated MCQs live
    csv_path: Optional[str] = None  # optional override