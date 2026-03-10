from pydantic import BaseModel, Field
from typing import List


class QuizGenerationRequest(BaseModel):
    MCQs: List[str] = Field(..., description="List of csv paths")
    numQuizzes: int = Field(..., ge=1)
    numMCQs: int = Field(..., ge=1)
    listTopics: List[str] = Field(default_factory=list)

    numTopics: int = Field(..., ge=0)
    numDifficulties: int = Field(default=6, ge=1)
    topicMode: int = 1
    levelMode: int = 1
    orderLevel: int = 2
