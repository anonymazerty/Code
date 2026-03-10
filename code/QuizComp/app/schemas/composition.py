from pydantic import BaseModel, Field
from typing import List
from typing import Optional

class QuizCompositionRequest(BaseModel):
    dataUUID: str = Field(..., description="Unique identifier of the data source")
    teacherTopic: List[float] = Field(..., description="A list of floats, each corresponding to a topic and representing its percentage participation in the quiz")
    teacherLevel: List[float] = Field(..., description="A list of floats, each corresponding to a level and representing its percentage participation in the quiz")
    pathToModel: str = Field(..., description="Path to the model to be used for quiz composition")
    alfaValue: float = Field(0.5, description="Represents the weight of the teacher's preferences in the composition process. Must be one of [0, 0.25, 0.5, 0.75, 1]")
    startQuizId: Optional[int] = None

    @classmethod
    def __get_validators__(cls):
        yield from super().__get_validators__()
        yield cls.validate_alfa_value

    @staticmethod
    def validate_alfa_value(values):
        allowed = [0, 0.25, 0.5, 0.75, 1]
        if 'alfaValue' in values and values['alfaValue'] not in allowed:
            raise ValueError(f"alfaValue must be one of {allowed}")
        return values