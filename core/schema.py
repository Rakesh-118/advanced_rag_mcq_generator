from pydantic import BaseModel, Field
from typing import Dict, List


class MCQ(BaseModel):
    question: str = Field(..., min_length=5)
    options: Dict[str, str]
    answer: str
    explanation: str

    class Config:
        extra = "forbid"


class MCQList(BaseModel):
    mcqs: List[MCQ]

    class Config:
        extra = "forbid"
