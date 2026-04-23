from pydantic import BaseModel, Field
from typing import List


class Source(BaseModel):
    document: str
    page: int
    chunk: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float = Field(..., ge=0, le=1)