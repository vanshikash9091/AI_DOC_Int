
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentRecord(BaseModel):
    id: int = Field(gt=0)
    filename: str = Field(min_length=1)
    file_type: str
    page_count: int | None = Field(default=None, ge=1)
    word_count: int = Field(ge=1)
    summary: str = Field(min_length=1)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    count: int = Field(ge=0)
    documents: list[DocumentRecord]


class DocumentSummaryResponse(BaseModel):
    message: str
    document_id: int
    filename: str
    file_type: str
    page_count: int | None
    word_count: int
    summary: str


class ErrorResponse(BaseModel):
    detail: str


class DocumentQuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class DocumentQuestionResponse(BaseModel):
    document_id: int
    filename: str
    question: str
    answer: str
    confidence: float = Field(ge=0, le=1)
    source_passage: str | None
