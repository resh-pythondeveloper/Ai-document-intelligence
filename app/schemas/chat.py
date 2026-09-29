from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: int | None = None
    document_id: int | None = Field(
        default=None,
        ge=1,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class ChatSource(BaseModel):
    chunk_id: str
    document_id: int
    filename: str
    page_number: int
    score: float


class ChatResponse(BaseModel):
    question: str
    session_id: int
    answer: str
    sources: list[ChatSource]