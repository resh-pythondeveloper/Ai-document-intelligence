from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    score: float
    chunk_id: str
    document_id: int
    filename: str
    page_number: int
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]