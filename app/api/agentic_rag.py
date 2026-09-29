from functools import lru_cache

from fastapi import APIRouter, HTTPException
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from app.services.agentic_rag import AgenticRAGService

router = APIRouter(prefix="/api/v1/agentic-rag", tags=["Agentic RAG"])


class AgenticRequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_id: int | None = Field(default=None, ge=1)


@lru_cache
def get_service() -> AgenticRAGService:
    # Build the LLM clients and LangGraph once, not on every request
    return AgenticRAGService()


@router.post("/")
def agentic_rag(request: AgenticRequest):
    try:
        result = get_service().graph.invoke(
            {
                "original_question": request.question,
                "search_question": request.question,
                "document_id": request.document_id,
                "route": "",
                "documents": [],
                "relevant_documents": [],
                "sources": [],
                "context": "",
                "answer": "",
                "rewritten_question": "",
                "retry_count": 0,
            }
        )
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            raise HTTPException(429, "Gemini quota exceeded. Try again later.")
        raise HTTPException(502, "The AI provider rejected the request.")

    return {
        "question": request.question,
        "route": result.get("route"),
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
        "rewritten_question": result.get("rewritten_question") or None,
        "retry_count": result.get("retry_count", 0),
        "retrieved_count": len(result.get("documents", [])),
        "relevant_count": len(result.get("relevant_documents", [])),
    }
