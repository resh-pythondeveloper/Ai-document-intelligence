from pathlib import Path

from google.genai import errors as genai_errors
from uuid import uuid4
from app.core.config import MAX_UPLOAD_MB
from app.db.postgres import get_db
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from app.services.ingestion_service import IngestionService
from app.models.document import Document
from app.schemas.document import DocumentResponse
from app.services.pdf_service import PDFService
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
)

from app.services.search_service import SearchService
from app.services.qdrant_service import QdrantService
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 1. Validate file type
    original_name = Path(file.filename or "document.pdf").name  # strips any ../ path parts
    if file.content_type != "application/pdf" or not original_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    # 2. Read with a size limit
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is larger than {MAX_UPLOAD_MB} MB.",
        )
    if not contents.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not a valid PDF.",
        )

    # 3. Save PDF under a unique name
    file_path = UPLOAD_DIR / f"{uuid4().hex}_{original_name}"
    file_path.write_bytes(contents)

    # 4. Save metadata in PostgreSQL
    document = Document(
        filename=original_name,
        title=Path(original_name).stem,
        file_path=str(file_path),
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document

@router.get("", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Delete vectors first. If Qdrant is unavailable, keep the
    # PostgreSQL record so the document is not left half-deleted.
    try:
        vectors_deleted = QdrantService().delete_by_document_id(
            document_id=document.id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not delete document vectors from Qdrant.",
        ) from exc

    # Delete the physical PDF if it still exists.
    file_path = Path(document.file_path)
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError as exc:
            # Qdrant vectors were already deleted, so do not delete the
            # PostgreSQL record if the PDF cannot be removed.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vectors were deleted, but the PDF file could not be removed.",
            ) from exc

    # Finally remove the document metadata from PostgreSQL.
    db.delete(document)
    db.commit()

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "vectors_deleted": vectors_deleted,
    }


@router.get("/{document_id}/chunks")
def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    pdf_service = PDFService()

    pages = pdf_service.extract_pages(document.file_path)

    chunks = pdf_service.create_chunks(
        pages=pages,
        document_id=document.id,
        filename=document.filename,
    )

    return {
        "document_id": document.id,
        "filename": document.filename,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

@router.post("/{document_id}/ingest")
def ingest_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    ingestion_service = IngestionService()

    result = ingestion_service.process_document(
        document_id=document.id,
        filename=document.filename,
        file_path=document.file_path,
    )

    return {
        "message": "Document indexed successfully",
        "document_id": document.id,
        **result,
    }

@router.post(
    "/search",
    response_model=SearchResponse,
)
def semantic_search(
    request: SearchRequest,
):

    search_service = SearchService()

    results = search_service.search(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "results": results,
    }

@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):

    try:
        result = RAGService().chat(
            db=db,
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            session_id=request.session_id,
        )
    except ValueError as exc:  # unknown session_id
        raise HTTPException(status_code=404, detail=str(exc))
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            raise HTTPException(429, "Gemini quota exceeded. Try again later.")
        raise HTTPException(502, "The AI provider rejected the request.")

    return {
        "question": request.question,
        "session_id": result["session_id"],
        "answer": result["answer"],
        "sources": result["sources"],
    }
