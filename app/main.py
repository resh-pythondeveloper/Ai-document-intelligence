from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agentic_rag import router as agentic_rag_router
from app.api.documents import router as documents_router
from app.core.config import CORS_ORIGINS

app = FastAPI(title="AI Document Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(agentic_rag_router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
