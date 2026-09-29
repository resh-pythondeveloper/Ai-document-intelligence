# AI Document Intelligence

Upload PDFs, index them, and ask questions with page-level citations.
Backend: FastAPI + Qdrant + PostgreSQL + Gemini (LangChain / LangGraph). Frontend: React (Vite).

## Run

```bash
cp .env.
docker compose up -d            # PostgreSQL + Qdrant
uv sync
uv run alembic upgrade head
uv run fastapi dev app/main.py  # API on :8000, docs at /docs

cd frontend && npm install && npm run dev   # UI on :5173
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | /api/v1/documents | List uploaded documents |
| POST | /api/v1/documents/upload | Upload a PDF |
| POST | /api/v1/documents/{id}/ingest | Chunk, embed and index a PDF |
| POST | /api/v1/documents/chat | Standard RAG chat with saved history |
| POST | /api/v1/agentic-rag/ | Agentic RAG (routing, grading, query rewrite) |
