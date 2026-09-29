from google import genai
from google.genai import types

from app.core.config import GEMINI_API_KEY

BATCH_SIZE = 50  # Gemini accepts up to 100 texts per request


class EmbeddingService:

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = "gemini-embedding-001"

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts[start:start + BATCH_SIZE],
                config=types.EmbedContentConfig(
                    output_dimensionality=768,
                    task_type=task_type,
                ),
            )
            vectors.extend(e.values for e in result.embeddings)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def embed_documents(self, chunks: list[dict]) -> list[dict]:
        """Embed all chunks in batches (one API call per 50 chunks)."""
        vectors = self._embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
        return [{**c, "vector": v} for c, v in zip(chunks, vectors)]
