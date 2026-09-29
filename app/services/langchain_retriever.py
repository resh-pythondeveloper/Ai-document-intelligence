from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient, models

from app.core.config import GEMINI_API_KEY, QDRANT_COLLECTION, QDRANT_URL


class LangChainRetriever:
    """Searches Qdrant directly.

    Chunks are stored with a flat payload (document_id, filename, page_number,
    text). LangChain's QdrantVectorStore expects metadata nested under a
    "metadata" key, so it would return documents with empty metadata (no
    filename/page in citations). Building Documents here avoids that.
    """

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY,
            output_dimensionality=768,
        )
        self.client = QdrantClient(url=QDRANT_URL)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[Document]:
        if not self.client.collection_exists(QDRANT_COLLECTION):
            return []

        query_filter = None
        if document_id is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        points = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=self.embeddings.embed_query(query),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            Document(
                page_content=point.payload["text"],
                metadata={
                    "chunk_id": point.payload["chunk_id"],
                    "document_id": point.payload["document_id"],
                    "filename": point.payload["filename"],
                    "page_number": point.payload["page_number"],
                    "score": point.score,
                },
            )
            for point in points
        ]
