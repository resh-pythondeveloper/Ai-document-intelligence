from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


class SearchService:

    def __init__(self):

        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):

        # 1. Convert question to vector
        query_vector = (
            self.embedding_service.embed_text(
                query
            )
        )

        # 2. Search Qdrant
        results = self.qdrant_service.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        # 3. Format results
        formatted_results = []

        for result in results:

            payload = result.payload

            formatted_results.append({
                "score": result.score,
                "chunk_id": payload["chunk_id"],
                "document_id": payload["document_id"],
                "filename": payload["filename"],
                "page_number": payload["page_number"],
                "text": payload["text"],
            })

        return formatted_results