from qdrant_client import QdrantClient, models

from app.core.config import (
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from uuid import uuid5, NAMESPACE_URL

class QdrantService:

    def __init__(self):

        self.client = QdrantClient(
            url=QDRANT_URL
        )

        self.collection_name = QDRANT_COLLECTION

    def create_collection(self):

        if self.client.collection_exists(
            self.collection_name
        ):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
        )

    def insert_chunks(
        self,
        chunks: list[dict],
    ):
        points = []

        for chunk in chunks:

            # Convert our chunk ID into a valid deterministic UUID
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    chunk["chunk_id"],
                )
            )

            point = models.PointStruct(
                id=point_id,
                vector=chunk["vector"],
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)

    def delete_by_document_id(
        self,
        document_id: int,
    ):
        """Delete every vector belonging to a document."""

        if not self.client.collection_exists(self.collection_name):
            return 0

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    ),
                )
            ]
        )

        # Count first so the API can report how many vectors were removed.
        total = self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        ).count

        if total > 0:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=query_filter
                ),
                wait=True,
            )

        return total

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: int | None = None,score_threshold: float | None = None,
    ):
        query_filter = None

        if document_id is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        return results