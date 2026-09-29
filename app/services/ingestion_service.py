from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import PDFService
from app.services.qdrant_service import QdrantService


class IngestionService:

    def __init__(self):

        self.pdf_service = PDFService()
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()

    def process_document(
        self,
        document_id: int,
        filename: str,
        file_path: str,
    ):

        # 1. Extract PDF pages
        pages = self.pdf_service.extract_pages(
            file_path
        )

        # 2. Create chunks
        chunks = self.pdf_service.create_chunks(
            pages=pages,
            document_id=document_id,
            filename=filename,
        )

        # 3. Create embeddings
        embedded_chunks = (
            self.embedding_service.embed_documents(
                chunks
            )
        )

        # 4. Create Qdrant collection
        self.qdrant_service.create_collection()

        # 5. Store vectors
        total_inserted = (
            self.qdrant_service.insert_chunks(
                embedded_chunks
            )
        )

        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "vectors": total_inserted,
        }