from sqlalchemy.orm import Session

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.llm_service import LLMService
from app.services.chat_service import ChatService


class RAGService:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.llm_service = LLMService()
        self.chat_service = ChatService()

    def chat(
        self,
        db: Session,
        question: str,
        top_k: int = 5,
        document_id: int | None = None,
        session_id: int | None = None,
    ):

        # 1. Create or retrieve session
        if session_id is None:

            session = self.chat_service.create_session(
                db=db,
                document_id=document_id,
                title=question[:100],
            )

            session_id = session.id

        else:

            session = self.chat_service.get_session(
                db=db,
                session_id=session_id,
            )

            if not session:
                raise ValueError(
                    "Chat session not found"
                )

        # 2. Get conversation history
        history_messages = (
            self.chat_service.get_history(
                db=db,
                session_id=session_id,
                limit=10,
            )
        )

        history_parts = []

        for message in history_messages:

            history_parts.append(
                f"{message.role}: {message.content}"
            )

        history = "\n".join(history_parts)

        # 3. Save user message
        self.chat_service.save_message(
            db=db,
            session_id=session_id,
            role="USER",
            content=question,
        )

        # 4. Embed question
        query_vector = (
            self.embedding_service.embed_text(
                question
            )
        )

        # 5. Search Qdrant
        results = self.qdrant_service.search(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
        )

        # 6. No relevant results
        if not results:

            answer = (
                "I could not find relevant information "
                "in the provided documents."
            )

            self.chat_service.save_message(
                db=db,
                session_id=session_id,
                role="ASSISTANT",
                content=answer,
            )

            return {
                "session_id": session_id,
                "answer": answer,
                "sources": [],
            }

        # 7. Build context
        context_parts = []
        sources = []

        for result in results:

            payload = result.payload

            context_parts.append(
                f"""
Document: {payload["filename"]}
Page: {payload["page_number"]}

{payload["text"]}
"""
            )

            sources.append(
                {
                    "chunk_id": payload["chunk_id"],
                    "document_id": payload["document_id"],
                    "filename": payload["filename"],
                    "page_number": payload["page_number"],
                    "score": result.score,
                }
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        # 8. Generate answer
        answer = self.llm_service.generate_answer(
            question=question,
            context=context,
            history=history,
        )

        # 9. Save assistant response
        self.chat_service.save_message(
            db=db,
            session_id=session_id,
            role="ASSISTANT",
            content=answer,
        )

        return {
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
        }