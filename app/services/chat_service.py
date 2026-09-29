from sqlalchemy.orm import Session

from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage


class ChatService:

    def create_session(
        self,
        db: Session,
        document_id: int | None = None,
        title: str | None = None,
    ) -> ChatSession:

        session = ChatSession(
            document_id=document_id,
            title=title,
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        return session

    def get_session(
        self,
        db: Session,
        session_id: int,
    ) -> ChatSession | None:

        return db.get(
            ChatSession,
            session_id,
        )

    def save_message(
        self,
        db: Session,
        session_id: int,
        role: str,
        content: str,
    ) -> ChatMessage:

        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    def get_history(
        self,
        db: Session,
        session_id: int,
        limit: int = 10,
    ) -> list[ChatMessage]:

        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))
