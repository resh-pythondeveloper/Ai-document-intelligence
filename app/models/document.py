from datetime import datetime
from app.db.postgres import Base
from sqlalchemy.orm import Mapped,mapped_column
from sqlalchemy import DateTime, String, Text

class Document(Base):
    __tablename__="documents"
    id:Mapped[int]=mapped_column(primary_key=True,
        index=True,)
    filename:Mapped[str]=mapped_column(String(255),
        nullable=False,)
    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )