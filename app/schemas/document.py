from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    id: int
    filename: str
    title: str | None
    file_path: str
    description: str | None
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)