from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):
    language: str | None = None


class SessionOut(BaseModel):
    session_id: str
    language: str | None = None
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    role: str
    content: str
    language: str | None = None
    intent: str | None = None
    confidence: float | None = None
    created_at: datetime


class SessionHistory(BaseModel):
    session_id: str
    messages: list[MessageOut]
