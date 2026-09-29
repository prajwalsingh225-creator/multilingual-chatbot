from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(
        default=None, description="Omit on the first message; reuse the returned id afterwards."
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    confidence: float
    language: str
    entities: dict[str, str] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)
    is_follow_up: bool = False
    needs_input: bool = False