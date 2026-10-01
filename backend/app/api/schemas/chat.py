from pydantic import BaseModel, Field

from app.core.config import settings

# Single source of truth: the limit lives in settings and is imported everywhere else.
MAX_MESSAGE_CHARS = settings.MAX_MESSAGE_CHARS


class ChatRequest(BaseModel):
    """One inbound chat turn.

    Omit ``session_id`` on the first message and reuse the returned id afterwards to keep
    the conversation (language, pending slot and remembered entities) alive.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "where is my order"},
                {"message": "ORD-12345", "session_id": "0bee713fb34a490e9ca11438de8384d5"},
                {"message": "मेरा ऑर्डर कहाँ है"},
                {"message": "hola, quiero cancelar mi pedido"},
            ]
        }
    }

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_CHARS,
        description="The user's message, 1-1000 characters. Any language is accepted.",
    )
    session_id: str | None = Field(
        default=None,
        description="Omit on the first message; reuse the returned id afterwards.",
    )


class ChatResponse(BaseModel):
    """The bot's reply plus everything the UI needs to render the turn."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "0bee713fb34a490e9ca11438de8384d5",
                    "reply": "Sure, I can help. Please share your order ID (e.g. ORD-12345).",
                    "intent": "track_order",
                    "confidence": 1.0,
                    "language": "en",
                    "entities": {},
                    "suggestions": [],
                    "is_follow_up": False,
                    "needs_input": True,
                },
                {
                    "session_id": "0bee713fb34a490e9ca11438de8384d5",
                    "reply": "Your order ORD-12345 is being processed and will be packed soon.",
                    "intent": "track_order",
                    "confidence": 0.0,
                    "language": "en",
                    "entities": {"order_id": "ORD-12345"},
                    "suggestions": [],
                    "is_follow_up": True,
                    "needs_input": False,
                },
            ]
        }
    }

    session_id: str
    reply: str
    intent: str
    confidence: float
    language: str
    entities: dict[str, str] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)
    is_follow_up: bool = False
    needs_input: bool = False