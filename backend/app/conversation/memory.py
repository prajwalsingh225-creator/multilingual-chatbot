"""Short-term conversation memory (bounded, in-process)."""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    content: str
    language: str | None = None
    intent: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class ConversationMemory:
    def __init__(self, max_messages: int) -> None:
        self._messages: deque[Message] = deque(maxlen=max_messages)

    def add(
        self,
        role: str,
        content: str,
        *,
        language: str | None = None,
        intent: str | None = None,
    ) -> Message:
        msg = Message(role=role, content=content, language=language, intent=intent)
        self._messages.append(msg)
        return msg

    def recent(self, n: int | None = None) -> list[Message]:
        items = list(self._messages)
        return items if n is None else items[-n:]

    def last_user_intent(self) -> str | None:
        for msg in reversed(self._messages):
            if msg.role == "user" and msg.intent:
                return msg.intent
        return None

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
