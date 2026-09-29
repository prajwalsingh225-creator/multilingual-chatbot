"""In-memory session store with idle timeout.

Messages are also persisted to the database by the pipeline; sessions themselves
live in memory (a restart starts fresh sessions).
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.conversation.memory import ConversationMemory
from app.core.exceptions import SessionNotFoundError


@dataclass
class Session:
    session_id: str
    memory: ConversationMemory
    language: str | None = None
    pending_intent: str | None = None  # intent waiting for a missing slot (e.g. order_id)
    last_intent: str | None = None
    entities: dict[str, str] = field(default_factory=dict)  # carried across turns
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))

    def touch(self) -> None:
        self.last_active = datetime.now(UTC)


class SessionManager:
    def __init__(self, timeout_minutes: int, max_context_messages: int) -> None:
        self._timeout = timedelta(minutes=timeout_minutes)
        self._max_messages = max_context_messages
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _expired(self, session: Session) -> bool:
        return datetime.now(UTC) - session.last_active > self._timeout

    def create(self, language: str | None = None) -> Session:
        session = Session(
            session_id=uuid.uuid4().hex,
            memory=ConversationMemory(self._max_messages),
            language=language,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or self._expired(session):
                self._sessions.pop(session_id, None)
                raise SessionNotFoundError(f"Session '{session_id}' not found or expired")
            return session

    def get_or_create(self, session_id: str | None) -> tuple[Session, bool]:
        """Return (session, created). Unknown/expired ids get a brand-new session."""
        if session_id:
            try:
                return self.get(session_id), False
            except SessionNotFoundError:
                pass
        return self.create(), True

    def delete(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                raise SessionNotFoundError(f"Session '{session_id}' not found")

    def purge_expired(self) -> int:
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if self._expired(s)]
            for sid in stale:
                del self._sessions[sid]
            return len(stale)
