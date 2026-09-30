import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import MessageRecord, SessionRecord


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_session(self, session_id: str, language: str | None = None) -> SessionRecord:
        record = SessionRecord(id=session_id, language=language)
        self.db.add(record)
        self.db.commit()
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self.db.get(SessionRecord, session_id)

    def update_language(self, session_id: str, language: str) -> None:
        record = self.get_session(session_id)
        if record and record.language != language:
            record.language = language
            self.db.commit()

    def touch_session(
        self,
        session_id: str,
        language: str | None,
        pending_intent: str | None,
        entities: dict[str, str],
    ) -> None:
        """Persist the per-turn session state and bump ``last_active``.

        Called on EVERY turn so an active conversation never looks expired after a
        restart, and so slot-filling state survives the process going away. All three
        state fields are written unconditionally: a cleared ``pending_intent`` (None) is
        a real update, not a "leave unchanged" signal.
        """
        record = self.get_session(session_id)
        if record is None:
            return
        record.last_active = datetime.now(UTC)
        if language is not None:
            record.language = language
        # A cleared slot ("no pending intent") is stored as NULL, not as an empty string.
        record.pending_intent = pending_intent or None
        record.entities = json.dumps(entities, ensure_ascii=False) if entities else None
        self.db.commit()

    def load_session_state(self, record: SessionRecord) -> dict[str, str]:
        """Decode the persisted ``entities`` JSON back into a dict."""
        if not record.entities:
            return {}
        try:
            loaded = json.loads(record.entities)
        except json.JSONDecodeError:
            return {}
        return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        language: str | None = None,
        intent: str | None = None,
        confidence: float | None = None,
    ) -> MessageRecord:
        record = MessageRecord(
            session_id=session_id,
            role=role,
            content=content,
            language=language,
            intent=intent,
            confidence=confidence,
        )
        self.db.add(record)
        self.db.commit()
        return record

    def list_messages(self, session_id: str) -> list[MessageRecord]:
        stmt = select(MessageRecord).where(MessageRecord.session_id == session_id)
        return list(self.db.scalars(stmt.order_by(MessageRecord.id)))

    def recent_messages(self, session_id: str, limit: int) -> list[MessageRecord]:
        """Return the last ``limit`` messages, oldest first (newest-first query)."""
        stmt = (
            select(MessageRecord)
            .where(MessageRecord.session_id == session_id)
            .order_by(MessageRecord.id.desc())
            .limit(limit)
        )
        return list(reversed(list(self.db.scalars(stmt))))

    def delete_session(self, session_id: str) -> bool:
        record = self.get_session(session_id)
        if record is None:
            return False
        self.db.delete(record)
        self.db.commit()
        return True