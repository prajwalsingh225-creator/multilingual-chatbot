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

    def delete_session(self, session_id: str) -> bool:
        record = self.get_session(session_id)
        if record is None:
            return False
        self.db.delete(record)
        self.db.commit()
        return True
