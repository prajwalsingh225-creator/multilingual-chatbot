"""Orchestrates one chat turn:

session -> language -> preprocess -> intent -> entities -> context -> handler -> reply
"""

from dataclasses import dataclass

from fastapi import Request
from sqlalchemy.orm import Session as DBSession

from app.api.schemas.chat import ChatResponse
from app.conversation.context_manager import ContextManager
from app.conversation.session_manager import Session, SessionManager
from app.core.config import Settings
from app.core.exceptions import EmptyMessageError, MessageTooLongError
from app.core.logging import get_logger
from app.database.repository import ConversationRepository
from app.intents.handlers import HandlerContext, execute_intent
from app.models.model_registry import ModelRegistry
from app.nlp.entity_extractor import EntityExtractor
from app.nlp.intent_classifier import IntentClassifier
from app.nlp.language_detector import LanguageDetector
from app.nlp.preprocessor import Preprocessor
from app.response.response_manager import ResponseManager
from app.response.templates import TemplateStore
from app.response.translator import Translator

logger = get_logger(__name__)


@dataclass
class ChatPipeline:
    settings: Settings
    models: ModelRegistry
    detector: LanguageDetector
    preprocessor: Preprocessor
    classifier: IntentClassifier
    extractor: EntityExtractor
    sessions: SessionManager
    context: ContextManager
    responder: ResponseManager

    def _rehydrate(self, session_id: str, repo: ConversationRepository) -> Session | None:
        """Rebuild an in-memory session from the database after a process restart.

        Returns ``None`` when the id is unknown or the stored session has expired, in
        which case the caller starts a genuinely new conversation.
        """
        record = repo.get_session(session_id)
        if record is None:
            return None
        last_active = record.last_active or record.created_at
        if self.sessions.is_expired_at(last_active):
            logger.info("session expired in db, starting a new one")
            return None

        session = self.sessions.new_session(session_id, language=record.language)
        session.created_at = record.created_at
        session.last_active = last_active
        for row in repo.recent_messages(session_id, self.sessions.max_messages):
            session.memory.add(
                row.role, row.content, language=row.language, intent=row.intent
            )
        for row in reversed(repo.recent_messages(session_id, self.sessions.max_messages)):
            if row.role == "user" and row.intent:
                session.last_intent = row.intent
        self.sessions.restore(session)
        logger.info(
            "rehydrated session from db: restored=%d messages",
            len(session.memory),
        )
        return session

    def process(self, message: str, session_id: str | None, db: DBSession) -> ChatResponse:
        text = message.strip()
        if not text:
            raise EmptyMessageError()
        if len(text) > self.settings.MAX_MESSAGE_CHARS:
            raise MessageTooLongError(self.settings.MAX_MESSAGE_CHARS)

        repo = ConversationRepository(db)
        session, created = self.sessions.get_or_create(session_id)
        if created:
            restored = self._rehydrate(session_id, repo) if session_id else None
            if restored is not None:
                session = restored
            else:
                repo.create_session(session.session_id)

        detected = self.detector.detect(text)
        language = (
            session.language or detected.language
            if detected.method == "default"
            else detected.language
        )
        session.language = language
        repo.touch_session(session.session_id, language)

        processed = self.preprocessor.process(text)
        prediction = self.classifier.predict(processed)
        entities = self.extractor.extract(text)
        resolved = self.context.resolve(session, prediction, entities)

        result = execute_intent(
            HandlerContext(
                intent=resolved.intent,
                entities=resolved.entities,
                language=language,
                session_id=session.session_id,
                email=self.settings.SUPPORT_EMAIL,
                phone=self.settings.SUPPORT_PHONE,
            )
        )
        reply = self.responder.build_reply(result, language)
        suggestions = self.responder.build_suggestions(result, language)

        session.pending_intent = result.pending_intent
        session.last_intent = resolved.intent
        session.memory.add("user", text, language=language, intent=resolved.intent)
        session.memory.add("assistant", reply, language=language)
        session.touch()

        repo.add_message(
            session.session_id, "user", text, language, resolved.intent, resolved.confidence
        )
        repo.add_message(session.session_id, "assistant", reply, language, resolved.intent)

        # Log shape, never the message itself: user text can carry PII.
        logger.info(
            "turn session=%s lang=%s intent=%s conf=%.2f follow_up=%s src=%s "
            "in_len=%d entities=%s",
            session.session_id[:8],
            language,
            resolved.intent,
            resolved.confidence,
            resolved.is_follow_up,
            prediction.source,
            len(text),
            ",".join(sorted(resolved.entities)),
        )
        return ChatResponse(
            session_id=session.session_id,
            reply=reply,
            intent=resolved.intent,
            confidence=round(resolved.confidence, 4),
            language=language,
            entities=resolved.entities,
            suggestions=suggestions,
            is_follow_up=resolved.is_follow_up,
            needs_input=result.pending_intent is not None,
        )


def build_pipeline(settings: Settings) -> ChatPipeline:
    models = ModelRegistry()
    models.load(settings.MODEL_DIR)
    classifier = IntentClassifier.from_intents_file(
        settings.intents_file, settings.INTENT_CONFIDENCE_THRESHOLD, models.intent_model
    )
    translator = Translator(TemplateStore(settings.translations_dir), settings.DEFAULT_LANGUAGE)
    return ChatPipeline(
        settings=settings,
        models=models,
        detector=LanguageDetector(settings.SUPPORTED_LANGUAGES, settings.DEFAULT_LANGUAGE),
        preprocessor=Preprocessor(),
        classifier=classifier,
        extractor=EntityExtractor(),
        sessions=SessionManager(settings.SESSION_TIMEOUT_MINUTES, settings.MAX_CONTEXT_MESSAGES),
        context=ContextManager(),
        responder=ResponseManager(translator),
    )


def get_pipeline(request: Request) -> ChatPipeline:
    pipeline: ChatPipeline = request.app.state.pipeline
    return pipeline