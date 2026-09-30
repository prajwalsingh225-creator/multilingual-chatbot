"""Session store, bounded memory and context resolution."""

from __future__ import annotations

import pytest

from app.conversation.context_manager import ContextManager
from app.conversation.memory import ConversationMemory
from app.conversation.session_manager import SessionManager
from app.core.exceptions import SessionNotFoundError
from app.nlp.intent_classifier import IntentPrediction

FALLBACK = IntentPrediction("fallback", 0.0, "rules", "fallback")


def _prediction(intent: str, confidence: float = 0.9) -> IntentPrediction:
    return IntentPrediction(intent, confidence, "rules", intent)


def _manager(max_messages: int = 10) -> SessionManager:
    return SessionManager(timeout_minutes=30, max_context_messages=max_messages)


def test_memory_is_bounded() -> None:
    memory = ConversationMemory(max_messages=4)
    for index in range(10):
        memory.add("user", f"message {index}")
    assert len(memory) == 4


def test_memory_keeps_the_most_recent_messages() -> None:
    memory = ConversationMemory(max_messages=3)
    for index in range(5):
        memory.add("user", f"m{index}")
    assert [m.content for m in memory.recent()] == ["m2", "m3", "m4"]


def test_memory_last_user_intent() -> None:
    memory = ConversationMemory(max_messages=10)
    memory.add("user", "hi", intent="greeting")
    memory.add("assistant", "hello")
    memory.add("user", "track", intent="track_order")
    assert memory.last_user_intent() == "track_order"


def test_memory_clear() -> None:
    memory = ConversationMemory(max_messages=5)
    memory.add("user", "hi")
    memory.clear()
    assert len(memory) == 0


def test_session_manager_create_and_get() -> None:
    manager = _manager()
    session = manager.create()
    assert manager.get(session.session_id).session_id == session.session_id


def test_unknown_session_raises() -> None:
    with pytest.raises(SessionNotFoundError):
        _manager().get("does-not-exist")


def test_get_or_create_returns_the_existing_session() -> None:
    manager = _manager()
    session = manager.create()
    same, created = manager.get_or_create(session.session_id)
    assert created is False
    assert same.session_id == session.session_id


def test_get_or_create_with_none_makes_a_new_session() -> None:
    _, created = _manager().get_or_create(None)
    assert created is True


def test_expired_sessions_are_purged() -> None:
    manager = SessionManager(timeout_minutes=0, max_context_messages=5)
    manager.create()
    assert manager.purge_expired() >= 1


def test_delete_removes_the_session() -> None:
    manager = _manager()
    session = manager.create()
    manager.delete(session.session_id)
    with pytest.raises(SessionNotFoundError):
        manager.get(session.session_id)


def test_pending_intent_resumes_on_a_bare_order_id() -> None:
    context = ContextManager()
    session = _manager().create()
    session.pending_intent = "track_order"
    resolved = context.resolve(session, FALLBACK, {"order_id": "ORD-12345"})
    assert resolved.intent == "track_order"
    assert resolved.is_follow_up is True


def test_entities_persist_across_turns() -> None:
    context = ContextManager()
    session = _manager().create()
    context.resolve(session, _prediction("track_order"), {"order_id": "ORD-12345"})
    resolved = context.resolve(session, _prediction("cancel_order"), {})
    assert resolved.entities["order_id"] == "ORD-12345"


def test_a_confident_new_intent_overrides_pending() -> None:
    context = ContextManager()
    session = _manager().create()
    session.pending_intent = "track_order"
    resolved = context.resolve(session, _prediction("refund", 0.95), {})
    assert resolved.intent == "refund"
    assert resolved.is_follow_up is False


def test_bare_entity_continues_the_last_intent() -> None:
    context = ContextManager()
    session = _manager().create()
    session.last_intent = "track_order"
    resolved = context.resolve(session, FALLBACK, {"order_id": "ORD-1"})
    assert resolved.intent == "track_order"
    assert resolved.is_follow_up is True


def test_bare_entity_does_not_continue_after_a_greeting() -> None:
    context = ContextManager()
    session = _manager().create()
    session.last_intent = "greeting"
    resolved = context.resolve(session, FALLBACK, {"order_id": "ORD-1"})
    assert resolved.intent == "fallback"


def test_pipeline_memory_stays_bounded(settings_obj) -> None:
    manager = _manager(max_messages=settings_obj.MAX_CONTEXT_MESSAGES)
    session = manager.create()
    for index in range(settings_obj.MAX_CONTEXT_MESSAGES * 3):
        session.memory.add("user", f"m{index}")
        session.memory.add("assistant", f"r{index}")
    assert len(session.memory) == settings_obj.MAX_CONTEXT_MESSAGES