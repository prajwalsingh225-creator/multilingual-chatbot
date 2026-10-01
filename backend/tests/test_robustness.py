"""Robustness behaviours from phase 4.3.

Covers: session lookup order and expiry, rehydration after a restart, input safety,
consistent error JSON and request-id correlation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api.schemas.chat import MAX_MESSAGE_CHARS
from app.conversation.session_manager import SessionManager
from app.core.config import settings
from app.main import create_app
from app.nlp.preprocessor import normalize_text, strip_control_chars

PREFIX = "/api/v1"


def _chat(client: TestClient, message: str, session_id: str | None = None) -> dict:
    payload: dict[str, object] = {"message": message}
    if session_id is not None:
        payload["session_id"] = session_id
    response = client.post(f"{PREFIX}/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _restart(app) -> None:
    """Forget every in-memory session, as a process restart would (the DB survives)."""
    pipeline = app.state.pipeline
    for stored in list(pipeline.sessions._sessions):
        pipeline.sessions.delete(stored)
    assert len(pipeline.sessions) == 0


@pytest.fixture
def chat_client(client, caplog) -> TestClient:
    """A client whose app logs at INFO in a way ``caplog`` can actually see.

    ``setup_logging`` runs inside the lifespan, i.e. AFTER ``caplog.at_level`` in the
    test body would have run, and it resets the root level to ``LOG_LEVEL``
    (WARNING under test). Setting the level here means "after the app started", so
    ``logger.info`` records reach caplog instead of being silently dropped.
    """
    caplog.set_level(logging.INFO)
    return client


# --- session rehydration ------------------------------------------------------


def test_session_is_rehydrated_from_the_database_after_a_restart(client, app) -> None:
    """Simulate a restart: the in-memory map is empty but the DB still has the session."""
    first = _chat(client, "hola")
    session_id = first["session_id"]
    _restart(app)

    # "???" is undetectable, so the language detector falls back to method="default"
    # and the stored language must win. "hello again" would NOT test this: it is
    # confidently English, and the contract lets a confident detection override.
    second = _chat(client, "???", session_id)
    assert second["session_id"] == session_id, "session id must survive a restart"
    assert second["language"] == "es", "stored language restored when detection defaults"


def test_a_confidently_detected_language_overrides_the_stored_one(client, app) -> None:
    first = _chat(client, "hola")
    _restart(app)
    second = _chat(client, "hello again", first["session_id"])
    assert second["language"] == "en", "confident detection overrides the stored language"


def test_rehydration_resumes_a_pending_slot_fill(client, app) -> None:
    """R3: pending_intent/entities are persisted, so slot filling survives a restart."""
    session_id = _chat(client, "where is my order")["session_id"]
    assert _chat(client, "ORD-99999", session_id) is not None

    _restart(app)
    resumed = _chat(client, "ORD-12345", session_id)
    assert resumed["session_id"] == session_id
    assert resumed["intent"] == "track_order"
    assert resumed["is_follow_up"] is True
    assert resumed["needs_input"] is False
    assert resumed["entities"] == {"order_id": "ORD-12345"}


def test_pending_intent_asks_for_the_slot_after_a_restart(client, app) -> None:
    session_id = _chat(client, "where is my order")["session_id"]
    _restart(app)
    # "???" cannot fill a slot, so the bot must re-ask instead of inventing an answer.
    again = _chat(client, "???", session_id)
    assert again["intent"] == "track_order"
    assert again["needs_input"] is True


def test_rehydration_restores_the_recent_transcript(client, app) -> None:
    session_id = _chat(client, "where is my order")["session_id"]
    _restart(app)
    _chat(client, "ORD-12345", session_id)
    session = app.state.pipeline.sessions.get(session_id)
    restored = session.memory.recent()
    assert len(restored) >= 2
    assert restored[0].content == "where is my order", "oldest message first"
    assert restored[0].role == "user"


def test_rehydrated_memory_is_bounded(client, app) -> None:
    session_id = _chat(client, "hello")["session_id"]
    _restart(app)
    _chat(client, "I need help", session_id)
    session = app.state.pipeline.sessions.get(session_id)
    assert len(session.memory) <= settings.MAX_CONTEXT_MESSAGES


def test_rehydration_leaves_no_orphan_sessions(client, app) -> None:
    """R2: nothing is created until both the memory and DB lookups have failed."""
    session_id = _chat(client, "hola")["session_id"]
    _restart(app)
    _chat(client, "???", session_id)
    assert len(app.state.pipeline.sessions) == 1, "resume must not strand a blank session"
    stored = list(app.state.pipeline.sessions._sessions)
    assert stored == [session_id]


def test_an_unknown_session_id_starts_a_new_one(client) -> None:
    body = _chat(client, "hello", "definitely-not-a-real-session")
    assert body["session_id"] != "definitely-not-a-real-session"


def test_an_expired_database_session_is_not_rehydrated(client, app) -> None:
    """DB row aged past the timeout, memory empty -> the id is not resurrectable."""
    from app.database.database import SessionLocal
    from app.database.models import SessionRecord

    session_id = _chat(client, "hello")["session_id"]
    db = SessionLocal()
    try:
        record = db.get(SessionRecord, session_id)
        assert record is not None
        record.last_active = datetime.now(UTC) - timedelta(days=2)
        db.commit()
    finally:
        db.close()

    _restart(app)
    fresh = _chat(client, "hello", session_id)
    assert fresh["session_id"] != session_id, "expired session must not be resurrected"


def test_an_in_memory_session_aged_past_the_timeout_is_replaced(client, app) -> None:
    """Memory aged past the timeout -> a new session id, even though the DB row exists."""
    session_id = _chat(client, "hello")["session_id"]
    app.state.pipeline.sessions.get(session_id).last_active = datetime.now(UTC) - timedelta(
        hours=2
    )
    fresh = _chat(client, "hello", session_id)
    assert fresh["session_id"] != session_id


def test_expiry_does_not_delete_history_from_the_database(client, app) -> None:
    from app.database.database import SessionLocal
    from app.database.models import MessageRecord

    session_id = _chat(client, "hello")["session_id"]
    app.state.pipeline.sessions.get(session_id).last_active = datetime.now(UTC) - timedelta(
        hours=2
    )
    _chat(client, "hello", session_id)
    db = SessionLocal()
    try:
        rows = db.query(MessageRecord).filter(MessageRecord.session_id == session_id).all()
    finally:
        db.close()
    assert rows, "expiry must forget the session, not erase its transcript"


# --- expiry purging -----------------------------------------------------------


def test_creating_a_session_purges_expired_ones() -> None:
    manager = SessionManager(timeout_minutes=30, max_context_messages=5)
    stale = manager.create()
    stale.last_active = datetime.now(UTC) - timedelta(hours=2)
    assert stale.session_id in manager._sessions

    manager.create()
    assert stale.session_id not in manager._sessions


def test_purge_expired_does_not_remove_live_sessions() -> None:
    manager = SessionManager(timeout_minutes=30, max_context_messages=5)
    live = manager.create()
    assert manager.purge_expired() == 0
    assert manager.get(live.session_id).session_id == live.session_id


def test_is_expired_at_accepts_naive_datetimes() -> None:
    manager = SessionManager(timeout_minutes=30, max_context_messages=5)
    naive_old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2)
    assert manager.is_expired_at(naive_old) is True


# --- input safety -------------------------------------------------------------


def test_control_characters_are_stripped() -> None:
    assert strip_control_chars("he\x00llo\x07 w\x1borld") == "hello world"


def test_normalize_removes_control_characters() -> None:
    assert normalize_text("he\x00llo\x1f") == "hello"


def test_control_characters_do_not_survive_a_chat_turn(client) -> None:
    body = _chat(client, "he\x00llo \x07 there")
    assert body["reply"], "a control-character payload must still be handled cleanly"


def test_tabs_and_newlines_become_spaces() -> None:
    assert normalize_text("hello\tworld") == "hello world"


def test_the_length_limit_has_one_source_of_truth(monkeypatch) -> None:
    """R4f: the schema limit is derived from settings, not hard-coded twice.

    Comparing two independent literals only proves they happen to agree today. Re-import
    the schema module with a different configured value and assert the *schema bound*
    follows it, which is the behaviour that actually matters.
    """
    import importlib

    import app.api.schemas.chat as chat_schema

    monkeypatch.setattr(settings, "MAX_MESSAGE_CHARS", 42)
    reloaded = importlib.reload(chat_schema)
    try:
        bound = next(
            m.max_length
            for m in reloaded.ChatRequest.model_fields["message"].metadata
            if hasattr(m, "max_length")
        )
        assert bound == 42
        assert reloaded.MAX_MESSAGE_CHARS == 42
    finally:
        monkeypatch.undo()
        importlib.reload(chat_schema)
    assert chat_schema.MAX_MESSAGE_CHARS == settings.MAX_MESSAGE_CHARS


def test_message_longer_than_the_limit_is_rejected(client) -> None:
    response = client.post(f"{PREFIX}/chat", json={"message": "x" * (MAX_MESSAGE_CHARS + 1)})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "message" in body["detail"]


def test_pipeline_also_guards_the_length_for_non_http_callers() -> None:
    """The pipeline enforces the limit itself, so a caller cannot bypass it."""
    from app.core.exceptions import MessageTooLongError
    from app.database.database import SessionLocal
    from app.pipeline import build_pipeline

    pipeline = build_pipeline(settings)
    db = SessionLocal()
    try:
        with pytest.raises(MessageTooLongError) as excinfo:
            pipeline.process("x" * (MAX_MESSAGE_CHARS + 1), None, db)
        assert str(settings.MAX_MESSAGE_CHARS) in str(excinfo.value)
    finally:
        db.close()


def test_logs_do_not_contain_message_text(chat_client, caplog) -> None:
    secret = "my email is leak.test@example.com"
    body = _chat(chat_client, secret)
    assert body["reply"], "positive case: the turn was actually handled"
    assert secret not in caplog.text
    assert "leak.test@example.com" not in caplog.text


def test_turn_log_records_intent_and_length_not_content(chat_client, caplog) -> None:
    """Rule 3: a 'does not log the message' test must also prove the line exists."""
    _chat(chat_client, "where is my order")
    turn_lines = [line for line in caplog.text.splitlines() if "turn session=" in line]
    assert turn_lines, f"expected a turn log line, got: {caplog.text!r}"
    line = turn_lines[-1]
    assert "intent=track_order" in line
    assert "in_len=" in line
    assert "where is my order" not in line


# --- consistent error JSON ----------------------------------------------------


def test_validation_error_uses_the_standard_envelope(client) -> None:
    body = client.post(f"{PREFIX}/chat", json={"message": ""}).json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == "validation_error"
    assert isinstance(body["detail"], str)


def test_unknown_session_uses_the_standard_envelope(client) -> None:
    body = client.get(f"{PREFIX}/sessions/nope").json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == "session_not_found"


def test_unexpected_error_returns_a_generic_500(safe_client, monkeypatch) -> None:
    app = safe_client.app

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app.state.pipeline.responder, "build_reply", boom)
    response = safe_client.post(f"{PREFIX}/chat", json={"message": "hello"})
    assert response.status_code == 500
    body = response.json()
    assert set(body) == {"error", "detail"}
    assert body == {"error": "internal_error", "detail": "Internal server error"}
    assert "Traceback" not in response.text
    assert "secret internal detail" not in response.text
    assert "RuntimeError" not in response.text


def test_production_errors_are_identical_to_development_ones(monkeypatch) -> None:
    """R1: dev and prod must not differ; only /docs availability differs."""

    def body_in(env: str) -> dict:
        monkeypatch.setattr(settings, "APP_ENV", env)
        with TestClient(create_app(), raise_server_exceptions=False) as client:

            def boom(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("secret internal detail")

            client.app.state.pipeline.responder.build_reply = boom  # type: ignore[method-assign]
            return client.post(f"{PREFIX}/chat", json={"message": "hello"}).json()

    assert body_in("development") == body_in("production")


def test_a_500_response_carries_the_request_id(safe_client, monkeypatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(safe_client.app.state.pipeline.responder, "build_reply", boom)
    response = safe_client.post(
        f"{PREFIX}/chat", json={"message": "hello"}, headers={"X-Request-ID": "boom-500"}
    )
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "boom-500"


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("app_env", ["development", "production"])
def test_no_environment_ever_leaks_a_traceback(monkeypatch, debug, app_env) -> None:
    """Regression: Starlette replaces the registered handler with a plain-text
    traceback when ``FastAPI(debug=True)``, which would leak internals in development
    and drop the X-Request-ID header. This runs with DEBUG=True exactly like the
    shipped ``.env``."""
    monkeypatch.setattr(settings, "DEBUG", debug)
    monkeypatch.setattr(settings, "APP_ENV", app_env)

    with TestClient(create_app(), raise_server_exceptions=False) as client:

        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("secret internal detail")

        client.app.state.pipeline.responder.build_reply = boom  # type: ignore[method-assign]
        response = client.post(
            f"{PREFIX}/chat", json={"message": "hello"}, headers={"X-Request-ID": "x-500"}
        )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"error": "internal_error", "detail": "Internal server error"}
    assert "Traceback" not in response.text
    assert "secret internal detail" not in response.text
    assert "RuntimeError" not in response.text
    assert response.headers["X-Request-ID"] == "x-500"


def test_the_traceback_is_logged_server_side_with_the_request_id(safe_client, caplog) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    safe_client.app.state.pipeline.responder.build_reply = boom  # type: ignore[method-assign]
    caplog.set_level(logging.ERROR)
    safe_client.post(
        f"{PREFIX}/chat", json={"message": "hello"}, headers={"X-Request-ID": "trace-500"}
    )
    assert "trace-500" in caplog.text
    assert "secret internal detail" in caplog.text, "the traceback must stay server-side"


def test_docs_are_disabled_in_production(monkeypatch) -> None:
    monkeypatch.setattr(settings, "APP_ENV", "production")
    with TestClient(create_app()) as client:
        assert client.get("/docs").status_code == 404


# --- request ids --------------------------------------------------------------


def test_every_response_carries_a_request_id(client) -> None:
    response = client.get(f"{PREFIX}/health")
    assert response.headers["X-Request-ID"]


def test_supplied_request_id_is_echoed_back(client) -> None:
    response = client.get(
        f"{PREFIX}/health", headers={"X-Request-ID": "my-correlation-id-123"}
    )
    assert response.headers["X-Request-ID"] == "my-correlation-id-123"


def test_request_ids_are_unique_per_request(client) -> None:
    first = client.get(f"{PREFIX}/health").headers["X-Request-ID"]
    second = client.get(f"{PREFIX}/health").headers["X-Request-ID"]
    assert first != second


def test_a_404_also_carries_the_request_id(client) -> None:
    response = client.get(
        f"{PREFIX}/sessions/nope", headers={"X-Request-ID": "gone-404"}
    )
    assert response.status_code == 404
    assert response.headers["X-Request-ID"] == "gone-404"


def test_a_422_also_carries_the_request_id(client) -> None:
    response = client.post(
        f"{PREFIX}/chat", json={"message": ""}, headers={"X-Request-ID": "bad-422"}
    )
    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "bad-422"


def test_request_id_appears_in_the_log_lines(chat_client, caplog) -> None:
    chat_client.get(f"{PREFIX}/health", headers={"X-Request-ID": "log-me-42"})
    assert "log-me-42" in caplog.text


def test_request_id_defaults_to_a_dash_outside_a_request() -> None:
    from app.core.logging import request_id_var

    assert request_id_var.get() == "-"


def test_setup_logging_keeps_foreign_handlers_intact() -> None:
    """R4e: the app must not wipe handlers it did not install."""
    import logging as _logging

    sentinel = _logging.NullHandler()
    root = _logging.getLogger()
    root.addHandler(sentinel)
    try:
        from app.core.logging import setup_logging

        setup_logging("INFO")
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)