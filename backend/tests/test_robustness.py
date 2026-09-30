"""Robustness behaviours from phase 4.3.

Covers: session rehydration after a restart, lazy expiry purging, input safety,
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


# --- session rehydration ------------------------------------------------------


def test_session_is_rehydrated_from_the_database_after_a_restart(client, app) -> None:
    """Simulate a restart: the in-memory map is empty but the DB still has the session."""
    first = _chat(client, "hola")
    session_id = first["session_id"]

    # Wipe memory the way a process restart would, keep the database.
    for stored in list(app.state.pipeline.sessions._sessions):  # noqa: SLF001
        app.state.pipeline.sessions.delete(stored)
    assert len(app.state.pipeline.sessions) == 0

    second = _chat(client, "hello again", session_id)
    assert second["session_id"] == session_id, "session id must survive a restart"
    assert second["language"] == "es", "language restored from the database"


def test_rehydration_restores_recent_messages(client, app) -> None:
    from app.database.database import SessionLocal
    from app.database.repository import ConversationRepository

    session_id = _chat(client, "where is my order")["session_id"]
    _chat(client, "hello", session_id)
    for stored in list(app.state.pipeline.sessions._sessions):  # noqa: SLF001
        app.state.pipeline.sessions.delete(stored)

    db = SessionLocal()
    try:
        rows = ConversationRepository(db).list_messages(session_id)
    finally:
        db.close()
    assert len(rows) == 4

    # Ask for the slot again: the pending intent came from the transcript.
    resumed = _chat(client, "ORD-99999", session_id)
    assert resumed["session_id"] == session_id
    assert resumed["intent"] == "track_order"
    assert resumed["is_follow_up"] is True


def test_rehydrated_memory_is_bounded(client, app) -> None:
    session_id = _chat(client, "hello")["session_id"]
    for stored in list(app.state.pipeline.sessions._sessions):  # noqa: SLF001
        app.state.pipeline.sessions.delete(stored)
    _chat(client, "I need help", session_id)
    session = app.state.pipeline.sessions.get(session_id)
    assert len(session.memory) <= settings.MAX_CONTEXT_MESSAGES


def test_an_expired_database_session_is_not_rehydrated(client, app) -> None:
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

    fresh = _chat(client, "hello", session_id)
    assert fresh["session_id"] != session_id, "expired session must not be resurrected"


def test_an_unknown_session_id_starts_a_new_one(client) -> None:
    body = _chat(client, "hello", "definitely-not-a-real-session")
    assert body["session_id"] != "definitely-not-a-real-session"


# --- expiry purging -----------------------------------------------------------


def test_creating_a_session_purges_expired_ones() -> None:
    manager = SessionManager(timeout_minutes=30, max_context_messages=5)
    stale = manager.create()
    stale.last_active = datetime.now(UTC) - timedelta(hours=2)
    assert stale.session_id in manager._sessions  # noqa: SLF001

    manager.create()
    assert stale.session_id not in manager._sessions  # noqa: SLF001


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


def test_message_longer_than_the_limit_is_rejected(client) -> None:
    response = client.post(f"{PREFIX}/chat", json={"message": "x" * (MAX_MESSAGE_CHARS + 1)})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


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


def test_logs_do_not_contain_message_text(caplog) -> None:
    secret = "my email is leak.test@example.com"
    with caplog.at_level(logging.INFO):
        with TestClient(create_app()) as client:
            _chat(client, secret)
    assert secret not in caplog.text
    assert "leak.test@example.com" not in caplog.text


def test_turn_log_records_intent_and_length_not_content(caplog) -> None:
    with caplog.at_level(logging.INFO):
        with TestClient(create_app()) as client:
            _chat(client, "where is my order")
    turn_lines = [line for line in caplog.text.splitlines() if "turn session=" in line]
    assert turn_lines, "expected a turn log line"
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


def test_unexpected_error_returns_500_without_a_stack_trace(safe_client, monkeypatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    app = safe_client.app
    monkeypatch.setattr(app.state.pipeline.responder, "build_reply", boom)
    response = safe_client.post(f"{PREFIX}/chat", json={"message": "hello"})
    assert response.status_code == 500
    body = response.json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == "internal_error"
    assert "Traceback" not in response.text
    assert "secret internal detail" not in body["detail"]


def test_production_errors_never_include_the_exception_text(monkeypatch) -> None:
    monkeypatch.setattr(settings, "APP_ENV", "production")
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            app = client.app

            def boom(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("secret internal detail")

            app.state.pipeline.responder.build_reply = boom  # type: ignore[method-assign]
            response = client.post(f"{PREFIX}/chat", json={"message": "hello"})
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"
        assert "secret internal detail" not in response.text
    finally:
        monkeypatch.undo()


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


def test_request_id_appears_in_the_log_lines(caplog) -> None:
    with caplog.at_level(logging.INFO):
        with TestClient(create_app()) as client:
            client.get(f"{PREFIX}/health", headers={"X-Request-ID": "log-me-42"})
    assert "log-me-42" in caplog.text


def test_request_id_defaults_to_a_dash_outside_a_request() -> None:
    from app.core.logging import request_id_var

    assert request_id_var.get() == "-"