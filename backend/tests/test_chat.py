"""End-to-end HTTP tests through the FastAPI app."""

from __future__ import annotations

import pytest

PREFIX = "/api/v1"


def _chat(client, message: str, session_id: str | None = None) -> dict:
    payload: dict[str, object] = {"message": message}
    if session_id is not None:
        payload["session_id"] = session_id
    response = client.post(f"{PREFIX}/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client) -> None:
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["languages"] == ["en", "hi", "es"]
    assert body["intent_model_loaded"] is False
    assert body["classifier"] == "rules"


def test_docs_are_served(client) -> None:
    assert client.get("/docs").status_code == 200


def test_openapi_schema_is_complete(client) -> None:
    schema = client.get("/openapi.json").json()
    assert f"{PREFIX}/chat" in schema["paths"]
    assert f"{PREFIX}/sessions/{{session_id}}" in schema["paths"]


def test_chat_returns_a_session_id(client) -> None:
    body = _chat(client, "hello")
    assert body["session_id"]
    assert body["language"] == "en"
    assert body["reply"]


def test_chat_creates_a_session(client) -> None:
    response = client.post(f"{PREFIX}/sessions", json={})
    assert response.status_code == 201
    assert response.json()["session_id"]


def test_empty_message_is_rejected(client) -> None:
    response = client.post(f"{PREFIX}/chat", json={"message": ""})
    assert response.status_code == 422


def test_whitespace_only_message_is_rejected(client) -> None:
    response = client.post(f"{PREFIX}/chat", json={"message": "   "})
    assert response.status_code == 422


def test_too_long_message_is_rejected(client) -> None:
    response = client.post(f"{PREFIX}/chat", json={"message": "x" * 1001})
    assert response.status_code == 422


def test_missing_session_id_is_tolerated(client) -> None:
    assert _chat(client, "hello")["session_id"]


def test_unknown_session_id_starts_a_new_session(client) -> None:
    body = _chat(client, "hello", "unknown-session-id")
    assert body["session_id"] != "unknown-session-id"


def test_unparseable_input_returns_fallback(client) -> None:
    body = _chat(client, "asdf qwerty zxcv")
    assert body["intent"] == "fallback"
    assert body["suggestions"]


def test_track_order_asks_for_the_order_id(client) -> None:
    body = _chat(client, "where is my order")
    assert body["intent"] == "track_order"
    assert body["needs_input"] is True


def test_slot_filling_completes_on_the_next_turn(client) -> None:
    first = _chat(client, "where is my order")
    second = _chat(client, "ORD-12345", first["session_id"])
    assert second["intent"] == "track_order"
    assert second["needs_input"] is False
    assert second["is_follow_up"] is True
    assert "ORD-12345" in second["reply"]


def test_entities_persist_across_http_turns(client) -> None:
    first = _chat(client, "track order ORD-54321")
    assert first["entities"]["order_id"] == "ORD-54321"
    second = _chat(client, "cancel order", first["session_id"])
    assert second["entities"]["order_id"] == "ORD-54321"
    assert second["intent"] == "cancel_order"


def test_hindi_message_is_answered_in_hindi(client) -> None:
    body = _chat(client, "मेरा ऑर्डर कहाँ है")
    assert body["language"] == "hi"
    assert any("ऀ" <= ch <= "ॿ" for ch in body["reply"])


def test_spanish_message_is_answered_in_spanish(client) -> None:
    body = _chat(client, "hola, quiero cancelar mi pedido")
    assert body["language"] == "es"
    assert body["reply"] != "Hello! How can I help you today?"


def test_hinglish_maps_to_hindi(client) -> None:
    body = _chat(client, "mera order kahan hai")
    assert body["language"] == "hi"


def test_session_history_has_two_messages_per_turn(client) -> None:
    session_id = _chat(client, "hello")["session_id"]
    _chat(client, "I need help", session_id)
    history = client.get(f"{PREFIX}/sessions/{session_id}").json()
    assert history["session_id"] == session_id
    assert len(history["messages"]) == 4
    assert [m["role"] for m in history["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_history_of_unknown_session_is_404(client) -> None:
    assert client.get(f"{PREFIX}/sessions/nope").status_code == 404


def test_delete_session_then_history_is_404(client) -> None:
    session_id = _chat(client, "hello")["session_id"]
    assert client.delete(f"{PREFIX}/sessions/{session_id}").status_code == 204
    assert client.get(f"{PREFIX}/sessions/{session_id}").status_code == 404


def test_delete_unknown_session_is_404(client) -> None:
    assert client.delete(f"{PREFIX}/sessions/nope").status_code == 404


def test_session_language_is_recorded(client) -> None:
    session_id = _chat(client, "hola")["session_id"]
    history = client.get(f"{PREFIX}/sessions/{session_id}").json()
    assert history["messages"][0]["language"] == "es"


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://localhost:3000"])
def test_cors_preflight_allows_the_frontend(client, origin: str) -> None:
    response = client.options(
        f"{PREFIX}/chat",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_header_is_present_on_responses(client) -> None:
    response = client.post(
        f"{PREFIX}/chat", json={"message": "hello"}, headers={"Origin": "http://localhost:5173"}
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"