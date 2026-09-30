# API Reference

Base URL: `http://localhost:8000/api/v1`
Interactive docs: `http://localhost:8000/docs` · schema export: `backend/docs/openapi.json`

All endpoints speak JSON. Supported languages: `en` (English), `hi` (Hindi/Hinglish), `es` (Spanish).

---

## Error format

Every non-2xx response uses the same two-key envelope. There is no third variant.

```json
{ "error": "<machine_readable_code>", "detail": "<human readable message>" }
```

| Status | `error` | When |
| --- | --- | --- |
| 404 | `session_not_found` | Unknown or deleted session id |
| 422 | `validation_error` | Request body failed schema validation (e.g. `message` too short) |
| 422 | `empty_message` | Message was whitespace only |
| 422 | `message_too_long` | Message exceeded `MAX_MESSAGE_CHARS` |
| 500 | `internal_error` | Unexpected server error. No stack trace is ever returned when `APP_ENV=production` |

Example:

```json
{ "error": "session_not_found", "detail": "Session 'abc123' not found" }
```

Every response (including errors) carries an `X-Request-ID` header. Send your own
`X-Request-ID` to have it echoed back and reused in the server log lines.

---

## `session_id` lifecycle

1. **First message** — omit `session_id`. The server creates a session and returns an id.
2. **Every later message** — send the same id back. The server then remembers your
   detected language, the intent it is currently waiting on (`pending_intent`) and the
   entities it has extracted (e.g. your order id).
3. **Expiry** — a session idles out after `SESSION_TIMEOUT_MINUTES` (default 30) and is
   forgotten. If the id is still present in the database and has not expired, it is
   transparently **rehydrated** (language + the last `MAX_CONTEXT_MESSAGES` messages
   restored) even after a server restart.
4. **Unknown id** — sending an id that is neither in memory nor in the database is *not*
   an error: you get a brand-new session and a different `session_id` in the response.
   Sending an expired id also starts a new session.
5. **Deletion** — `DELETE /sessions/{id}` removes it from memory and the database; later
   `GET` returns 404.

Session state lives in memory for speed and is mirrored to SQLite for history and
rehydration, so a single-process deployment is assumed.

---

## `needs_input` and `suggestions`

| Field | Meaning | What the UI should do |
| --- | --- | --- |
| `needs_input` | `true` when the bot asked a clarifying question (e.g. *"Please share your order ID"*) and is waiting for a slot to be filled. | Keep the conversation open; the user's next message is treated as the answer even if it classifies as `fallback`. |
| `suggestions` | Ready-made reply chips, already translated into the user's language. | Render as tappable buttons; sending one verbatim always works. |
| `is_follow_up` | `true` when this turn was resolved from session context rather than a fresh classification (e.g. a bare `ORD-12345` resuming a pending `track_order`). | Optional: use to visually mark continuation turns. |

`intent` is one of `greeting`, `goodbye`, `track_order`, `cancel_order`, `payment_issue`,
`refund`, `contact_support`, or `fallback` when nothing matched confidently.

---

## `POST /chat`

Send one message and receive the reply plus the updated conversation state.

**Request**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `message` | string | yes | 1–1000 characters. Any language. |
| `session_id` | string \| null | no | Omit on the first turn. |

**Response 200**

| Field | Type | Notes |
| --- | --- | --- |
| `session_id` | string | Always returned — store it. |
| `reply` | string | Bot text, in the detected language. |
| `intent` | string | Resolved intent, or `fallback`. |
| `confidence` | float | 0.0–1.0. |
| `language` | string | `en` / `hi` / `es`. |
| `entities` | object | Extracted slots, e.g. `{"order_id": "ORD-12345"}`. |
| `suggestions` | string[] | Translated quick replies. |
| `is_follow_up` | bool | See above. |
| `needs_input` | bool | See above. |

### curl

```bash
# 1. start a conversation (note the returned session_id)
curl -s -X POST localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"where is my order"}'
```

```json
{
  "session_id": "0bee713fb34a490e9ca11438de8384d5",
  "reply": "Sure, I can help. Please share your order ID (for example ORD-12345).",
  "intent": "track_order",
  "confidence": 1.0,
  "language": "en",
  "entities": {},
  "suggestions": [],
  "is_follow_up": false,
  "needs_input": true
}
```

```bash
# 2. fill the slot in the same session
curl -s -X POST localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"ORD-12345","session_id":"0bee713fb34a490e9ca11438de8384d5"}'
```

```json
{
  "session_id": "0bee713fb34a490e9ca11438de8384d5",
  "reply": "Your order ORD-12345 is being processed and will be packed shortly.",
  "intent": "track_order",
  "confidence": 0.0,
  "language": "en",
  "entities": { "order_id": "ORD-12345" },
  "suggestions": [],
  "is_follow_up": true,
  "needs_input": false
}
```

```bash
# Hindi / Hinglish
curl -s -X POST localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"मेरा ऑर्डर कहाँ है"}'

# Spanish
curl -s -X POST localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hola, quiero cancelar mi pedido"}'
```

---

## `GET /health`

Liveness probe and which classifier is live.

```bash
curl -s localhost:8000/api/v1/health
```

```json
{
  "status": "ok",
  "app": "Multilingual Context-Aware Chatbot",
  "env": "development",
  "languages": ["en", "hi", "es"],
  "intent_model_loaded": false,
  "classifier": "rules"
}
```

`classifier` is `"transformer"` once a fine-tuned model exists in `MODEL_DIR`, otherwise
`"rules"`. Use it in the UI (or in CI) to show which engine answered.

---

## `POST /sessions`

Create a session up front (optional — the first chat message creates one anyway).

```bash
curl -s -X POST localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" -d '{}'
```

**201 Created**

```json
{ "session_id": "5c69e82520fe486586247572c99a927d", "language": null, "created_at": "2026-09-30T17:28:55.349212Z" }
```

---

## `GET /sessions/{session_id}`

Full transcript, oldest first. Two messages are persisted per turn.

```bash
curl -s localhost:8000/api/v1/sessions/0bee713fb34a490e9ca11438de8384d5
```

```json
{
  "session_id": "0bee713fb34a490e9ca11438de8384d5",
  "messages": [
    {
      "role": "user",
      "content": "where is my order",
      "language": "en",
      "intent": "track_order",
      "confidence": 1.0,
      "created_at": "2026-09-30T17:28:55.349212Z"
    },
    {
      "role": "assistant",
      "content": "Sure, I can help. Please share your order ID (for example ORD-12345).",
      "language": "en",
      "intent": "track_order",
      "confidence": null,
      "created_at": "2026-09-30T17:28:55.349998Z"
    }
  ]
}
```

**404** `{"error": "session_not_found", "detail": "..."}` if the id is unknown or deleted.

---

## `DELETE /sessions/{session_id}`

Removes the session from memory and the database.

```bash
curl -s -o /dev/null -w "%{http_code}" -X DELETE \
  localhost:8000/api/v1/sessions/0bee713fb34a490e9ca11438de8384d5
# 204
```

**204 No Content** on success, **404** if the id does not exist.

---

## CORS

`CORS_ORIGINS` (default `http://localhost:5173,http://localhost:3000`) is applied with
credentials enabled, all methods and headers allowed. A preflight from an allowed origin
returns:

```
access-control-allow-origin: http://localhost:5173
access-control-allow-methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT, QUERY
```

Add your dev server's origin to `.env` (`CORS_ORIGINS`); it accepts a comma-separated
list or a JSON array.