# API Reference

Base URL: `http://localhost:8000/api/v1`
Interactive docs: `http://localhost:8000/docs` · exported schema: `backend/docs/openapi.json`

All endpoints speak JSON. Supported languages: `en` (English), `hi` (Hindi/Hinglish),
`es` (Spanish).

Every response below was captured from a running server or `TestClient` against the
committed model. Values such as `confidence` reflect the real model and will vary
slightly if you retrain.

---

## Error format

Every non-2xx response uses the same two-key envelope. There is no third variant.

```json
{ "error": "<machine_readable_code>", "detail": "<human readable message>" }
```

| Status | `error` | Real captured `detail` |
| --- | --- | --- |
| 404 | `session_not_found` | `Session 'nope' not found` |
| 422 | `validation_error` | `message: String should have at least 1 character` |
| 422 | `validation_error` | `message: String should have at most 1000 characters` |
| 500 | `internal_error` | `Internal server error` |

**The 500 body is identical in every environment.** The exception type and message are
never returned, not even in development; the full traceback is logged server-side with
the request id instead. `APP_ENV` changes exactly one thing: whether `/docs` is served.

### How each error is actually produced

| Cause | Result |
| --- | --- |
| `message` empty, too long, missing, or wrong type | **422 `validation_error`** from Pydantic schema validation, *before* the route runs. This is why an over-length body is `validation_error`, not `message_too_long`: the schema rejects it first. |
| `message` is only whitespace | **422 `validation_error`** — Pydantic's `min_length=1` rejects `" "` after the field's own strip. |
| Empty message reaching the pipeline directly (non-HTTP caller) | `EmptyMessageError` → 422 `empty_message` |
| Message over the limit reaching the pipeline directly | `MessageTooLongError` → 422 `message_too_long` |
| Unknown or deleted session id | **404 `session_not_found`** |
| Anything unexpected | **500 `internal_error`**, body is always the literal string above |

`empty_message` and `message_too_long` are therefore defensive: they guard the pipeline
for direct callers and cover the narrow window between schema validation and the route
handler. Over HTTP you will normally see `validation_error`.

## `session_id` lifecycle

1. **First message** — omit `session_id`. The server creates a session and returns an id.
2. **Every later message** — send the same id back. The server remembers the detected
   language, the intent it is waiting on (`pending_intent`) and the entities captured so
   far, so a slot-filling question can be answered across turns.
3. **Expiry** — a session idles out after `SESSION_TIMEOUT_MINUTES` (default 30). Expiry
   forgets the session; it never deletes the transcript from the database.
4. **Restart** — if the id is still in the database and has not expired, the server
   transparently **rehydrates** it (language, the last `MAX_CONTEXT_MESSAGES` messages,
   `last_intent`, `pending_intent` and entities) even though memory was lost.
5. **Unknown or expired id** — *not* an error. You get a brand-new session and a
   different `session_id` in the response. The old id is never resurrected.
6. **Deletion** — `DELETE /sessions/{id}` removes it from memory and the database; a
   later `GET` returns 404.

Lookup order is: in-memory (evicted if idle) → database rehydration (refused if the
stored `last_active` has expired) → create new. Nothing is created until both lookups
fail, so resuming a session after a restart never leaves an orphan session behind.

Session state lives in memory for speed and is mirrored to SQLite, so a **single-process**
deployment is assumed.

## Slot filling

When the bot needs a value it does not have (an order id), it replies with
`needs_input: true` and remembers `pending_intent`. The next message is treated as the
answer when it either:

* classifies as `fallback`, or
* contains the slot that the pending intent is waiting for (currently `order_id`).

The second rule matters with a trained model: a bare `ORD-12345` is classified as
`goodbye` at 0.56 confidence, which is just above the 0.55 threshold. Without the
entity rule the user would be told *"Goodbye! Have a great day."* after giving their
order number. A user who changes the subject (`hello`) still gets a normal greeting.

## `needs_input` and `suggestions`

| Field | Meaning | What the UI should do |
| --- | --- | --- |
| `needs_input` | `true` when the bot asked a clarifying question and is waiting for a slot. | Keep the conversation open; the next message may be the answer. |
| `suggestions` | Ready-made replies, already translated into the user's language. | Render as tappable buttons; sending one verbatim always works. |
| `is_follow_up` | `true` when this turn was resolved from session context rather than a fresh classification. | Optional: mark continuation turns. |
| `confidence` | The classifier's raw confidence for the predicted class, **not** for the resolved intent. During a slot fill it reflects the model's score for the class it actually predicted (e.g. `0.5607` for `goodbye`) while `intent` has been overridden to the pending `track_order`. | Do not threshold on this client-side; the server already applied `INTENT_CONFIDENCE_THRESHOLD`. |

`intent` is one of `greeting`, `goodbye`, `track_order`, `cancel_order`, `payment_issue`,
`refund`, `contact_support`, or `fallback`.

---

## `POST /chat`

Send one message and receive the reply plus the updated conversation state.

**Request**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `message` | string | yes | 1–1000 characters (`MAX_MESSAGE_CHARS`). Any language. |
| `session_id` | string \| null | no | Omit on the first turn. |

**Response 200**: `session_id`, `reply`, `intent`, `confidence`, `language`, `entities`,
`suggestions`, `is_follow_up`, `needs_input`.

### Turn 1 — capture the request id from the response header

```bash
curl -s -D- -o turn1.json -X POST localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-1" \
  -d '{"message":"where is my order"}'
```

```
HTTP/1.1 200 OK
x-request-id: demo-1
```

```json
{
  "session_id": "1786b986cdb1449bba6b2bdb417dcd3b",
  "reply": "Sure, I can help. Please share your order ID (for example ORD-12345).",
  "intent": "track_order",
  "confidence": 0.9907,
  "language": "en",
  "entities": {},
  "suggestions": [],
  "is_follow_up": false,
  "needs_input": true
}
```

### Turn 2 — fill the slot in the same session

```bash
curl -s -X POST localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"ORD-12345","session_id":"1786b986cdb1449bba6b2bdb417dcd3b"}'
```

```json
{
  "session_id": "1786b986cdb1449bba6b2bdb417dcd3b",
  "reply": "Your order ORD-12345 is being processed and will be packed shortly.",
  "intent": "track_order",
  "confidence": 0.5607,
  "language": "en",
  "entities": { "order_id": "ORD-12345" },
  "suggestions": [],
  "is_follow_up": true,
  "needs_input": false
}
```

Note `confidence: 0.5607` alongside `intent: "track_order"`: that is the model's score for
the class it predicted (`goodbye`), overridden because `ORD-12345` filled the pending
`order_id` slot. See *Slot filling* above.

### Hindi and Spanish

```bash
curl -s -X POST localhost:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"message":"मेरा ऑर्डर कहाँ है"}'
```

```json
{
  "session_id": "ab9ff4320a844108b559882b6e211e43",
  "reply": "ज़रूर, मैं मदद कर सकता हूं। कृपया अपना ऑर्डर आईडी बताएं (जैसे ORD-12345)।",
  "intent": "track_order",
  "confidence": 0.9904,
  "language": "hi",
  "entities": {},
  "suggestions": [],
  "is_follow_up": false,
  "needs_input": true
}
```

```bash
curl -s -X POST localhost:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"message":"hola, quiero cancelar mi pedido"}'
```

```json
{
  "session_id": "e5af0297557c4c68947e8c679649554f",
  "reply": "Claro, puedo ayudarte. Por favor, comparte tu número de pedido (por ejemplo ORD-12345).",
  "intent": "cancel_order",
  "confidence": 0.9903,
  "language": "es",
  "entities": {},
  "suggestions": [],
  "is_follow_up": false,
  "needs_input": true
}
```

A greeting, with the translated quick replies the UI should render as chips:

```json
{
  "intent": "greeting",
  "reply": "Hello! How can I help you today?",
  "suggestions": ["Track my order", "Request a refund", "Contact support"]
}
```

---

## `GET /health`

Liveness probe and which classifier is live. Captured with the promoted model:

```bash
curl -s localhost:8000/api/v1/health
```

```json
{
  "status": "ok",
  "app": "Multilingual Context-Aware Chatbot",
  "env": "development",
  "languages": ["en", "hi", "es"],
  "intent_model_loaded": true,
  "classifier": "transformer"
}
```

`classifier` is `"rules"` when `trained_models/intent_model` is absent or was not promoted
by the quality gate — a model that fails the gate is never placed there, so the stronger
rule-based fallback is not silently replaced by a weak model.

---

## `POST /sessions`

Create a session up front (optional — the first chat message creates one anyway).

```bash
curl -s -X POST localhost:8000/api/v1/sessions -H "Content-Type: application/json" -d '{}'
```

**201 Created**

```json
{
  "session_id": "f21eae72c7554177928cea16486b4f64",
  "language": null,
  "created_at": "2026-09-30T12:58:40.556509Z"
}
```

---

## `GET /sessions/{session_id}`

Full transcript, oldest first. Two messages are persisted per turn. Captured after the
two chat turns above:

```bash
curl -s localhost:8000/api/v1/sessions/1786b986cdb1449bba6b2bdb417dcd3b
```

```json
{
  "session_id": "1786b986cdb1449bba6b2bdb417dcd3b",
  "messages": [
    {
      "role": "user",
      "content": "where is my order",
      "language": "en",
      "intent": "track_order",
      "confidence": 0.9907015562057495,
      "created_at": "2026-09-30T12:58:40.676918Z"
    },
    {
      "role": "assistant",
      "content": "Sure, I can help. Please share your order ID (for example ORD-12345).",
      "language": "en",
      "intent": "track_order",
      "confidence": null,
      "created_at": "2026-09-30T12:58:40.677655Z"
    },
    {
      "role": "user",
      "content": "ORD-12345",
      "language": "en",
      "intent": "track_order",
      "confidence": 0.5606980323791504,
      "created_at": "2026-09-30T12:58:40.812957Z"
    },
    {
      "role": "assistant",
      "content": "Your order ORD-12345 is being processed and will be packed shortly.",
      "language": "en",
      "intent": "track_order",
      "confidence": null,
      "created_at": "2026-09-30T12:58:40.813612Z"
    }
  ]
}
```

`confidence` is `null` on assistant messages — only the classifier scores user input.
Timestamps are UTC with a `Z` suffix.

**404** `{"error": "session_not_found", "detail": "Session 'nope' not found"}`

---

## `DELETE /sessions/{session_id}`

Removes the session from memory and the database.

```bash
curl -s -o /dev/null -w "%{http_code}" -X DELETE \
  localhost:8000/api/v1/sessions/1786b986cdb1449bba6b2bdb417dcd3b
# 204
```

A subsequent `GET` on the same id returns **404**. Verified in sequence:

```
DELETE -> 204,  GET -> 404
```

---

## CORS

`CORS_ORIGINS` (default `http://localhost:5173,http://localhost:3000`) is applied with
credentials enabled and all methods/headers allowed. A captured preflight from an allowed
origin:

```
access-control-allow-origin: http://localhost:5173
access-control-allow-credentials: true
access-control-max-age: 600
x-request-id: 662041060d724b0ab583c9cfe345ffff
```

The exact `access-control-allow-methods` list is framework-version specific and
intentionally not documented; it is not part of the API contract. Add your dev server's
origin to `.env` (`CORS_ORIGINS`); it accepts a comma-separated list or a JSON array.