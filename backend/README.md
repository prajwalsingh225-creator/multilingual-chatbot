# Multilingual Context-Aware Chatbot — Backend

FastAPI backend for a customer-support chatbot that classifies intent in English, Hindi
and Spanish, holds multi-turn context (including slot filling), and falls back to a
rule-based classifier when no trained model is available.

- API docs: [`docs/API.md`](docs/API.md)
- Exported OpenAPI schema: `docs/openapi.json` (regenerate with `scripts/export_openapi.py`)
- Interactive docs while running: <http://localhost:8000/docs>

---

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd backend
uv sync --group dev            # runtime + test/lint/mypy tools
cp .env.example .env           # then edit if you need non-default values
```

`.env` is git-ignored. Every key in `app/core/config.py` has a commented default, so the
app runs with no `.env` at all; the file exists to show the available knobs.

### Run

```bash
uv run uvicorn app.main:app --reload --port 8000
uv run uvicorn app.main:app --reload --port 8000 --host 0.0.0.0   # expose to other devices
```

`/docs` is served when `APP_ENV` is anything other than `production`; in production only
`/api/v1/*` is reachable.

## Tests and checks

```bash
uv run pytest -q            # full suite
uv run pytest -q tests/test_robustness.py   # phase 4.3 behaviours
uv run ruff check .
uv run mypy app
```

The suite points `MODEL_DIR` at a non-existent directory so it never loads the 1.1 GB
model and runs in a few seconds. Tests that genuinely need a trained model live in
`tests/test_trained_model.py` and **skip themselves** when
`trained_models/intent_model` is absent.

## Train a model

```bash
uv sync --group training                              # pulls torch + transformers
uv run --group training python -m training.train_intent --prepare-only
uv run --group training python -m training.train_intent
```

- `training/config.yaml` holds the base model, epochs, learning rate, batch size and the
  train/val split ratio and seed.
- `--epochs N` and `--base-model NAME` override the config for one run.
- Training text goes through the same `normalize_text()` used at inference, so the two
  cannot drift apart.
- Expect roughly 8 minutes for 10 epochs on an 8-core Apple M2 on **CPU**. There is no
  CUDA here; `MPS` is not wired up.

## Evaluate

```bash
uv run python -m training.evaluate                                  # rules only, no torch needed
uv run --group training python -m training.evaluate \
    --with-transformer --model-dir trained_models/intent_model --json report.json
```

Everything is scored on the same held-out validation split. The rule baseline is built
from the **train split only** — scoring it on the full corpus it was generated from
produces a meaningless 1.0000.

The report includes overall accuracy, macro-F1, per-intent and per-language accuracy, a
confusion matrix, train-split accuracy (which separates under-fitting from over-fitting),
and a leakage check that counts exact and near duplicates across the split boundary.

```bash
uv run --group training python -m training.offtopic_report   # fallback rate on off-topic input
```

## Promote a model (quality gate)

A model is only ever placed in `trained_models/intent_model` if it passes:

> held-out accuracy ≥ 0.90 **and** every intent ≥ 0.80

Train into a staging directory, measure it, then promote only on a pass:

```bash
uv run --group training python -m training.train_intent \
    --model-dir trained_models/staging/run-02
uv run --group training python -m training.evaluate \
    --with-transformer --model-dir trained_models/staging/run-02 --json run-02.json
# if the numbers clear the gate:
cp trained_models/staging/run-02/{config.json,model.safetensors,tokenizer.json,tokenizer_config.json} \
   trained_models/intent_model/
```

Each staging run keeps a `training_meta.json` recording its config, metrics, eval-loss
curve and git commit.

**If the gate fails, `trained_models/intent_model` must not exist.** The app loads that
directory unconditionally and prefers it over the rules, so a weak model there silently
replaces a stronger fallback: `/health` would report `"classifier": "transformer"` while
accuracy dropped from 0.74 to 0.35.

Current promoted model: `xlm-roberta-base`, 10 epochs, lr 3e-5 — **0.9441** held-out
accuracy, macro-F1 0.9442, worst intent (`greeting`) 0.8571. The rule baseline on the
same split is 0.7413.

## What the classifier supports

Seven intents: `greeting`, `goodbye`, `track_order`, `cancel_order`, `payment_issue`,
`refund`, `contact_support`, plus `fallback` for anything under
`INTENT_CONFIDENCE_THRESHOLD` (default 0.55).

Order status is a deterministic mock (`app/intents/orders.py`): the status is derived
from the digits in the order id. Swap it for a real order service; nothing else depends
on it.

## API summary

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/chat` | Send one turn; returns the reply and conversation state |
| `GET` | `/api/v1/health` | Liveness plus which classifier is active |
| `POST` | `/api/v1/sessions` | Create a session (optional) |
| `GET` | `/api/v1/sessions/{id}` | Full transcript, oldest first |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session (204) |

Errors are always `{"error": "<code>", "detail": "<message>"}`. Every response carries
`X-Request-ID`; send your own to have it echoed and appear in the server logs.

## Regenerate the OpenAPI schema

Do this **last**, after any API change:

```bash
uv run python scripts/export_openapi.py
uv run python scripts/export_openapi.py --output docs/openapi.json --indent 2
```

## Project layout

```
app/
  api/            routes + request/response schemas
  conversation/   in-memory sessions, transcript, context resolution
  core/           settings, logging, exceptions
  database/       SQLAlchemy models + repository
  intents/        intent -> handler mapping and reply templates
  models/         transformer inference + safe loading
  nlp/            preprocessing, language detection, entity extraction, classification
  response/       translation and reply rendering
  pipeline.py     one chat turn, end to end
data/raw/         intents.json, translations
data/processed/   generated train.json (regenerated, do not hand-edit)
docs/             API.md + openapi.json
scripts/          export_openapi.py
tests/
training/         dataset prep, fine-tuning, evaluation, off-topic report
```

## Troubleshooting

**zsh: `max_length=64 # comment` breaks the command.** In zsh an unquoted `#` starts a
comment *only* at the start of a word, but after `=` it is parsed as a literal argument.
`uv run ... --max_length=64 # keep it short` passes `#`, `keep`, `it`, `short` to the
program. Put the comment on its own line, or quote it.

**`sqlite3.OperationalError: no such column: last_active`** (or `pending_intent`,
`entities`). The database predates a schema change. `create_all()` creates missing tables
but never adds columns to existing ones, and there is no migration tool here. Back it up
and let startup recreate it:

```bash
cp chatbot.db ~/chatbot.db.bak && rm chatbot.db
uv run python -c "from app.database.database import init_db; init_db()"
```

Losing the file loses local conversation history; that is the intended trade-off for a
development database.

**"No trained intent model at ...; using rule-based fallback"** is expected and harmless
whenever the gate has not been run. `/health` shows `"classifier": "rules"`.

**`ModelLoadError: Failed to load intent model from ...`** means the directory exists but
is unreadable — usually a half-copied model or a corrupted `config.json`. Delete it and
re-copy, or remove it entirely to fall back to rules.

**Slot filling answers with "Goodbye!"** — you are probably running a model older than
the `ContextManager` entity rule. A bare `ORD-12345` is classified `goodbye` at 0.56,
just over the threshold; the entity rule in `app/conversation/context_manager.py` is what
overrides it. Covered by `test_a_bare_order_id_fills_the_pending_slot`.

**`StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated`** comes
from the installed FastAPI/Starlette version, not from this code, and is emitted once per
session at import time.

## Known limitations

- **Off-topic input is classified too confidently.** On 60 deliberately off-topic
  sentences the promoted model falls back only 28% of the time, usually picking
  `goodbye` or `contact_support` at 0.9+ confidence. The seven intents have no
  out-of-distribution examples, so raising `INTENT_CONFIDENCE_THRESHOLD` does not help.
  The reliable fix is adding explicit off-topic training data, which would mean an
  eighth `fallback` class and changes the documented intent contract.
- In-memory sessions assume a single process; a multi-worker deployment would need a
  shared session store.
- Order status is mocked (see above).