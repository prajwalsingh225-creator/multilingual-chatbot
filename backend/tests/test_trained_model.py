"""Tests that exercise a REAL fine-tuned model.

These are opt-in and NEVER run in the default suite. Two conditions must both hold:

1. ``RUN_MODEL_TESTS=1`` is set (the suite must stay fast and must never depend on a
   ~1.1 GB artefact), and
2. ``trained_models/intent_model`` actually exists (otherwise there is nothing to test).

Run them with::

    RUN_MODEL_TESTS=1 uv run pytest -m real_model

The model is loaded ONCE per session by the ``real_classifier`` fixture, shared by every
test in this module. The rest of the suite deliberately points ``MODEL_DIR`` at a
non-existent path (see tests/conftest.py) and always exercises the rule-based classifier.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from app.core.config import settings
from app.core.exceptions import ModelLoadError
from app.models.model_loader import load_intent_model
from app.nlp.intent_classifier import FALLBACK_INTENT, IntentClassifier
from app.nlp.preprocessor import Preprocessor

MODEL_DIR = Path(__file__).resolve().parents[1] / "trained_models" / "intent_model"
has_model = (MODEL_DIR / "config.json").exists()
model_tests_enabled = os.environ.get("RUN_MODEL_TESTS") == "1"

_skip_without_model = pytest.mark.skipif(
    not has_model or not model_tests_enabled,
    reason=(
        "real-model tests are opt-in: set RUN_MODEL_TESTS=1 and promote a model to "
        f"trained_models/intent_model (has_model={has_model}, "
        f"RUN_MODEL_TESTS={os.environ.get('RUN_MODEL_TESTS')})"
    ),
)


def requires_model(func):
    """Tag a test with the ``real_model`` marker *and* the skip gate in one decorator."""
    return _skip_without_model(pytest.mark.real_model(func))


@pytest.fixture(scope="session")
def real_classifier() -> IntentClassifier:
    """Rule classifier that ALSO has the real transformer attached. Session-scoped."""
    transformer = load_intent_model(MODEL_DIR)
    assert transformer is not None
    return IntentClassifier(
        settings.INTENT_CONFIDENCE_THRESHOLD, transformer, None
    )


@pytest.fixture(scope="session")
def pre() -> Preprocessor:
    return Preprocessor()


@requires_model
def test_the_promoted_model_loads(real_classifier) -> None:
    assert real_classifier.transformer is not None
    assert len(real_classifier.transformer.labels) == 7


@requires_model
def test_the_model_covers_every_intent(real_classifier) -> None:
    labels = set(real_classifier.transformer.labels)
    assert labels == {
        "greeting",
        "goodbye",
        "track_order",
        "cancel_order",
        "payment_issue",
        "refund",
        "contact_support",
    }


@requires_model
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("where is my order", "track_order"),
        ("i want to cancel my order", "cancel_order"),
        ("my payment was declined", "payment_issue"),
        ("i need a refund", "refund"),
        ("hello there", "greeting"),
        ("bye", "goodbye"),
        ("connect me to a human agent", "contact_support"),
    ],
)
def test_english_intents_are_classified(real_classifier, pre, text, expected) -> None:
    prediction = real_classifier.predict(pre.process(text))
    assert prediction.intent == expected
    assert prediction.source == "transformer"


@requires_model
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("मेरा ऑर्डर कहाँ है", "track_order"),
        ("मुझे अपना ऑर्डर रद्द करना है", "cancel_order"),
        ("मेरा पेमेंट नहीं हो रहा", "payment_issue"),
        ("मुझे रिफंड चाहिए", "refund"),
    ],
)
def test_hindi_intents_are_classified(real_classifier, pre, text, expected) -> None:
    assert real_classifier.predict(pre.process(text)).intent == expected


@requires_model
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("donde esta mi pedido", "track_order"),
        ("quiero cancelar mi pedido", "cancel_order"),
        ("mi pago no se ha aceptado", "payment_issue"),
        ("quiero un reembolso", "refund"),
    ],
)
def test_spanish_intents_are_classified(real_classifier, pre, text, expected) -> None:
    assert real_classifier.predict(pre.process(text)).intent == expected


@requires_model
def test_gibberish_does_not_crash_the_model(real_classifier, pre) -> None:
    prediction = real_classifier.predict(pre.process("!!!@@@###"))
    assert prediction.intent in {FALLBACK_INTENT, *real_classifier.transformer.labels}


@requires_model
def test_low_confidence_falls_back_to_fallback_intent(real_classifier, pre) -> None:
    """A prediction below the threshold must surface as ``fallback``, not a real intent."""
    prediction = real_classifier.predict(pre.process("hmm"))
    if prediction.confidence < settings.INTENT_CONFIDENCE_THRESHOLD:
        assert prediction.intent == FALLBACK_INTENT


def test_health_reports_the_transformer_client(client) -> None:
    """No real model needed: MODEL_DIR is fake here, so this only checks that the
    ``classifier`` field stays consistent with ``intent_model_loaded``."""
    body = client.get("/api/v1/health").json()
    assert body["classifier"] in {"rules", "transformer"}
    assert body["classifier"] == ("transformer" if body["intent_model_loaded"] else "rules")


# --- corrupt model handling (no real model required) --------------------------


def test_a_corrupt_model_directory_raises_a_clear_model_load_error(tmp_path) -> None:
    """A model dir that exists but cannot be parsed must fail loudly, not silently."""
    corrupt = tmp_path / "intent_model"
    corrupt.mkdir()
    (corrupt / "config.json").write_text("{ this is not valid json", encoding="utf-8")

    with pytest.raises(ModelLoadError) as excinfo:
        load_intent_model(corrupt)
    message = str(excinfo.value)
    assert str(corrupt) in message, "the error must name the directory it tried"
    assert "Failed to load intent model" in message


def test_a_missing_model_directory_returns_none(tmp_path) -> None:
    """No config.json at all is the normal 'not trained yet' case, not an error."""
    assert load_intent_model(tmp_path / "does-not-exist") is None


def test_a_model_config_without_weights_raises(tmp_path) -> None:
    """Valid config.json but no weights: the failure must be reported, not swallowed."""
    partial = tmp_path / "intent_model"
    partial.mkdir()
    (partial / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["XLMRobertaForSequenceClassification"],
                "model_type": "xlm-roberta",
                "id2label": {"0": "greeting"},
                "num_labels": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError):
        load_intent_model(partial)


@requires_model
def test_the_real_model_directory_is_not_emptied_by_a_failed_load(tmp_path) -> None:
    """Guard against a loader that truncates or moves the directory on failure."""
    before = sorted(p.name for p in MODEL_DIR.iterdir())
    broken = tmp_path / "broken"
    shutil.copytree(MODEL_DIR, broken)
    (broken / "config.json").write_text("nonsense", encoding="utf-8")
    with pytest.raises(ModelLoadError):
        load_intent_model(broken)
    assert sorted(p.name for p in MODEL_DIR.iterdir()) == before

# --- slot filling with the real model ---------------------------------------


@pytest.fixture
def model_app(real_classifier, monkeypatch):
    """An app wired to the REAL classifier, on the normal throwaway test database."""
    from fastapi.testclient import TestClient

    import app.main as main_module
    from app.core.config import settings as app_settings
    from app.pipeline import build_pipeline as real_build_pipeline

    def factory(config):
        pipeline = real_build_pipeline(config)
        pipeline.classifier = real_classifier
        return pipeline

    monkeypatch.setattr(main_module, "build_pipeline", factory)
    app = main_module.create_app()
    assert app_settings.MODEL_DIR.name  # the test database comes from conftest
    with TestClient(app) as client:
        yield client


@requires_model
def test_a_bare_order_id_fills_the_pending_slot(model_app) -> None:
    """Regression: the model scores "ORD-12345" as "goodbye" at 0.56, just over the
    0.55 threshold, which used to drop the pending intent and answer "Goodbye!"."""
    client = model_app
    first = client.post("/api/v1/chat", json={"message": "where is my order"}).json()
    assert first["needs_input"] is True

    second = client.post(
        "/api/v1/chat",
        json={"message": "ORD-12345", "session_id": first["session_id"]},
    ).json()
    assert second["intent"] == "track_order", second
    assert second["is_follow_up"] is True
    assert second["needs_input"] is False
    assert second["entities"] == {"order_id": "ORD-12345"}
    assert "Goodbye" not in second["reply"]


@requires_model
def test_a_greeting_while_a_slot_is_pending_is_still_a_greeting(model_app) -> None:
    """Changing the subject must not be swallowed by the pending intent."""
    client = model_app
    first = client.post("/api/v1/chat", json={"message": "where is my order"}).json()
    second = client.post(
        "/api/v1/chat", json={"message": "hello", "session_id": first["session_id"]}
    ).json()
    assert second["intent"] == "greeting", second
    assert second["needs_input"] is False


@requires_model
def test_slot_filling_survives_a_restart_with_the_real_model(model_app) -> None:
    client = model_app
    first = client.post("/api/v1/chat", json={"message": "where is my order"}).json()
    session_id = first["session_id"]
    for stored in list(client.app.state.pipeline.sessions._sessions):
        client.app.state.pipeline.sessions.delete(stored)

    resumed = client.post(
        "/api/v1/chat", json={"message": "ORD-12345", "session_id": session_id}
    ).json()
    assert resumed["session_id"] == session_id
    assert resumed["intent"] == "track_order"
    assert resumed["is_follow_up"] is True
    assert resumed["entities"] == {"order_id": "ORD-12345"}
