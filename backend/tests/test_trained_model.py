"""Tests that exercise a REAL fine-tuned model.

These are skipped when ``trained_models/intent_model`` does not exist, so the normal
suite stays fast and never depends on a trained artefact. The model is loaded ONCE per
session: it is ~1.1 GB and loading it per test would dominate the runtime.

The rest of the suite deliberately points ``MODEL_DIR`` at a non-existent path
(see tests/conftest.py) and therefore always exercises the rule-based classifier.
"""

from __future__ import annotations

import json
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
requires_model = pytest.mark.skipif(
    not has_model,
    reason="no trained model in trained_models/intent_model (quality gate not promoted)",
)


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


@requires_model
def test_health_reports_the_transformer_client(client) -> None:
    """In the test app MODEL_DIR is fake, so this documents the ``classifier`` field."""
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