"""Language detection, preprocessing and config parsing behaviour."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.nlp.preprocessor import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Where is my order?", "en"),
        ("मेरा ऑर्डर कहाँ है", "hi"),
        ("mera order kahan hai", "hi"),
        ("hola, quiero cancelar mi pedido", "es"),
        ("¿dónde está mi pedido?", "es"),
        ("   ", "en"),
    ],
)
def test_detect_language(detector, text: str, expected: str) -> None:
    assert detector.detect(text).language == expected


def test_blank_text_reports_default_method(detector) -> None:
    assert detector.detect("   ").method == "default"


def test_devanagari_is_detected_by_script(detector) -> None:
    assert detector.detect("मेरा ऑर्डर कहाँ है").method == "script"


def test_normalize_collapses_whitespace_and_casefolds(pre) -> None:
    assert pre.process("  Where   IS   my ORDER?  ").normalized == "where is my order?"


def test_processed_text_keeps_original(pre) -> None:
    processed = pre.process("Hello World")
    assert processed.original == "Hello World"
    assert processed.normalized == "hello world"


def test_tokenize_strips_punctuation(pre) -> None:
    assert tokenize("hola, ¿qué tal?") == ["hola", "qué", "tal"]


def test_tokenize_keeps_devanagari_matra_together(pre) -> None:
    """Whitespace tokenisation must not split a Devanagari word into matras."""
    tokens = tokenize("मेरा ऑर्डर")
    assert "मेरा" in tokens
    assert "रा" not in tokens


def test_tokenize_drops_devanagari_danda(pre) -> None:
    assert tokenize("नमस्ते।") == ["नमस्ते"]


def test_supported_languages_parse_comma_separated() -> None:
    parsed = Settings(SUPPORTED_LANGUAGES="en,hi,es", DEFAULT_LANGUAGE="en")
    assert parsed.SUPPORTED_LANGUAGES == ["en", "hi", "es"]


def test_supported_languages_parse_json_list() -> None:
    parsed = Settings(SUPPORTED_LANGUAGES='["en", "hi"]', DEFAULT_LANGUAGE="hi")
    assert parsed.SUPPORTED_LANGUAGES == ["en", "hi"]


def test_invalid_default_language_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(SUPPORTED_LANGUAGES="en,hi", DEFAULT_LANGUAGE="de")


def test_relative_model_dir_resolves_under_backend() -> None:
    from app.core.config import BASE_DIR

    resolved = Settings(MODEL_DIR="trained_models/intent_model")
    assert resolved.MODEL_DIR.is_absolute()
    assert resolved.MODEL_DIR == BASE_DIR / "trained_models" / "intent_model"


def test_data_paths_are_derived(settings_obj: Settings) -> None:
    assert settings_obj.intents_file.name == "intents.json"
    assert settings_obj.intents_file.parent.name == "raw"
    assert settings_obj.translations_dir.name == "translations"


def test_is_production_flag() -> None:
    assert Settings(APP_ENV="production").is_production is True
    assert Settings(APP_ENV="development").is_production is False