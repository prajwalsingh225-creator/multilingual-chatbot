"""Intent classification, entity extraction, handlers, registry and translations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.intents.handlers import HandlerContext, execute_intent
from app.intents.registry import registry
from app.nlp.intent_classifier import FALLBACK_INTENT


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", "greeting"),
        ("where is my order", "track_order"),
        ("cancel order ORD-12345", "cancel_order"),
        ("मेरा ऑर्डर कहाँ है", "track_order"),
        ("quiero un reembolso", "refund"),
        ("my payment failed", "payment_issue"),
        ("I need help", "contact_support"),
        ("asdf qwerty zxcv", FALLBACK_INTENT),
    ],
)
def test_rule_based_intent(classifier, pre, text: str, expected: str) -> None:
    assert classifier.predict(pre.process(text)).intent == expected


def test_classification_source_is_rules(classifier, pre) -> None:
    assert classifier.predict(pre.process("hello")).source == "rules"


def test_low_confidence_input_becomes_fallback(settings_obj, pre) -> None:
    from app.nlp.intent_classifier import IntentClassifier

    strict = IntentClassifier.from_intents_file(settings_obj.intents_file, 0.99, None)
    assert strict.predict(pre.process("asdf qwerty zxcv")).intent == FALLBACK_INTENT


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("track ORD-12345", "ORD-12345"),
        ("order #98765 please", "ORD-98765"),
        ("order ord 12345", "ORD-12345"),
        ("where is 12345", "ORD-12345"),
    ],
)
def test_order_id_extraction(extractor, text: str, expected: str) -> None:
    assert extractor.extract(text)["order_id"] == expected


def test_phone_extraction(extractor) -> None:
    assert extractor.extract("call 9876543210")["phone"] == "9876543210"


def test_email_extraction(extractor) -> None:
    assert extractor.extract("write to a.b@example.com")["email"] == "a.b@example.com"


def test_phone_is_not_mistaken_for_an_order_id(extractor) -> None:
    entities = extractor.extract("call 9876543210")
    assert "order_id" not in entities


def test_every_intent_in_the_data_file_has_a_handler(settings_obj) -> None:
    data = json.loads(settings_obj.intents_file.read_text(encoding="utf-8"))
    for item in data["intents"]:
        assert item["name"] in registry.intents(), f"no handler for {item['name']}"


def test_unknown_intent_falls_back_to_the_fallback_handler() -> None:
    ctx = HandlerContext(
        intent="does_not_exist", entities={}, language="en", session_id="s1"
    )
    assert execute_intent(ctx).template_key == "fallback"


def test_translation_files_share_identical_keys(settings_obj) -> None:
    stores = {
        lang: json.loads(path.read_text(encoding="utf-8"))
        for lang, path in (
            (p.stem, p) for p in sorted(settings_obj.translations_dir.glob("*.json"))
        )
    }
    assert set(stores) == set(settings_obj.SUPPORTED_LANGUAGES)
    reference = set(stores["en"])
    for lang, templates in stores.items():
        assert set(templates) == reference, f"{lang} key set differs from en"


def test_every_template_key_emitted_by_a_handler_exists(settings_obj) -> None:
    emitted = {
        execute_intent(
            HandlerContext(intent=i, entities={"order_id": "ORD-1"}, language="en", session_id="s")
        ).template_key
        for i in registry.intents()
    }
    english = json.loads(
        (settings_obj.translations_dir / "en.json").read_text(encoding="utf-8")
    )
    missing = emitted - set(english)
    assert not missing, f"template keys missing from en.json: {missing}"


def test_track_order_without_id_asks_and_sets_pending_intent(settings_obj) -> None:
    result = execute_intent(
        HandlerContext(intent="track_order", entities={}, language="en", session_id="s")
    )
    assert result.template_key == "ask_order_id"
    assert result.pending_intent == "track_order"


def test_track_order_with_id_returns_a_status(settings_obj) -> None:
    result = execute_intent(
        HandlerContext(
            intent="track_order",
            entities={"order_id": "ORD-12345"},
            language="en",
            session_id="s",
        )
    )
    assert result.template_key in {
        "order_status_processing",
        "order_status_shipped",
        "order_status_delivered",
    }


def test_contact_support_passes_email_and_phone(settings_obj) -> None:
    result = execute_intent(
        HandlerContext(
            intent="contact_support",
            entities={},
            language="en",
            session_id="s",
            email=settings_obj.SUPPORT_EMAIL,
            phone=settings_obj.SUPPORT_PHONE,
        )
    )
    assert result.params == {
        "email": settings_obj.SUPPORT_EMAIL,
        "phone": settings_obj.SUPPORT_PHONE,
    }


def test_translator_fills_params(settings_obj) -> None:
    from app.response.templates import TemplateStore
    from app.response.translator import Translator

    translator = Translator(TemplateStore(settings_obj.translations_dir), "en")
    text = translator.translate("order_status_shipped", "en", order_id="ORD-12345")
    assert "ORD-12345" in text


def test_translator_leaves_missing_params_visible(settings_obj) -> None:
    from app.response.templates import TemplateStore
    from app.response.translator import Translator

    translator = Translator(TemplateStore(settings_obj.translations_dir), "en")
    assert "{order_id}" in translator.translate("order_status_shipped", "en")


def test_translator_falls_back_to_default_language(settings_obj) -> None:
    from app.response.templates import TemplateStore
    from app.response.translator import Translator

    translator = Translator(TemplateStore(settings_obj.translations_dir), "en")
    assert translator.translate("greeting", "zz") == translator.translate("greeting", "en")


def test_translator_uses_the_requested_language(settings_obj) -> None:
    from app.response.templates import TemplateStore
    from app.response.translator import Translator

    translator = Translator(TemplateStore(settings_obj.translations_dir), "en")
    assert translator.translate("greeting", "hi") != translator.translate("greeting", "en")


def test_spanish_templates_keep_accents(settings_obj) -> None:
    spanish = json.loads(
        (settings_obj.translations_dir / "es.json").read_text(encoding="utf-8")
    )
    assert "ó" in spanish["goodbye"]


def test_hindi_templates_use_devanagari(settings_obj) -> None:
    hindi = json.loads((settings_obj.translations_dir / "hi.json").read_text(encoding="utf-8"))
    assert any("ऀ" <= ch <= "ॿ" for ch in hindi["greeting"])

# --- training-data quality (phase 4.1) ----------------------------------------

MIN_EXAMPLES_PER_INTENT_AND_LANGUAGE = 30


def _intents_data(settings_obj) -> list[dict]:
    return json.loads(settings_obj.intents_file.read_text(encoding="utf-8"))["intents"]


def test_every_intent_has_enough_examples_per_language(settings_obj) -> None:
    short = {
        item["name"]: {lang: len(examples) for lang, examples in item["examples"].items()}
        for item in _intents_data(settings_obj)
        if any(
            len(examples) < MIN_EXAMPLES_PER_INTENT_AND_LANGUAGE
            for examples in item["examples"].values()
        )
    }
    assert not short, f"intents with too few examples: {short}"


def test_every_intent_covers_all_supported_languages(settings_obj) -> None:
    for item in _intents_data(settings_obj):
        assert set(item["examples"]) == set(settings_obj.SUPPORTED_LANGUAGES), item["name"]


def test_no_duplicate_examples_within_an_intent(settings_obj) -> None:
    for item in _intents_data(settings_obj):
        for lang, examples in item["examples"].items():
            assert len(set(examples)) == len(examples), f"{item['name']}/{lang} has duplicates"


def test_hindi_examples_include_devanagari(settings_obj) -> None:
    for item in _intents_data(settings_obj):
        hindi = item["examples"]["hi"]
        assert any(any("ऀ" <= ch <= "ॿ" for ch in ex) for ex in hindi), item["name"]


def test_hindi_examples_include_hinglish(settings_obj) -> None:
    """The hi bucket must also carry romanised Hindi so Hinglish can be classified."""
    for item in _intents_data(settings_obj):
        romanised = [
            ex
            for ex in item["examples"]["hi"]
            if not any("ऀ" <= ch <= "ॿ" for ch in ex)
        ]
        assert romanised, f"{item['name']} has no Hinglish examples"


def test_dataset_includes_typos(settings_obj) -> None:
    """Typos must be represented, otherwise the model never learns to tolerate them."""
    typos = ["cancle my order", "paymnt failed", "rembolso", "nesesito ayuda", "helo"]
    present = {
        ex
        for item in _intents_data(settings_obj)
        for examples in item["examples"].values()
        for ex in examples
    }
    assert set(typos) & present, "no deliberate typo examples found in intents.json"


def test_prepared_dataset_is_in_sync_with_raw_intents(settings_obj) -> None:
    """Regenerate with: uv run python -m training.train_intent --prepare-only"""
    from training.train_intent import build_dataset

    train_file = settings_obj.DATA_DIR / "processed" / "train.json"
    assert train_file.exists(), "run `uv run python -m training.train_intent --prepare-only`"
    on_disk = json.loads(train_file.read_text(encoding="utf-8"))
    # build_dataset writes out_file as a side effect, so build into a temp path and
    # compare content, never let a failing assertion clobber the real train.json.
    with tempfile.TemporaryDirectory() as tmpdir:
        expected = build_dataset(
            settings_obj.intents_file,
            Path(tmpdir) / "train.json",
            settings_obj.DATA_DIR / "raw" / "out_of_scope.json",
        )
    # Content, not just count: a len() check would pass if one example were dropped and
    # another added, which is exactly the drift this test exists to catch.
    def key(record: dict) -> tuple[str, str, str]:
        return (record["text"], record["label"], record["language"])

    on_disk_keys = {key(r) for r in on_disk}
    expected_keys = {key(r) for r in expected}

    missing = expected_keys - on_disk_keys
    unexpected = on_disk_keys - expected_keys
    assert not missing, (
        f"{len(missing)}/{len(expected_keys)} regenerated records are absent from "
        f"train.json, e.g. {sorted(missing)[:5]}"
    )
    assert not unexpected, (
        f"{len(unexpected)}/{len(on_disk_keys)} train.json records are not reproducible "
        f"from the raw files, e.g. {sorted(unexpected)[:5]}"
    )
    assert len(on_disk) == len(expected_keys), (
        f"train.json has {len(on_disk)} rows but only {len(on_disk_keys)} distinct "
        f"(text, label, language) tuples; regeneration yields {len(expected_keys)}"
    )


def test_prepared_dataset_text_is_normalised(settings_obj) -> None:
    from app.nlp.preprocessor import normalize_text

    train_file = settings_obj.DATA_DIR / "processed" / "train.json"
    records = json.loads(train_file.read_text(encoding="utf-8"))
    assert records
    for record in records[:50]:
        assert record["text"] == normalize_text(record["text"])


def test_split_keeps_every_label_in_both_partitions(settings_obj) -> None:
    from training.train_intent import split_dataset

    train_file = settings_obj.DATA_DIR / "processed" / "train.json"
    records = json.loads(train_file.read_text(encoding="utf-8"))
    train, val = split_dataset(records, 0.2, 42)
    assert {r["label"] for r in train} == {r["label"] for r in val}
    assert len(train) + len(val) == len(records)
