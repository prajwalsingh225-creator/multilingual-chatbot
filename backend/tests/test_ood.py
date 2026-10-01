"""Phase 5: out-of-domain (OOD) rejection.

Three groups of tests:

  1. integrity of the frozen eval set and the training negatives -- schema, counts, and
     the guarantee that the eval set shares no normalized text with anything used for
     training. The eval set is never trained on, so these tests are what stop that from
     silently changing.
  2. classifier mapping -- the internal ``out_of_scope`` label must surface as the public
     ``"fallback"`` intent, must never leak to the API, and must leave the rule classifier
     working (rules have no out_of_scope examples and must be unaffected).
  3. context safety -- the class that makes slot answers look out-of-scope must not break
     multi-turn slot filling. These use a stub classifier, so they are model-free and fast.

Real-model threshold checks live in tests/test_trained_model.py and stay opt-in.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from app.conversation.context_manager import ContextManager
from app.conversation.memory import ConversationMemory
from app.conversation.session_manager import Session
from app.core.config import settings
from app.nlp.intent_classifier import (
    FALLBACK_INTENT,
    OUT_OF_SCOPE_LABEL,
    IntentClassifier,
    IntentPrediction,
)
from app.nlp.preprocessor import Preprocessor, normalize_text

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVAL_FILE = BACKEND_DIR / "data" / "eval" / "ood_eval.json"

# SHA-256 of the frozen eval file. Changing the eval set is allowed, but it must be a
# deliberate act: update this constant and reports/eval_set_provenance.md in the same
# commit, and say why. Without this, an eval file could be edited after seeing a
# candidate's scores and the OOD numbers would silently mean something else.
# Last updated for eval set 1.1: two in-domain track_order rows removed from ood,
# 26 Hinglish control rows added.
EVAL_SHA256 = "2e7d577145952b5dd85a661532e7572bb1ad9c582be9ec4500551f87293faf21"

INTENTS_FILE = BACKEND_DIR / "data" / "raw" / "intents.json"
OUT_OF_SCOPE_FILE = BACKEND_DIR / "data" / "raw" / "out_of_scope.json"
TRAIN_FILE = BACKEND_DIR / "data" / "processed" / "train.json"

OOD_LANGUAGES = {"en", "hi", "es"}
ALL_LANGUAGES = {"en", "hi", "es", "hinglish"}

# Unicode word characters, so Devanagari matras stay attached to their consonant.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def eval_set() -> dict[str, Any]:
    return _load(EVAL_FILE)


@pytest.fixture(scope="session")
def out_of_scope() -> dict[str, Any]:
    return _load(OUT_OF_SCOPE_FILE)


@pytest.fixture(scope="session")
def training_texts() -> set[str]:
    """Every normalized string the model was trained on."""
    texts: set[str] = set()
    intents = _load(INTENTS_FILE)
    for intent in intents["intents"]:
        for examples in intent["examples"].values():
            texts.update(normalize_text(e) for e in examples)
    negatives = _load(OUT_OF_SCOPE_FILE)
    for examples in negatives["examples"].values():
        texts.update(normalize_text(e) for e in examples)
    processed = TRAIN_FILE
    if processed.exists():
        for record in json.loads(processed.read_text(encoding="utf-8")):
            texts.add(normalize_text(record["text"]))
    return texts


# --- 1. eval set + negative integrity ------------------------------------------


def test_eval_set_has_the_required_schema(eval_set) -> None:
    assert set(eval_set) >= {"version", "ood", "in_domain"}
    assert eval_set["version"], "the eval set must be versioned"
    for row in eval_set["ood"] + eval_set["in_domain"]:
        assert set(row) == {"text", "language", "kind"}, row
        assert row["text"].strip(), row
        assert normalize_text(row["text"]), f"{row['text']!r} normalizes to nothing"


def test_eval_set_meets_the_per_language_minimums(eval_set) -> None:
    counts: dict[str, int] = {}
    for row in eval_set["ood"]:
        counts[row["language"]] = counts.get(row["language"], 0) + 1
    for language in OOD_LANGUAGES:
        assert counts.get(language, 0) >= 40, f"{language}: {counts.get(language, 0)} < 40"
    assert counts.get("hinglish", 0) >= 20, counts

    control: dict[str, int] = {}
    for row in eval_set["in_domain"]:
        control[row["language"]] = control.get(row["language"], 0) + 1
    for language in OOD_LANGUAGES:
        assert control.get(language, 0) >= 20, f"{language}: {control.get(language, 0)} < 20"


def test_eval_set_covers_the_required_failure_kinds(eval_set) -> None:
    kinds = {row["kind"] for row in eval_set["ood"]}
    required = {
        "gibberish",
        "chit_chat",
        "unrelated_task",
        "other_domain",
        "out_of_language",
        "short_long",
        "numbers_only",
    }
    assert required <= kinds, f"missing kinds: {required - kinds}"


def test_the_eval_set_matches_its_recorded_hash() -> None:
    """The frozen eval set must not change silently.

    Every OOD number in the run table is measured against these bytes. If the file moves
    without EVAL_SHA256 and reports/eval_set_provenance.md being updated in the same
    commit, the numbers no longer describe the set they claim to describe -- and the most
    likely reason to edit it is that a candidate scored badly on it.
    """
    actual = hashlib.sha256(EVAL_FILE.read_bytes()).hexdigest()
    assert actual == EVAL_SHA256, (
        "data/eval/ood_eval.json changed without its recorded hash being updated.\n"
        f"  recorded: {EVAL_SHA256}\n"
        f"  actual:   {actual}\n"
        "If this change is intentional, update EVAL_SHA256 in tests/test_ood.py and the\n"
        "sha256 + change log in reports/eval_set_provenance.md in the same commit, and\n"
        "re-measure every baseline that is quoted against this file."
    )


def _word_set(text: str) -> frozenset[str]:
    """Distinct word tokens, accent-folded, so word order and diacritics do not hide a match."""
    folded = "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    )
    return frozenset(_WORD_RE.findall(folded))


def _is_content_free(text: str) -> bool:
    """True for '???', '@@@@###$$$', '🙏' and 'जीजीजीजी' -- no propositional content.

    A row qualifies when it yields no word tokens at all (punctuation, emoji, digits) or
    exactly one distinct word token that is itself a short run of one or two characters
    repeated ('जीजीजीजी' -> {'जीजीजीजी'}). Those are deliberate gibberish probes, so a
    token-set collision with a training row that is also gibberish ('@@@@', 'जीजीजी') is
    not leakage.

    This test exists for real sentences that were mislabelled -- 'मेरा सामान कहाँ है' is a
    reordering of 'सामान कहाँ है मेरा' and is a genuine track_order request -- so anything
    carrying several distinct words is treated as a claim and must be checked.
    """
    tokens = _word_set(text)
    if not tokens:
        return True
    if len(tokens) == 1:
        only = next(iter(tokens))
        return len(only) <= 2 or len(set(only)) <= 2
    return False


def test_the_eval_set_shares_no_token_multiset_with_training(eval_set) -> None:
    """Catch eval rows that are permutations of a training example.

    normalize_text casefolds and NFKC-normalises but keeps diacritics, so
    'dónde está mi paquete' never matched the training example 'donde esta mi paquete'.
    Both that row and 'मेरा सामान कहाँ है' (a reordering of 'सामान कहाँ है मेरा') were real
    track_order requests sitting in the ood set, counting correct predictions as failures.
    """
    train_texts: set[str] = set()
    intents = _load(INTENTS_FILE)
    for intent in intents["intents"]:
        for examples in intent["examples"].values():
            train_texts.update(normalize_text(e) for e in examples)
    negatives = _load(OUT_OF_SCOPE_FILE)
    for examples in negatives["examples"].values():
        train_texts.update(normalize_text(e) for e in examples)

    train_word_sets = {_word_set(t) for t in train_texts}
    offenders: list[tuple[str, str, str]] = []
    for row in eval_set["ood"] + eval_set["in_domain"]:
        text = row["text"]
        if _is_content_free(text):
            continue
        if _word_set(normalize_text(text)) in train_word_sets:
            offenders.append((row["language"], row["kind"], text))
    assert not offenders, (
        f"{len(offenders)}/{len(eval_set['ood']) + len(eval_set['in_domain'])} eval rows "
        f"are a permutation of a training example and are really in-domain: {offenders[:5]}"
    )


def test_the_eval_set_never_overlaps_training_data(eval_set, training_texts) -> None:
    """The frozen eval set must share no normalized text with anything trained on.

    If this fails, the OOD numbers are measuring training data and mean nothing.
    """
    eval_texts = {normalize_text(r["text"]) for r in eval_set["ood"] + eval_set["in_domain"]}
    leaked = eval_texts & training_texts
    assert not leaked, f"eval text leaked into training data: {sorted(leaked)[:5]}"


def test_the_eval_set_has_no_internal_duplicates(eval_set) -> None:
    texts = [normalize_text(r["text"]) for r in eval_set["ood"] + eval_set["in_domain"]]
    assert len(texts) == len(set(texts)), "duplicate rows make the rates misleading"


def test_training_negatives_have_no_duplicates_and_no_label_noise(
    out_of_scope, training_texts
) -> None:
    texts = [
        normalize_text(e) for examples in out_of_scope["examples"].values() for e in examples
    ]
    dupes = {t for t in texts if texts.count(t) > 1}
    assert not dupes, f"duplicated negatives: {sorted(dupes)[:5]}"

    # A negative must never be text the model is meant to classify as a real intent.
    intents = _load(INTENTS_FILE)
    real_texts = {
        normalize_text(e)
        for intent in intents["intents"]
        for examples in intent["examples"].values()
        for e in examples
    }
    assert not (set(texts) & real_texts)


def test_training_negatives_meet_the_per_language_minimums(out_of_scope) -> None:
    counts = {lang: len(items) for lang, items in out_of_scope["examples"].items()}
    for language in OOD_LANGUAGES:
        assert counts.get(language, 0) >= 100, f"{language}: {counts.get(language, 0)} < 100"
    assert counts.get("hinglish", 0) >= 40, counts
    assert set(counts) == ALL_LANGUAGES, counts


def test_intents_file_still_has_exactly_seven_public_intents() -> None:
    """out_of_scope is internal; the public intent list must not grow."""
    names = {intent["name"] for intent in _load(INTENTS_FILE)["intents"]}
    assert len(names) == 7, names
    assert OUT_OF_SCOPE_LABEL not in names


# --- 2. classifier mapping ------------------------------------------------------


class _StubTransformer:
    """Returns a fixed label at a fixed confidence, to test the mapping in isolation."""

    def __init__(self, label: str, confidence: float) -> None:
        self._label = label
        self._confidence = confidence

    @property
    def labels(self) -> list[str]:
        return [self._label]

    def predict(self, text: str, top_k: int = 1) -> list[tuple[str, float]]:
        return [(self._label, self._confidence)]


def test_out_of_scope_is_mapped_to_fallback() -> None:
    """The internal label must surface as the public fallback intent."""
    classifier = IntentClassifier(0.55, _StubTransformer(OUT_OF_SCOPE_LABEL, 0.99), None)
    prediction = classifier.predict(Preprocessor().process("tell me a joke"))
    assert prediction.intent == FALLBACK_INTENT
    assert prediction.intent != OUT_OF_SCOPE_LABEL
    # ...but the raw label is preserved for logging.
    assert prediction.raw_intent == OUT_OF_SCOPE_LABEL


def test_out_of_scope_is_rejected_even_with_high_confidence() -> None:
    """Confidence must not be able to talk the model out of an explicit rejection."""
    classifier = IntentClassifier(0.0, _StubTransformer(OUT_OF_SCOPE_LABEL, 1.0), None)
    prediction = classifier.predict(Preprocessor().process("what is the weather"))
    assert prediction.intent == FALLBACK_INTENT


def test_a_real_intent_still_passes_through_at_confidence() -> None:
    """Positive case: the mapping must not reject legitimate intents."""
    classifier = IntentClassifier(0.55, _StubTransformer("track_order", 0.9), None)
    prediction = classifier.predict(Preprocessor().process("where is my order"))
    assert prediction.intent == "track_order"
    assert prediction.raw_intent == "track_order"


def test_the_confidence_threshold_still_applies_as_a_second_net() -> None:
    """A low-confidence real label is still fallback, independently of out_of_scope."""
    classifier = IntentClassifier(0.55, _StubTransformer("refund", 0.2), None)
    prediction = classifier.predict(Preprocessor().process("hmm"))
    assert prediction.intent == FALLBACK_INTENT
    assert prediction.raw_intent == "refund"


def test_the_rule_classifier_is_unaffected_by_out_of_scope() -> None:
    """Rules have no out_of_scope examples; adding the label must not break them."""
    classifier = IntentClassifier.from_intents_file(settings.intents_file, 0.55, None)
    assert classifier.transformer is None
    pre = Preprocessor()
    assert classifier.predict(pre.process("where is my order")).intent == "track_order"
    # The rule classifier's own miss is still a fallback, not the internal label.
    assert classifier.predict(pre.process("tell me a joke")).intent == FALLBACK_INTENT


def test_no_pipeline_output_can_contain_the_internal_label(client) -> None:
    """End-to-end: the API must never expose out_of_scope, whatever the model says."""
    for message in ("tell me a joke", "???", "what is the weather", "where is my order"):
        response = client.post("/api/v1/chat", json={"message": message})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["intent"] != OUT_OF_SCOPE_LABEL, body
        assert body["intent"] in {
            "cancel_order",
            "contact_support",
            "goodbye",
            "greeting",
            "payment_issue",
            "refund",
            "track_order",
            FALLBACK_INTENT,
        }, body


# --- 3. context safety (model-free) ---------------------------------------------


def _session(pending: str | None = None, last_intent: str | None = None) -> Session:
    return Session(
        session_id="s" * 32,
        memory=ConversationMemory(10),
        pending_intent=pending,
        last_intent=last_intent,
    )


@pytest.mark.parametrize(
    "message",
    ["ORD-12345", "12345", "#98765", "ord 12345 please"],
)
def test_a_slot_answer_resumes_the_pending_intent_even_if_classified_out_of_scope(
    message,
) -> None:
    """4.4: bare slot values now look out-of-scope, and must still fill the slot."""
    session = _session(pending="track_order")
    prediction = IntentPrediction(FALLBACK_INTENT, 0.97, "transformer", OUT_OF_SCOPE_LABEL)
    entities = {"order_id": "ORD-12345"}

    resolved = ContextManager().resolve(session, prediction, entities)

    assert resolved.intent == "track_order"
    assert resolved.is_follow_up is True


def test_pure_gibberish_keeps_the_pending_intent_and_re_asks() -> None:
    """4.4: gibberish during slot filling must not lose the pending intent."""
    session = _session(pending="refund")
    prediction = IntentPrediction(FALLBACK_INTENT, 0.98, "transformer", OUT_OF_SCOPE_LABEL)

    resolved = ContextManager().resolve(session, prediction, {})

    assert resolved.intent == "refund"
    assert resolved.is_follow_up is True
    # The slot is still unfilled, so the handler re-asks and pending_intent survives.
    assert session.pending_intent == "refund"
    assert not resolved.entities


@pytest.mark.parametrize("other", ["cancel_order", "contact_support", "greeting"])
def test_a_clear_different_intent_still_overrides_pending(other) -> None:
    """4.4: a confident, different in-domain intent must win over the pending slot."""
    session = _session(pending="track_order")
    prediction = IntentPrediction(other, 0.95, "transformer", other)

    resolved = ContextManager().resolve(session, prediction, {})

    assert resolved.intent == other
    assert resolved.is_follow_up is False


def test_out_of_scope_with_no_pending_intent_stays_fallback() -> None:
    """No pending question, no bare entity: the rejection must stand."""
    session = _session(pending=None, last_intent=None)
    prediction = IntentPrediction(FALLBACK_INTENT, 0.99, "transformer", OUT_OF_SCOPE_LABEL)

    resolved = ContextManager().resolve(session, prediction, {})

    assert resolved.intent == FALLBACK_INTENT
    assert resolved.is_follow_up is False