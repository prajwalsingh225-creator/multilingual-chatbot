"""Intent classification.

Uses the fine-tuned Transformer when available; otherwise falls back to a token-overlap
matcher over data/raw/intents.json so the whole pipeline works before training.
"""

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.models.transformer import IntentTransformer
from app.nlp.preprocessor import ProcessedText, tokenize

FALLBACK_INTENT = "fallback"


@dataclass(frozen=True)
class IntentPrediction:
    intent: str  # FALLBACK_INTENT when below the confidence threshold
    confidence: float
    source: str  # "transformer" | "rules"
    raw_intent: str  # best guess before thresholding


def _fold(token: str) -> str:
    """Strip Latin accents (dónde -> donde). Devanagari marks are left untouched."""
    decomposed = unicodedata.normalize("NFKD", token)
    kept = "".join(c for c in decomposed if not "\u0300" <= c <= "\u036f")
    return unicodedata.normalize("NFC", kept)


def _rule_tokens(text: str) -> set[str]:
    return {_fold(t) for t in tokenize(text) if not any(c.isdigit() for c in t)}


class IntentClassifier:
    def __init__(
        self,
        threshold: float,
        transformer: IntentTransformer | None = None,
        examples: dict[str, list[str]] | None = None,
    ) -> None:
        self.threshold = threshold
        self.transformer = transformer
        self._examples = {
            intent: [_rule_tokens(e) for e in texts if _rule_tokens(e)]
            for intent, texts in (examples or {}).items()
        }

    @classmethod
    def from_intents_file(
        cls, path: Path, threshold: float, transformer: IntentTransformer | None = None
    ) -> "IntentClassifier":
        data = json.loads(path.read_text(encoding="utf-8"))
        examples = {
            item["name"]: [
                ex for lang_examples in item["examples"].values() for ex in lang_examples
            ]
            for item in data["intents"]
        }
        return cls(threshold, transformer, examples)

    def predict(self, processed: ProcessedText) -> IntentPrediction:
        if self.transformer is not None:
            label, conf = self.transformer.predict(processed.normalized, top_k=1)[0]
            source = "transformer"
        else:
            label, conf = self._rule_predict(processed.normalized)
            source = "rules"
        intent = label if conf >= self.threshold else FALLBACK_INTENT
        return IntentPrediction(intent, conf, source, label)

    def _rule_predict(self, text: str) -> tuple[str, float]:
        msg = _rule_tokens(text)
        if not msg:
            return FALLBACK_INTENT, 0.0
        best_intent, best = FALLBACK_INTENT, 0.0
        for intent, examples in self._examples.items():
            for ex in examples:
                overlap = len(ex & msg)
                if not overlap:
                    continue
                score = 0.6 * overlap / len(ex) + 0.4 * overlap / len(msg)
                if score > best:
                    best_intent, best = intent, score
        return best_intent, round(best, 4)
