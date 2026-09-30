"""Language detection for short chat messages.

Order: script check (Devanagari) -> marker words (handles Hinglish/short Spanish)
-> ``langdetect`` -> default language.
"""

import re
from dataclasses import dataclass

from app.nlp.preprocessor import tokenize

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

_MARKERS: dict[str, set[str]] = {
    "es": {
        "hola", "gracias", "pedido", "dónde", "donde", "quiero", "cancelar", "pago",
        "reembolso", "ayuda", "buenos", "buenas", "adiós", "adios", "necesito", "mi",
        "estado", "hablar", "soporte", "está", "cobraron", "días", "dias",
    },
    "hi": {
        "mera", "meri", "mujhe", "kahan", "kya", "nahi", "chahiye", "karna", "karo",
        "hai", "ho", "gaya", "madad", "namaste", "paisa", "bhai",
    },
    "en": {
        "the", "is", "my", "where", "what", "want", "please", "order", "hello", "help",
        "refund", "payment", "cancel", "track", "thanks", "i", "you",
    },
}
# Ambiguous words only count for half a point and never decide a language alone.
_WEAK = {"mi", "order", "i", "is", "ho", "hai"}


@dataclass(frozen=True)
class LanguageResult:
    language: str
    confidence: float
    method: str  # "script" | "markers" | "langdetect" | "default"


class LanguageDetector:
    def __init__(self, supported: list[str], default: str = "en") -> None:
        self.supported = supported
        self.default = default

    def detect(self, text: str) -> LanguageResult:
        text = text.strip()
        if not text:
            return LanguageResult(self.default, 0.0, "default")

        # 1. Script
        letters = [c for c in text if c.isalpha()]
        if letters:
            ratio = sum(1 for c in letters if _DEVANAGARI.match(c)) / len(letters)
            if ratio > 0.3 and "hi" in self.supported:
                return LanguageResult("hi", min(1.0, ratio), "script")

        # 2. Marker words
        tokens = set(tokenize(text))
        scores: dict[str, float] = {}
        for lang, words in _MARKERS.items():
            if lang not in self.supported:
                continue
            hits = tokens & words
            strong = hits - _WEAK
            if strong or len(hits) >= 2:
                scores[lang] = len(strong) + 0.5 * len(hits & _WEAK)
        if scores:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                return LanguageResult(ranked[0][0], 0.8, "markers")

        # 3. langdetect (seeded for determinism)
        try:
            from langdetect import DetectorFactory, detect_langs
            from langdetect.lang_detect_exception import LangDetectException
        except ImportError:
            return self._default()
        DetectorFactory.seed = 0
        try:
            for cand in detect_langs(text):
                if cand.lang in self.supported and cand.prob >= 0.6:
                    return LanguageResult(cand.lang, float(cand.prob), "langdetect")
        except LangDetectException:
            pass
        return self._default()

    def _default(self) -> LanguageResult:
        return LanguageResult(self.default, 0.0, "default")
