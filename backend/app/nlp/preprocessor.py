"""Text normalisation shared by inference and training (keep them identical)."""

import re
import string
import unicodedata
from dataclasses import dataclass, field

_WS = re.compile(r"\s+")
# C0/C1 control characters except tab, newline and carriage return, which become spaces.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# The curly quotes are part of the punctuation we strip, hence the RUF001 waiver.
_STRIP = string.punctuation + "¿¡।॥“”‘’…"  # noqa: RUF001


def strip_control_chars(text: str) -> str:
    """Remove control/formatting characters that could poison logs or the tokenizer."""
    return _CONTROL.sub("", text)


def normalize_text(text: str) -> str:
    """Strip control characters, NFKC-normalise, casefold and collapse whitespace."""
    text = unicodedata.normalize("NFKC", strip_control_chars(text))
    text = _WS.sub(" ", text).strip()
    return text.casefold()


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer that also works for Devanagari (``\\w`` splits matras)."""
    tokens = (tok.strip(_STRIP) for tok in normalize_text(text).split(" "))
    return [tok for tok in tokens if tok]


@dataclass(frozen=True)
class ProcessedText:
    original: str
    normalized: str
    tokens: list[str] = field(default_factory=list)


class Preprocessor:
    def process(self, text: str) -> ProcessedText:
        return ProcessedText(
            original=text, normalized=normalize_text(text), tokens=tokenize(text)
        )
