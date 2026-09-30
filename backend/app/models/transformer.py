"""Thin inference wrapper around a fine-tuned sequence-classification model."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoadedModel:
    tokenizer: Any
    model: Any
    id2label: dict[int, str]
    device: str


class IntentTransformer:
    def __init__(self, loaded: LoadedModel, max_length: int = 64) -> None:
        """``max_length`` must match the value used during training (config.yaml
        ``model.max_length``); a mismatch truncates differently at inference time."""
        self._loaded = loaded
        self.max_length = max_length

    @property
    def labels(self) -> list[str]:
        return [self._loaded.id2label[i] for i in sorted(self._loaded.id2label)]

    def predict(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k (label, probability) pairs, best first."""
        import torch

        batch = self._loaded.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self._loaded.device)
        with torch.no_grad():
            logits = self._loaded.model(**batch).logits[0]
        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k)
        return [
            (self._loaded.id2label[int(i)], float(v))
            for v, i in zip(values.tolist(), indices.tolist(), strict=True)
        ]
