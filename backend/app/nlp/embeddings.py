"""Sentence embeddings (mean-pooled Transformer outputs).

Optional utility for semantic similarity / retrieval later (e.g. FAQ matching or
few-shot intent fallback). Not used by the default pipeline.
"""

from typing import Any


class Embedder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device).eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        import torch

        self._load()
        batch = self._tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=1)
        return pooled.cpu().tolist()

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b, strict=True)))  # vectors are normalised
