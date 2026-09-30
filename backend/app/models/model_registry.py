"""Holds the loaded model(s) for the lifetime of the app."""

from pathlib import Path

from app.models.model_loader import load_intent_model
from app.models.transformer import IntentTransformer


class ModelRegistry:
    def __init__(self) -> None:
        self.intent_model: IntentTransformer | None = None

    def load(self, model_dir: Path) -> None:
        self.intent_model = load_intent_model(model_dir)

    @property
    def intent_model_loaded(self) -> bool:
        return self.intent_model is not None
