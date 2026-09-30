"""Load the fine-tuned intent model from disk, safely.

Returns ``None`` when no trained model exists so the app can still run with the
rule-based fallback classifier during early development.
"""

from pathlib import Path

from app.core.exceptions import ModelLoadError
from app.core.logging import get_logger
from app.models.transformer import IntentTransformer, LoadedModel

logger = get_logger(__name__)


def load_intent_model(model_dir: Path) -> IntentTransformer | None:
    if not (model_dir / "config.json").exists():
        logger.warning("No trained intent model at %s; using rule-based fallback", model_dir)
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.to(device).eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        logger.info(
            "Loaded intent model from %s on %s (%d labels)",
            model_dir,
            device,
            len(id2label),
        )
        return IntentTransformer(LoadedModel(tokenizer, model, id2label, device))
    except Exception as exc:
        raise ModelLoadError(f"Failed to load intent model from {model_dir}: {exc}") from exc
