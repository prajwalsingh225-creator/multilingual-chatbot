"""Compare the rule-based baseline against the fine-tuned Transformer.

    uv run python -m training.evaluate            # rule baseline + transformer (if trained)
    uv run python -m training.evaluate --all      # every example, not just the held-out split
    uv run python -m training.evaluate --rules-only

Reports overall accuracy plus a per-intent breakdown, so regressions on a single
intent are visible instead of hidden in an average.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.models.model_loader import load_intent_model
from app.nlp.intent_classifier import FALLBACK_INTENT, IntentClassifier
from app.nlp.preprocessor import Preprocessor

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def _accuracy(
    pairs: list[tuple[str, str]],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Return overall accuracy and a per-label {correct, total, accuracy} breakdown.

    ``pairs`` is a list of ``(predicted, actual)`` tuples.
    """
    per_label: dict[str, dict[str, float]] = {}
    hits = 0
    for predicted, actual in pairs:
        bucket = per_label.setdefault(actual, {"correct": 0.0, "total": 0.0})
        bucket["total"] += 1
        if predicted == actual:
            bucket["correct"] += 1
            hits += 1
    for bucket in per_label.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    overall = hits / len(pairs) if pairs else 0.0
    return overall, per_label


def _report(title: str, overall: float, per_label: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {title}: overall accuracy {overall:.4f} ({len(per_label)} intents) ===")
    for label in sorted(per_label):
        bucket = per_label[label]
        print(
            f"  {label:<18} {bucket['correct']:>3.0f}/{bucket['total']:<3.0f} "
            f"{bucket['accuracy']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Evaluate on every example.")
    parser.add_argument("--rules-only", action="store_true", help="Skip the Transformer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    data_cfg = config.get("data", {})
    train_file = BASE_DIR / data_cfg.get("train_file", "data/processed/train.json")
    seed = int(data_cfg.get("split_seed", 42))

    records: list[dict[str, Any]] = json.loads(train_file.read_text(encoding="utf-8"))
    from training.train_intent import split_dataset  # local import: avoids a cycle at module load

    if args.all:
        eval_records = records
    else:
        _, eval_records = split_dataset(records, float(data_cfg.get("val_ratio", 0.2)), seed)
    labels = [record["label"] for record in eval_records]
    texts = [record["text"] for record in eval_records]
    print(f"evaluating on {len(eval_records)} examples from {train_file}")

    rules = IntentClassifier.from_intents_file(settings.intents_file, 0.0, None)
    pre = Preprocessor()
    rule_preds = [rules.predict(pre.process(t)).intent for t in texts]
    overall, per_label = _accuracy(list(zip(rule_preds, labels, strict=True)))
    _report("rule baseline", overall, per_label)

    if args.rules_only:
        return

    model_dir = Path(args.model_dir) if args.model_dir else settings.MODEL_DIR
    transformer = load_intent_model(model_dir)
    if transformer is None:
        print(f"\nno trained model at {model_dir}; skipping the Transformer comparison")
        return

    model_cls = IntentClassifier.from_intents_file(settings.intents_file, 0.0, transformer)
    model_preds = [model_cls.predict(pre.process(t)).intent for t in texts]
    overall, per_label = _accuracy(list(zip(model_preds, labels, strict=True)))
    _report(f"transformer ({model_dir.name})", overall, per_label)
    fallback_rate = sum(1 for p in model_preds if p == FALLBACK_INTENT) / len(model_preds)
    print(f"\ntransformer fallback rate: {fallback_rate:.4f}")


if __name__ == "__main__":
    main()