"""Dataset preparation and fine-tuning for the multilingual intent classifier.

Two modes:

    uv run python -m training.train_intent --prepare-only   # build data/processed/train.json
    uv run --group training python -m training.train_intent  # fine-tune and save to MODEL_DIR

Heavy dependencies (torch / transformers) are imported inside :func:`main` so that
``--prepare-only`` stays fast and works on machines without a GPU stack.

The training text is normalised with the *same* :func:`app.nlp.preprocessor.normalize_text`
used at inference time, so the two never drift apart.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.nlp.preprocessor import normalize_text

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def build_dataset(intents_file: Path, out_file: Path) -> list[dict[str, Any]]:
    """Flatten ``intents.json`` into ``[{"text", "label", "language"}]`` records.

    Writes the records to ``out_file`` as JSON and returns them.
    """
    payload = json.loads(Path(intents_file).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for intent in payload["intents"]:
        name = intent["name"]
        for language, examples in intent["examples"].items():
            for example in examples:
                text = normalize_text(example)
                if not text:
                    continue
                records.append({"text": text, "label": name, "language": language})

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


def split_dataset(
    records: list[dict[str, Any]], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split per label so every intent appears in both partitions."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for group in by_label.values():
        rng.shuffle(group)
        cut = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        if len(group) - cut < 1:
            cut = 0
        val.extend(group[:cut])
        train.extend(group[cut:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build data/processed/train.json; do not train.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument(
        "--base-model", type=str, default=None, help="Override config base_model."
    )
    parser.add_argument("--model-dir", type=Path, default=None, help="Override MODEL_DIR.")
    args = parser.parse_args()

    config = _load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    intents_file = BASE_DIR / data_cfg.get("intents_file", "data/raw/intents.json")
    train_file = BASE_DIR / data_cfg.get("train_file", "data/processed/train.json")
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    seed = int(data_cfg.get("split_seed", 42))
    base_model = args.base_model or model_cfg.get("base_model", settings.MODEL_NAME)
    max_length = int(model_cfg.get("max_length", 64))
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 3))
    model_dir = Path(args.model_dir) if args.model_dir else settings.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    records = build_dataset(intents_file, train_file)
    train_records, val_records = split_dataset(records, val_ratio, seed)
    print(f"prepared {len(records)} records -> {train_file}")
    print(f"train={len(train_records)} val={len(val_records)} base_model={base_model}")

    if args.prepare_only:
        return

    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    labels = sorted({record["label"] for record in records})
    label2id = {label: index for index, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(labels), label2id=label2id, id2label=dict(enumerate(labels))
    )

    class EncodedDataset(Dataset):
        """Pre-tokenised records; the default collator batches and pads the labels."""

        def __init__(self, rows: dict[str, list[torch.Tensor]]) -> None:
            self._rows = rows
            self._keys = list(rows)

        def __len__(self) -> int:
            return len(self._rows["labels"])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {key: self._rows[key][index] for key in self._keys}

    def encode(items: list[dict[str, Any]]) -> EncodedDataset:
        batch = tokenizer(
            [item["text"] for item in items],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        rows = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
        rows["labels"] = torch.tensor(
            [label2id[item["label"]] for item in items], dtype=torch.long
        )
        return EncodedDataset(rows)

    batch_size = int(train_cfg.get("batch_size", 16))
    steps_per_epoch = max(1, len(train_records) // batch_size)
    total_steps = steps_per_epoch * epochs
    args_obj = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        # transformers v5 removed warmup_ratio; derive the equivalent step count.
        warmup_steps=int(total_steps * float(train_cfg.get("warmup_ratio", 0.1))),
        optim="adamw_torch",
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=25,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args_obj,
        train_dataset=encode(train_records),
        eval_dataset=encode(val_records),
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"saved model + tokenizer to {model_dir} (trained on {device})")


if __name__ == "__main__":
    main()