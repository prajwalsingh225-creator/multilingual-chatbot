"""Evaluate the intent classifier honestly.

    uv run python -m training.evaluate                       # rules only (no torch needed)
    uv run python -m training.evaluate --model-dir trained_models/staging/run-01
    uv run python -m training.evaluate --model-dir trained_models/intent_model --with-transformer

Everything is scored on the SAME held-out validation split:

  * the rule classifier is built from the TRAIN split only, never from the full corpus;
  * the transformer is scored on that same validation split;
  * train-split accuracy is printed too, which is what separates "under-trained"
    from "over-fitting".

Reports overall accuracy, macro-F1, per-intent accuracy, per-language accuracy and a
confusion matrix, plus a train/val leakage report (exact and near duplicates).

Heavy dependencies (torch / transformers) are imported lazily so the rule report runs
on a machine without a GPU stack.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.nlp.intent_classifier import IntentClassifier
from app.nlp.preprocessor import Preprocessor
from training.train_intent import BASE_DIR, build_dataset, split_dataset

NEAR_DUPLICATE_RATIO = 0.9

# Rule fallback used when scoring: a rule miss must still be counted as a miss.
RULE_SCORING_THRESHOLD = 0.0


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


# --- leakage -----------------------------------------------------------------


def exact_duplicate_report(train: Sequence[dict], val: Sequence[dict]) -> dict[str, Any]:
    """Exact duplicates across the split boundary, after ``normalize_text``."""
    train_texts = Counter(row["text"] for row in train)
    val_texts = Counter(row["text"] for row in val)
    shared = set(train_texts) & set(val_texts)

    # Duplicates *inside* each split inflate training counts too.
    train_internal = sum(count - 1 for count in train_texts.values() if count > 1)
    val_internal = sum(count - 1 for count in val_texts.values() if count > 1)
    return {
        "across_split": len(shared),
        "examples": sorted(shared)[:5],
        "train_internal": train_internal,
        "val_internal": val_internal,
    }


def near_duplicate_report(
    train: Sequence[dict], val: Sequence[dict], ratio: float = NEAR_DUPLICATE_RATIO
) -> dict[str, Any]:
    """Near-duplicate val rows, using character-level similarity against train.

    The corpus is templated ("... ORD-12345" / "... ORD-67890"), so a high character
    similarity usually means the val sentence is the same template as a train sentence.
    """
    train_texts = sorted({row["text"] for row in train})
    pairs: list[tuple[float, str, str]] = []
    for val_row in val:
        best = 0.0
        best_text = ""
        for train_text in train_texts:
            score = SequenceMatcher(None, val_row["text"], train_text).ratio()
            if score > best:
                best, best_text = score, train_text
        if best >= ratio:
            pairs.append((round(best, 4), val_row["text"], best_text))
    pairs.sort(reverse=True)
    return {
        "ratio_threshold": ratio,
        "count": len(pairs),
        "of_val": len(val),
        "examples": [
            {"similarity": score, "val": val_text, "train": train_text}
            for score, val_text, train_text in pairs[:5]
        ],
    }


# --- metrics -----------------------------------------------------------------


def confusion_matrix(
    pairs: Iterable[tuple[str, str]], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """``matrix[gold][predicted] += 1``."""
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for gold, pred in pairs:
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def per_group_accuracy(
    pairs: Sequence[tuple[str, str, str]], index: int
) -> dict[str, dict[str, int]]:
    """Accuracy grouped by ``row[index]`` of ``(gold, pred, group_key)`` triples."""
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for triple in pairs:
        key = triple[index]
        totals[key] = totals.get(key, 0) + 1
        if triple[0] == triple[1]:
            correct[key] = correct.get(key, 0) + 1
    return {
        key: {
            "correct": correct.get(key, 0),
            "total": totals[key],
            "accuracy": correct.get(key, 0) / totals[key],
        }
        for key in sorted(totals)
    }


def macro_f1(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> dict[str, Any]:
    """Macro-averaged F1 over the gold labels present in the gold set."""
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    present = [label for label in labels if per_label[label]["support"] > 0]
    macro = sum(per_label[label]["f1"] for label in present) / len(present) if present else 0.0
    return {"macro_f1": round(macro, 4), "per_label": per_label}


def render_matrix(matrix: dict[str, dict[str, int]], labels: Sequence[str]) -> str:
    width = max(len(label) for label in labels)
    header = " " * (width + 2) + " ".join(f"{label[:9]:>9}" for label in labels)
    lines = [header]
    for gold in labels:
        cells = " ".join(f"{matrix[gold][pred]:>9}" for pred in labels)
        lines.append(f"{gold:<{width}} |{cells}")
    return "\n".join(lines)


def evaluate_predictions(
    name: str, rows: Sequence[dict[str, Any]], predict: Any, labels: Sequence[str]
) -> dict[str, Any]:
    """Score ``predict(text) -> intent`` over ``rows``."""
    pre = Preprocessor()
    pairs: list[tuple[str, str]] = []
    triples: list[tuple[str, str, str]] = []
    for row in rows:
        processed = pre.process(row["text"])
        prediction = predict(processed)
        gold = row["label"]
        pairs.append((gold, prediction))
        triples.append((gold, prediction, row["language"]))

    correct = sum(1 for gold, pred in pairs if gold == pred)
    report = {
        "name": name,
        "n": len(pairs),
        "accuracy": round(correct / len(pairs), 4) if pairs else 0.0,
        "correct": correct,
    }
    report.update(macro_f1(pairs, labels))
    # triples are (gold, predicted, language): index 0 groups per intent, index 2 per language.
    report["per_intent"] = per_group_accuracy(triples, 0)
    report["per_language"] = per_group_accuracy(triples, 2)
    report["confusion_matrix"] = confusion_matrix(pairs, labels)
    report["_pairs"] = pairs
    return report


def print_report(report: dict[str, Any], labels: Sequence[str], show_matrix: bool = True) -> None:
    print(f"\n=== {report['name']} ===")
    print(
        f"n={report['n']}  accuracy={report['accuracy']:.4f}"
        f"  macro_f1={report['macro_f1']:.4f}"
    )
    print("per-intent accuracy:")
    for label, stats in report["per_intent"].items():
        print(
            f"  {label:<18} {stats['correct']:>4}/{stats['total']:<4}"
            f" {stats['accuracy']:.4f}"
        )
    print("per-language accuracy:")
    for language, stats in report["per_language"].items():
        print(
            f"  {language:<18} {stats['correct']:>4}/{stats['total']:<4}"
            f" {stats['accuracy']:.4f}"
        )
    weakest = min(report["per_intent"].items(), key=lambda kv: kv[1]["accuracy"])
    print(f"weakest intent: {weakest[0]} {weakest[1]['accuracy']:.4f}")
    if show_matrix:
        print("confusion matrix (rows=gold, cols=pred):")
        print(render_matrix(report["confusion_matrix"], labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Evaluate this model directory instead of MODEL_DIR (staging models).",
    )
    parser.add_argument(
        "--with-transformer",
        action="store_true",
        help="Also score the fine-tuned transformer (requires the training extra).",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the report as JSON.")
    args = parser.parse_args()

    config = _load_config(args.config)
    data_cfg = config.get("data", {})
    intents_file = BASE_DIR / data_cfg.get("intents_file", "data/raw/intents.json")
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    seed = int(data_cfg.get("split_seed", 42))

    train_file = data_cfg.get("train_file", "data/processed/train.json")
    records = build_dataset(intents_file, BASE_DIR / train_file)
    train_records, val_records = split_dataset(records, val_ratio, seed)
    labels = sorted({row["label"] for row in records})
    print(
        f"corpus={len(records)}  train={len(train_records)}"
        f" val={len(val_records)} intents={len(labels)}"
    )

    print("\n--- leakage report ---")
    exact = exact_duplicate_report(train_records, val_records)
    print(
        f"exact duplicates: across_split={exact['across_split']} "
        f"train_internal={exact['train_internal']} val_internal={exact['val_internal']}"
    )
    for example in exact["examples"]:
        print(f"  dup: {example}")
    near = near_duplicate_report(train_records, val_records)
    print(
        f"near duplicates (similarity>={near['ratio_threshold']}):"
        f" {near['count']}/{near['of_val']} val rows"
    )
    for example in near["examples"]:
        print(f"  {example['similarity']}: val={example['val']!r} ~ train={example['train']!r}")

    reports: list[dict[str, Any]] = []

    # Rule classifier built from TRAIN examples only.
    train_examples: dict[str, list[str]] = {}
    for row in train_records:
        train_examples.setdefault(row["label"], []).append(row["text"])
    rule = IntentClassifier(RULE_SCORING_THRESHOLD, None, train_examples)
    rule_pre = Preprocessor()
    reports.append(
        evaluate_predictions(
            "rules (built from TRAIN split) on VAL",
            val_records,
            lambda p: rule.predict(p).intent,
            labels,
        )
    )
    reports.append(
        evaluate_predictions(
            "rules (built from TRAIN split) on TRAIN",
            train_records,
            lambda p: rule.predict(p).intent,
            labels,
        )
    )

    if args.with_transformer:
        from app.core.exceptions import ModelLoadError
        from app.models.model_loader import load_intent_model

        model_dir = args.model_dir or settings.MODEL_DIR
        transformer = load_intent_model(model_dir)
        if transformer is None:
            raise ModelLoadError(
                f"no config.json in {model_dir}; train a model or pass --model-dir"
            )
        reports.append(
            evaluate_predictions(
                f"transformer ({model_dir}) on VAL",
                val_records,
                lambda p: transformer.predict(p.normalized, top_k=1)[0][0],
                labels,
            )
        )
        reports.append(
            evaluate_predictions(
                f"transformer ({model_dir}) on TRAIN",
                train_records,
                lambda p: transformer.predict(p.normalized, top_k=1)[0][0],
                labels,
            )
        )

    for report in reports:
        print_report(report, labels)

    print("\n--- summary ---")
    for report in reports:
        weakest = min(report["per_intent"].items(), key=lambda kv: kv[1]["accuracy"])
        print(
            f"{report['name']:<52} acc={report['accuracy']:.4f}"
            f" macro_f1={report['macro_f1']:.4f}"
            f" weakest={weakest[0]}:{weakest[1]['accuracy']:.4f}"
        )

    if args.json:
        payload = {
            "train_n": len(train_records),
            "val_n": len(val_records),
            "labels": labels,
            "leakage": {"exact": exact, "near": near},
            "reports": [
                {key: value for key, value in report.items() if key != "_pairs"}
                for report in reports
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")

    del rule_pre


if __name__ == "__main__":
    main()