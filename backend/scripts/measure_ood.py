"""Measure out-of-domain (OOD) rejection and in-domain false rejection on the FROZEN set.

    uv run python -m scripts.measure_ood --classifier transformer
    uv run python -m scripts.measure_ood --classifier rules
    uv run python -m scripts.measure_ood --model-dir trained_models/staging/run-02 --sweep

To score an UNSEEN file of real user sentences (same shape: "ood" and "in_domain" arrays
of {"text", "language", "kind"}) without touching the frozen set:

    uv run python -m scripts.measure_ood --extra-set /path/to/my_user_rows.json --run-id mine

``--extra-set`` is read-only. It is never used for training or for choosing a model or a
threshold; it exists so real traffic can be checked after the decision is already made.

The eval set in ``data/eval/ood_eval.json`` is frozen: it is never used for training,
threshold tuning or model selection. This script only reads it.

Reports, for each configuration:

  * OOD rejection rate  -- fraction of off-topic rows answered with intent "fallback";
  * in-domain FALSE rejection -- legitimate requests wrongly answered "fallback";
  * mean confidence of the rows that were wrongly accepted;
  * the worst offenders (highest confidence wrong accepts), so failures are inspectable.

With ``--sweep`` the same model is scored across a range of confidence thresholds. That
answers whether raising INTENT_CONFIDENCE_THRESHOLD alone could separate OOD from
in-domain, which is the question that decides if the threshold is a viable lever.

Machine-readable output is written to ``reports/ood_<run-id>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVAL_FILE = BACKEND_DIR / "data" / "eval" / "ood_eval.json"
FALLBACK_INTENT = "fallback"

# Sensible defaults for the threshold sweep.
SWEEP_START = 0.55
SWEEP_STOP = 0.95
SWEEP_STEP = 0.05


def _bootstrap_env(model_dir: Path) -> None:
    """Point settings at a throwaway DB before the app package is imported."""
    scratch = tempfile.mkdtemp(prefix="measure-ood-")
    os.environ.setdefault("APP_ENV", "development")
    os.environ["DATABASE_URL"] = f"sqlite:///{scratch}/measure_ood.db"
    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ.setdefault("DEFAULT_LANGUAGE", "en")
    os.environ.setdefault("SUPPORTED_LANGUAGES", "en,hi,es")


def load_eval() -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    return {"ood": payload["ood"], "in_domain": payload["in_domain"]}


def sha256_of(path: Path) -> str:
    """Hash of the eval bytes, so a report can never be read against a different file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_extra_set(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load a caller-supplied unseen file, in the frozen set's own shape.

    This is a measurement-only escape hatch for real user traffic. Nothing here is ever
    fed to training or used to choose a model; the script only reads the file.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "ood" not in payload or "in_domain" not in payload:
        raise SystemExit(
            f"{path} must be a JSON object with 'ood' and 'in_domain' arrays, each row "
            '{"text": str, "language": str, "kind": str}'
        )
    for split in ("ood", "in_domain"):
        for row in payload[split]:
            missing = {"text", "language", "kind"} - set(row)
            if missing:
                raise SystemExit(f"{path}: {split} row {row!r} is missing {sorted(missing)}")
            if not row["text"].strip():
                raise SystemExit(f"{path}: {split} row has empty text: {row!r}")
    print(f"loaded extra set: {len(payload['ood'])} ood, {len(payload['in_domain'])} in_domain")
    return {"ood": payload["ood"], "in_domain": payload["in_domain"]}


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def evaluate(
    classifier: Any, pre: Any, rows: Sequence[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """Score ``rows``; a row is 'rejected' when the public intent is ``fallback``."""
    by_language: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "rejected": 0})
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "rejected": 0})
    wrong_accepts: list[dict[str, Any]] = []
    total = rejected = 0

    classifier.threshold = threshold
    for row in rows:
        prediction = classifier.predict(pre.process(row["text"]))
        is_rejected = prediction.intent == FALLBACK_INTENT
        total += 1
        rejected += int(is_rejected)
        for bucket, key in ((by_language, row["language"]), (by_kind, row["kind"])):
            bucket[key]["total"] += 1
            bucket[key]["rejected"] += int(is_rejected)
        if not is_rejected:
            wrong_accepts.append(
                {
                    "text": row["text"],
                    "language": row["language"],
                    "kind": row["kind"],
                    "intent": prediction.intent,
                    "raw_intent": prediction.raw_intent,
                    "confidence": prediction.confidence,
                }
            )

    wrong_accepts.sort(key=lambda item: -item["confidence"])
    confidences = [item["confidence"] for item in wrong_accepts]
    return {
        "total": total,
        "rejected": rejected,
        "rate": _rate(rejected, total),
        "per_language": {
            lang: {
                "total": counts["total"],
                "rejected": counts["rejected"],
                "rate": _rate(counts["rejected"], counts["total"]),
            }
            for lang, counts in sorted(by_language.items())
        },
        "per_kind": {
            kind: {
                "total": counts["total"],
                "rejected": counts["rejected"],
                "rate": _rate(counts["rejected"], counts["total"]),
            }
            for kind, counts in sorted(by_kind.items())
        },
        "mean_confidence_of_wrong_accepts": (
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        ),
        "worst_offenders": wrong_accepts[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None, help="Model to score.")
    parser.add_argument(
        "--classifier",
        choices=["transformer", "rules"],
        default="transformer",
        help="'transformer' needs a model dir; 'rules' needs none.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--sweep", action="store_true", help="Sweep the threshold.")
    parser.add_argument(
        "--extra-set",
        type=Path,
        default=None,
        help=(
            "Score an additional UNSEEN user file instead of (or alongside) the frozen set. "
            "Read-only and never used for training or model selection."
        ),
    )
    parser.add_argument("--run-id", type=str, default="baseline", help="Name for the report file.")
    parser.add_argument("--output-dir", type=Path, default=BACKEND_DIR / "reports")
    args = parser.parse_args()

    model_dir = args.model_dir or BACKEND_DIR / "trained_models" / "intent_model"
    _bootstrap_env(model_dir)

    from app.core.config import settings
    from app.models.model_loader import load_intent_model
    from app.nlp.intent_classifier import IntentClassifier
    from app.nlp.preprocessor import Preprocessor

    transformer = None
    if args.classifier == "transformer":
        transformer = load_intent_model(model_dir)
        if transformer is None:
            raise SystemExit(f"no model at {model_dir}; use --classifier rules")
        print(f"model labels: {transformer.labels}")

    classifier = IntentClassifier.from_intents_file(
        settings.intents_file, settings.INTENT_CONFIDENCE_THRESHOLD, transformer
    )
    pre = Preprocessor()
    threshold = args.threshold if args.threshold is not None else (
        settings.INTENT_CONFIDENCE_THRESHOLD
    )
    data = load_eval()

    extra: dict[str, list[dict[str, Any]]] | None = None
    if args.extra_set is not None:
        extra = load_extra_set(args.extra_set)
        print(
            f"\n*** EXTRA SET {args.extra_set} -- read-only, never trained on, "
            "never used for model selection ***"
        )

    ood = evaluate(classifier, pre, data["ood"], threshold)
    control = evaluate(classifier, pre, data["in_domain"], threshold)
    extra_ood = evaluate(classifier, pre, extra["ood"], threshold) if extra else None
    extra_control = evaluate(classifier, pre, extra["in_domain"], threshold) if extra else None

    print(f"\n=== OOD rejection @ threshold {threshold} ({args.classifier}) ===")
    print(f"overall {ood['rejected']}/{ood['total']} = {ood['rate']:.4f}")
    print("per language:")
    for lang, counts in ood["per_language"].items():
        print(f"  {lang:<9} {counts['rejected']}/{counts['total']} = {counts['rate']:.4f}")
    print("per kind:")
    for kind, counts in ood["per_kind"].items():
        print(f"  {kind:<15} {counts['rejected']}/{counts['total']} = {counts['rate']:.4f}")
    print(f"\nin-domain FALSE rejection: {control['rejected']}/{control['total']}"
          f" = {control['rate']:.4f}")
    print(f"mean confidence of wrong accepts: {ood['mean_confidence_of_wrong_accepts']}")
    print("\nworst offenders:")
    for item in ood["worst_offenders"]:
        print(f"  {item['confidence']:.4f} {item['intent']:<16} {item['text'][:44]}")

    if extra_ood is not None and extra_control is not None:
        print("\n=== EXTRA SET (unseen; measurement only) ===")
        print(
            f"ood rejection {extra_ood['rejected']}/{extra_ood['total']}"
            f" = {extra_ood['rate']:.4f}"
        )
        for lang, counts in extra_ood["per_language"].items():
            print(f"  {lang:<9} {counts['rejected']}/{counts['total']} = {counts['rate']:.4f}")
        print(
            f"false rejection {extra_control['rejected']}/{extra_control['total']}"
            f" = {extra_control['rate']:.4f}"
        )
        for lang, counts in extra_control["per_language"].items():
            print(f"  {lang:<9} {counts['rejected']}/{counts['total']} = {counts['rate']:.4f}")

    report: dict[str, Any] = {
        "run_id": args.run_id,
        "classifier": args.classifier,
        "model_dir": str(model_dir),
        "threshold": threshold,
        "eval_file": str(EVAL_FILE.relative_to(BACKEND_DIR)),
        "eval_file_sha256": sha256_of(EVAL_FILE),
        "ood": ood,
        "in_domain_control": control,
        "sweep": [],
    }
    if extra_ood is not None and extra_control is not None:
        report["extra_set"] = {
            "path": str(args.extra_set),
            "note": "unseen input, measurement only, never used for training or selection",
            "ood": extra_ood,
            "in_domain_control": extra_control,
        }

    if args.sweep:
        print("\n=== threshold sweep ===")
        steps = []
        value = SWEEP_START
        while value <= SWEEP_STOP + 1e-9:
            swept_ood = evaluate(classifier, pre, data["ood"], round(value, 2))
            swept_control = evaluate(classifier, pre, data["in_domain"], round(value, 2))
            steps.append(
                {
                    "threshold": round(value, 2),
                    "ood_rate": swept_ood["rate"],
                    "false_rejection": swept_control["rate"],
                }
            )
            print(
                f"  threshold {value:.2f}  ood={swept_ood['rate']:.4f}"
                f"  false_rejection={swept_control['rate']:.4f}"
            )
            value += SWEEP_STEP
        report["sweep"] = steps

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"ood_{args.run_id}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(BACKEND_DIR)}")


if __name__ == "__main__":
    main()