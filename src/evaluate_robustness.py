import argparse
import csv
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import joblib
import matplotlib
import numpy as np

from evasion_transforms import TRANSFORMATIONS
from preprocessing import preprocess_email
from risk_engine import assess_risk
from security_analyzer import analyze_email


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST = REPOSITORY_ROOT / "data" / "processed" / "test.csv"
DEFAULT_MODEL = REPOSITORY_ROOT / "models" / "baseline_lr.joblib"
DEFAULT_REPORT = REPOSITORY_ROOT / "reports" / "adversarial_robustness.json"
DEFAULT_FIGURE = REPOSITORY_ROOT / "reports" / "figures" / "adversarial_robustness.png"
RANDOM_SEED = 42


def load_phishing_records(path: Path) -> List[Dict[str, str]]:
    records = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as test_file:
        reader = csv.DictReader(test_file)
        required = {"subject", "body", "text", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Test CSV must contain subject, body, text, and label columns.")
        for row_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            if label not in {"0", "1"}:
                raise ValueError(f"Test row {row_number}: label must be 0 or 1.")
            if label == "1":
                records.append(
                    {"subject": row["subject"], "body": row["body"], "text": row["text"]}
                )
    if not records:
        raise ValueError("Locked test split contains no phishing records.")
    return records


def _probabilities(bundle: Mapping, texts: List[str]) -> np.ndarray:
    pipeline = bundle.get("pipeline")
    if pipeline is not None:
        return pipeline.predict_proba(texts)[:, 1]
    vectorizer = bundle["vectorizer"]
    return bundle["model"].predict_proba(vectorizer.transform(texts))[:, 1]


def _variant_metrics(
    bundle: Mapping,
    records: List[Dict[str, str]],
    texts: List[str],
    subjects: List[str],
    bodies: List[str],
    original_high_critical: Optional[List[bool]] = None,
) -> Dict[str, object]:
    probabilities = _probabilities(bundle, texts)
    threshold = float(bundle["threshold"])
    predicted_phishing = probabilities >= threshold
    high_critical = []
    for probability, subject, body in zip(probabilities, subjects, bodies):
        risk = assess_risk(float(probability), analyze_email(subject, body))
        high_critical.append(risk["risk_level"] in {"High", "Critical"})

    evaluated = len(records)
    false_negatives = int(evaluated - np.count_nonzero(predicted_phishing))
    false_negative_high_critical = sum(
        not model_prediction and risk_high
        for model_prediction, risk_high in zip(predicted_phishing, high_critical)
    )
    result = {
        "evaluated_messages": evaluated,
        "phishing_recall": float(np.mean(predicted_phishing)),
        "false_negative_count": false_negatives,
        "ml_false_negatives_remaining_high_critical_count": false_negative_high_critical,
        "ml_false_negatives_remaining_high_critical_percent": (
            round(false_negative_high_critical / false_negatives * 100, 4)
            if false_negatives
            else 0.0
        ),
        "average_phishing_probability": float(np.mean(probabilities)),
        "median_phishing_probability": float(statistics.median(map(float, probabilities))),
        "high_critical_percent": round(sum(high_critical) / evaluated * 100, 4),
        "medium_low_percent": round((evaluated - sum(high_critical)) / evaluated * 100, 4),
    }
    if original_high_critical is not None:
        dropped = sum(
            was_high and not remains_high
            for was_high, remains_high in zip(original_high_critical, high_critical)
        )
        original_high_count = sum(original_high_critical)
        result["dropped_high_critical_to_medium_low_count"] = dropped
        result["dropped_high_critical_to_medium_low_percent_of_all"] = round(
            dropped / evaluated * 100, 4
        )
        result["dropped_high_critical_to_medium_low_percent_of_original_high_critical"] = (
            round(dropped / original_high_count * 100, 4) if original_high_count else 0.0
        )
    return result, high_critical


def evaluate_robustness(
    bundle: Mapping,
    records: List[Dict[str, str]],
    transformations=TRANSFORMATIONS,
) -> Dict[str, object]:
    original_subjects = [record["subject"] for record in records]
    original_bodies = [record["body"] for record in records]
    original_texts = [record["text"] for record in records]
    original, original_high_critical = _variant_metrics(
        bundle,
        records,
        original_texts,
        original_subjects,
        original_bodies,
    )
    original["recall_change_vs_original"] = 0.0

    variants = {}
    for name, transform in transformations.items():
        transformed = [transform(record["subject"], record["body"]) for record in records]
        subjects = [parts[0] for parts in transformed]
        bodies = [parts[1] for parts in transformed]
        texts = [preprocess_email(subject, body)[2] for subject, body in transformed]
        metrics, _ = _variant_metrics(
            bundle,
            records,
            texts,
            subjects,
            bodies,
            original_high_critical,
        )
        metrics["recall_change_vs_original"] = (
            metrics["phishing_recall"] - original["phishing_recall"]
        )
        metrics["messages_changed"] = sum(
            subject != record["subject"] or body != record["body"]
            for record, subject, body in zip(records, subjects, bodies)
        )
        variants[name] = metrics

    return {
        "purpose": (
            "Defensive robustness testing of an existing detector; transformations are "
            "controlled evaluation perturbations, not malicious-content generation."
        ),
        "random_seed": RANDOM_SEED,
        "test_source": "trec7",
        "model_threshold": float(bundle["threshold"]),
        "original": original,
        "transformations": variants,
    }


def write_report(report: Dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            json.dump(report, output, indent=2)
            output.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def save_figure(report: Dict[str, object], path: Path) -> None:
    labels = ["original"] + list(report["transformations"])
    results = [report["original"]] + list(report["transformations"].values())
    recall = [result["phishing_recall"] * 100 for result in results]
    high_critical = [result["high_critical_percent"] for result in results]
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(positions - width / 2, recall, width, label="ML phishing recall")
    axis.bar(positions + width / 2, high_critical, width, label="High/Critical risk")
    axis.set_ylabel("Percentage of phishing messages")
    axis.set_ylim(0, 100)
    axis.set_xticks(positions, [label.replace("_", "\n") for label in labels])
    axis.set_title("Locked-test phishing robustness by transformation")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phishing-evasion robustness.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    bundle = joblib.load(args.model)
    records = load_phishing_records(args.test)
    report = evaluate_robustness(bundle, records)
    write_report(report, args.report)
    save_figure(report, args.figure)
    print(f"Evaluated locked-test phishing messages: {len(records)}")
    print(f"Original recall: {report['original']['phishing_recall']:.4f}")
    for name, metrics in report["transformations"].items():
        print(
            f"{name}: recall={metrics['phishing_recall']:.4f} "
            f"change={metrics['recall_change_vs_original']:+.4f} "
            f"FN={metrics['false_negative_count']}"
        )
    print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
