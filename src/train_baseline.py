import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPOSITORY_ROOT / "data" / "processed" / "train.csv"
DEFAULT_DEV = REPOSITORY_ROOT / "data" / "processed" / "dev.csv"
DEFAULT_TEST = REPOSITORY_ROOT / "data" / "processed" / "test.csv"
DEFAULT_MODEL = REPOSITORY_ROOT / "models" / "baseline_lr.joblib"
DEFAULT_METRICS = REPOSITORY_ROOT / "reports" / "baseline_metrics.json"
DEFAULT_FIGURES = REPOSITORY_ROOT / "reports" / "figures"
RANDOM_SEED = 42
MIN_PHISHING_PRECISION = 0.90
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.98,
    "max_features": 200_000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}
LOGISTIC_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1_000,
    "solver": "liblinear",
    "random_state": RANDOM_SEED,
}


def load_split(path: Path) -> Tuple[List[str], np.ndarray]:
    texts = []
    labels = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as split_file:
        reader = csv.DictReader(split_file)
        if reader.fieldnames is None or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain text and label columns.")
        for row_number, row in enumerate(reader, start=2):
            text = row.get("text") or ""
            label = (row.get("label") or "").strip()
            if not text.strip():
                raise ValueError(f"{path}, row {row_number}: text is blank.")
            if label not in {"0", "1"}:
                raise ValueError(f"{path}, row {row_number}: label must be 0 or 1.")
            texts.append(text)
            labels.append(int(label))
    if not texts:
        raise ValueError(f"{path} contains no records.")
    return texts, np.asarray(labels, dtype=np.int8)


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_precision: float = MIN_PHISHING_PRECISION,
) -> Dict[str, object]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = np.flatnonzero(precision[:-1] >= minimum_precision)
    if eligible.size:
        index = max(eligible, key=lambda i: (recall[i], precision[i], thresholds[i]))
        objective = (
            "maximum phishing recall subject to phishing precision "
            f">= {minimum_precision:.2f} on dev"
        )
        constraint_met = True
    else:
        beta_squared = 4.0
        denominator = beta_squared * precision[:-1] + recall[:-1]
        f2 = np.divide(
            (1 + beta_squared) * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
        index = int(np.argmax(f2))
        objective = "maximum phishing F2 on dev; minimum precision was unattainable"
        constraint_met = False
    return {
        "threshold": float(thresholds[index]),
        "objective": objective,
        "minimum_precision": minimum_precision,
        "constraint_met": constraint_met,
        "precision_at_selection": float(precision[index]),
        "recall_at_selection": float(recall[index]),
    }


def evaluate(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    negative_count = tn + fp
    positive_count = tp + fn
    return {
        "rows": int(labels.size),
        "phishing": {
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
            "average_precision": float(average_precision_score(labels, probabilities)),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_count": int(fp),
        "false_positive_rate": float(fp / negative_count) if negative_count else 0.0,
        "false_negative_count": int(fn),
        "false_negative_rate": float(fn / positive_count) if positive_count else 0.0,
    }


def fit_and_select(
    train_path: Path,
    dev_path: Path,
    tfidf_params: Optional[Dict] = None,
    logistic_params: Optional[Dict] = None,
    minimum_precision: float = MIN_PHISHING_PRECISION,
):
    train_texts, train_labels = load_split(train_path)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**(tfidf_params or TFIDF_PARAMS))),
            ("classifier", LogisticRegression(**(logistic_params or LOGISTIC_PARAMS))),
        ]
    )
    pipeline.fit(train_texts, train_labels)

    dev_texts, dev_labels = load_split(dev_path)
    dev_probabilities = pipeline.predict_proba(dev_texts)[:, 1]
    selection = select_threshold(dev_labels, dev_probabilities, minimum_precision)
    dev_metrics = evaluate(dev_labels, dev_probabilities, selection["threshold"])
    return pipeline, selection, dev_labels, dev_probabilities, dev_metrics, len(train_texts)


def _plot_precision_recall(dev_labels, dev_probabilities, test_labels, test_probabilities, path):
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, labels, probabilities in (
        ("Dev", dev_labels, dev_probabilities),
        ("Locked test", test_labels, test_probabilities),
    ):
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        axis.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    axis.set(xlabel="Phishing recall", ylabel="Phishing precision", title="Precision-recall curves")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_confusion_matrices(dev_metrics, test_metrics, path):
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for axis, name, metrics in zip(axes, ("Dev", "Locked test"), (dev_metrics, test_metrics)):
        matrix = metrics["confusion_matrix"]
        values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
        axis.imshow(values, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(values[row, column]), ha="center", va="center")
        axis.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Benign", "Phishing"], yticklabels=["Benign", "Phishing"],
            xlabel="Predicted", ylabel="Actual", title=name,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            json.dump(data, output, indent=2)
            output.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def train_baseline(
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    model_path: Path,
    metrics_path: Path,
    figures_dir: Path,
) -> Dict:
    pipeline, selection, dev_labels, dev_probabilities, dev_metrics, train_rows = (
        fit_and_select(train_path, dev_path)
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "vectorizer": pipeline.named_steps["tfidf"],
        "model": pipeline.named_steps["classifier"],
        "threshold": selection["threshold"],
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(bundle, model_path)

    test_texts, test_labels = load_split(test_path)
    test_probabilities = pipeline.predict_proba(test_texts)[:, 1]
    test_metrics = evaluate(test_labels, test_probabilities, selection["threshold"])

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "random_seed": RANDOM_SEED,
        "threshold_selection": selection,
        "parameters": {
            "tfidf": {**TFIDF_PARAMS, "ngram_range": list(TFIDF_PARAMS["ngram_range"])},
            "logistic_regression": LOGISTIC_PARAMS,
        },
        "splits": {
            "train": {"rows": train_rows, "source": "trec5"},
            "dev": {"rows": int(dev_labels.size), "source": "trec6"},
            "test": {"rows": int(test_labels.size), "source": "trec7", "locked": True},
        },
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "dev": dev_metrics,
        "test": test_metrics,
        "artifacts": {
            "model": "models/baseline_lr.joblib",
            "precision_recall_curve": "reports/figures/baseline_precision_recall.png",
            "confusion_matrix": "reports/figures/baseline_confusion_matrices.png",
        },
    }
    _write_json(metrics, metrics_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_precision_recall(
        dev_labels, dev_probabilities, test_labels, test_probabilities,
        figures_dir / "baseline_precision_recall.png",
    )
    _plot_confusion_matrices(
        dev_metrics, test_metrics, figures_dir / "baseline_confusion_matrices.png"
    )
    return metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the source-disjoint TF-IDF baseline.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    metrics = train_baseline(
        args.train, args.dev, args.test, args.model, args.metrics, args.figures_dir
    )
    selection = metrics["threshold_selection"]
    print(f"Selected threshold: {selection['threshold']:.6f}")
    print(selection["objective"])
    for split in ("dev", "test"):
        values = metrics[split]
        phishing = values["phishing"]
        print(
            f"{split}: precision={phishing['precision']:.4f} "
            f"recall={phishing['recall']:.4f} f1={phishing['f1']:.4f} "
            f"f2={phishing['f2']:.4f} AP={phishing['average_precision']:.4f} "
            f"FP={values['false_positive_count']} FN={values['false_negative_count']}"
        )


if __name__ == "__main__":
    main()
