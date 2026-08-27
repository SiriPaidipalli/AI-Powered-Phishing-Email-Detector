from collections import defaultdict
from typing import Dict, Iterable, List, Mapping

import numpy as np

from preprocessing import preprocess_email
from risk_engine import assess_risk
from security_analyzer import analyze_email


MAX_BATCH_ROWS = 500
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class InputValidationError(ValueError):
    pass


def _validate_message(subject, body, location: str = "Message") -> None:
    if not isinstance(subject, str) or not subject.strip():
        raise InputValidationError(f"{location}: subject is missing or blank.")
    if not isinstance(body, str) or not body.strip():
        raise InputValidationError(f"{location}: body is missing or blank.")
    subject_clean, body_clean, _ = preprocess_email(subject, body)
    if not subject_clean and not body_clean:
        raise InputValidationError(f"{location}: no usable text remains after preprocessing.")


def _model_components(bundle: Mapping):
    pipeline = bundle.get("pipeline")
    if pipeline is not None:
        return pipeline, pipeline.named_steps["tfidf"], pipeline.named_steps["classifier"]
    return None, bundle["vectorizer"], bundle["model"]


def _probability(bundle: Mapping, text: str) -> float:
    pipeline, vectorizer, model = _model_components(bundle)
    if pipeline is not None:
        return float(pipeline.predict_proba([text])[0, 1])
    return float(model.predict_proba(vectorizer.transform([text]))[0, 1])


def model_feature_contributions(
    bundle: Mapping, text: str, limit: int = 8
) -> Dict[str, List[Dict[str, float]]]:
    _, vectorizer, model = _model_components(bundle)
    if not hasattr(model, "coef_"):
        return {"toward_phishing": [], "toward_legitimate": []}
    vector = vectorizer.transform([text])
    coefficients = model.coef_[0]
    names = vectorizer.get_feature_names_out()
    contributions = [
        (names[index], float(coefficients[index] * vector[0, index]))
        for index in vector.nonzero()[1]
    ]
    phishing = sorted(
        (item for item in contributions if item[1] > 0), key=lambda item: item[1], reverse=True
    )[:limit]
    legitimate = sorted(
        (item for item in contributions if item[1] < 0), key=lambda item: item[1]
    )[:limit]
    return {
        "toward_phishing": [
            {"feature": feature, "contribution": contribution}
            for feature, contribution in phishing
        ],
        "toward_legitimate": [
            {"feature": feature, "contribution": contribution}
            for feature, contribution in legitimate
        ],
    }


def group_indicators(indicators: Iterable[Mapping]) -> Dict[str, List[Mapping]]:
    grouped = defaultdict(list)
    for indicator in indicators:
        grouped[str(indicator["category"])].append(indicator)
    return dict(grouped)


def safe_spreadsheet_text(value: str) -> str:
    """Prevent user-controlled text from becoming a spreadsheet formula."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def analyze_message(bundle: Mapping, subject: str, body: str) -> Dict[str, object]:
    _validate_message(subject, body)
    _, _, model_text = preprocess_email(subject, body)
    probability = _probability(bundle, model_text)
    threshold = float(bundle["threshold"])
    security_analysis = analyze_email(subject, body)
    risk = assess_risk(probability, security_analysis)
    return {
        "subject": subject,
        "model_text": model_text,
        "ml_probability": probability,
        "model_threshold": threshold,
        "model_prediction": "Phishing" if probability >= threshold else "Legitimate",
        "model_evidence": model_feature_contributions(bundle, model_text),
        "security_analysis": security_analysis,
        "indicator_groups": group_indicators(security_analysis["indicators"]),
        "risk": risk,
    }


def analyze_batch(bundle: Mapping, records: Iterable[Mapping]) -> List[Dict[str, object]]:
    records = list(records)
    if not records:
        raise InputValidationError("Uploaded CSV contains no data rows.")
    if len(records) > MAX_BATCH_ROWS:
        raise InputValidationError(
            f"Batch contains {len(records)} rows; the maximum is {MAX_BATCH_ROWS}."
        )

    results = []
    errors = []
    for row_number, record in enumerate(records, start=2):
        try:
            result = analyze_message(bundle, record.get("subject"), record.get("body"))
        except InputValidationError as error:
            errors.append(str(error).replace("Message:", f"CSV row {row_number}:"))
            continue
        analysis = result["security_analysis"]
        results.append(
            {
                "subject": result["subject"],
                "phishing_probability": result["ml_probability"],
                "model_prediction": result["model_prediction"],
                "risk_level": result["risk"]["risk_level"],
                "risk_points": result["risk"]["risk_points"],
                "indicator_count": analysis["indicator_count"],
                "indicator_categories": ", ".join(analysis["categories"]),
            }
        )
    if errors:
        preview = " ".join(errors[:5])
        remaining = len(errors) - 5
        if remaining > 0:
            preview += f" {remaining} additional invalid rows were found."
        raise InputValidationError(preview)
    return results
