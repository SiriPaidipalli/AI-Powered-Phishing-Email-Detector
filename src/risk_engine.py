from typing import Dict

from security_analyzer import analyze_email


def _ml_points(probability: float) -> int:
    """Map ML probability to visible risk points; this is not calibration."""
    if probability < 0.25:
        return 0
    if probability < 0.50:
        return 1
    if probability < 0.75:
        return 2
    return 4


def _risk_level(points: int) -> str:
    if points <= 1:
        return "Low"
    if points <= 3:
        return "Medium"
    if points <= 5:
        return "High"
    return "Critical"


def assess_risk(ml_probability: float, analysis: Dict[str, object]) -> Dict[str, object]:
    """Combine ML and rules as transparent points, not as a new probability.

    ML contributes 0, 1, 2, or 4 points at probability boundaries 0.25, 0.50,
    and 0.75. Each distinct rule category contributes one point, capped at four.
    Total points map to Low (0-1), Medium (2-3), High (4-5), or Critical (6-8).
    """
    if not isinstance(ml_probability, (int, float)) or not 0 <= ml_probability <= 1:
        raise ValueError("ML probability must be a number between 0 and 1.")
    categories = list(dict.fromkeys(analysis.get("categories", [])))
    ml_points = _ml_points(float(ml_probability))
    indicator_points = min(len(categories), 4)
    total_points = ml_points + indicator_points
    return {
        "risk_level": _risk_level(total_points),
        "risk_points": total_points,
        "ml_probability": float(ml_probability),
        "ml_points": ml_points,
        "indicator_points": indicator_points,
        "indicator_categories": categories,
        "analysis": analysis,
        "disclaimer": "Risk points are a deterministic triage score, not a calibrated probability.",
    }


def analyze_email_risk(subject: str, body: str, ml_probability: float) -> Dict[str, object]:
    return assess_risk(ml_probability, analyze_email(subject, body))
