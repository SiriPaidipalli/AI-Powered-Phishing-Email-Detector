import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import joblib  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from app_inference import (  # noqa: E402
    MAX_BATCH_ROWS,
    InputValidationError,
    analyze_batch,
    analyze_message,
    safe_spreadsheet_text,
)


class AppInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
                ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
            ]
        )
        pipeline.fit(
            [
                "subject: meeting body: routine agenda",
                "subject: project body: weekly notes",
                "subject: urgent body: verify your account URL",
                "subject: alert body: reset your password URL",
            ],
            [0, 0, 1, 1],
        )
        cls.bundle = {
            "pipeline": pipeline,
            "vectorizer": pipeline.named_steps["tfidf"],
            "model": pipeline.named_steps["classifier"],
            "threshold": 0.5,
        }

    def test_single_analysis_uses_shared_preprocessing_and_keeps_evidence_separate(self):
        result = analyze_message(
            self.bundle,
            "<b>URGENT</b>",
            "Verify your account at https://example.test/login",
        )

        self.assertEqual(
            result["model_text"],
            "subject: urgent body: verify your account at URL",
        )
        self.assertIn("credentials", result["security_analysis"]["categories"])
        self.assertIn("model_evidence", result)
        self.assertIn("risk", result)

    def test_model_evidence_contains_both_named_directions(self):
        result = analyze_message(self.bundle, "Project alert", "Weekly notes and password reset")

        self.assertIn("toward_phishing", result["model_evidence"])
        self.assertIn("toward_legitimate", result["model_evidence"])

    def test_batch_returns_required_fields(self):
        results = analyze_batch(
            self.bundle,
            [{"subject": "Meeting", "body": "Routine agenda"}],
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            set(results[0]),
            {
                "subject",
                "phishing_probability",
                "model_prediction",
                "risk_level",
                "risk_points",
                "indicator_count",
                "indicator_categories",
            },
        )

    def test_batch_rejects_missing_values_and_excessive_size(self):
        with self.assertRaisesRegex(InputValidationError, "body is missing"):
            analyze_batch(self.bundle, [{"subject": "Notice", "body": None}])
        with self.assertRaisesRegex(InputValidationError, "maximum"):
            analyze_batch(
                self.bundle,
                [{"subject": "Notice", "body": "Body"}] * (MAX_BATCH_ROWS + 1),
            )

    def test_real_saved_model_can_run_application_inference(self):
        model_path = REPOSITORY_ROOT / "models" / "baseline_lr.joblib"
        bundle = joblib.load(model_path)

        result = analyze_message(bundle, "Meeting", "The agenda is attached for review.")

        self.assertGreaterEqual(result["ml_probability"], 0.0)
        self.assertLessEqual(result["ml_probability"], 1.0)

    def test_download_text_cannot_become_a_spreadsheet_formula(self):
        self.assertEqual(safe_spreadsheet_text("=HYPERLINK('x')"), "'=HYPERLINK('x')")
        self.assertEqual(safe_spreadsheet_text("Normal subject"), "Normal subject")


if __name__ == "__main__":
    unittest.main()
