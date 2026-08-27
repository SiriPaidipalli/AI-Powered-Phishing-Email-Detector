import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from evaluate_robustness import evaluate_robustness  # noqa: E402
from evasion_transforms import (  # noqa: E402
    TRANSFORMATIONS,
    add_benign_padding,
    insert_html_noise,
    obfuscate_urls,
    obfuscate_whitespace_punctuation,
    soften_credentials,
    soften_urgency,
    substitute_homoglyphs,
)


class EvasionTransformationTests(unittest.TestCase):
    def test_urgency_softening(self):
        subject, body = soften_urgency("Urgent warning", "Act now immediately")
        self.assertNotIn("urgent", subject.lower())
        self.assertNotIn("act now", body.lower())

    def test_credential_softening(self):
        _, body = soften_credentials("Account", "Verify your account and reset your password")
        self.assertNotIn("verify your account", body.lower())
        self.assertIn("account", body.lower())

    def test_benign_padding_is_deterministic(self):
        first = add_benign_padding("Subject", "Body", seed=42)
        second = add_benign_padding("Subject", "Body", seed=42)
        self.assertEqual(first, second)
        self.assertTrue(first[1].startswith("Body"))

    def test_url_obfuscation(self):
        _, body = obfuscate_urls("Link", "Visit https://login.example/path or [URL]")
        self.assertIn("hxxps://", body)
        self.assertIn("[.]", body)
        self.assertIn("[U R L]", body)

    def test_whitespace_punctuation_obfuscation(self):
        _, body = obfuscate_whitespace_punctuation("Alert", "Verify account password")
        self.assertIn("ver.ify", body.lower())
        self.assertIn("acc ount", body.lower())

    def test_homoglyph_substitution(self):
        _, body = substitute_homoglyphs("Alert", "Verify account password")
        self.assertNotEqual(body, "Verify account password")
        self.assertIn("а", body)

    def test_html_noise_insertion(self):
        _, body = insert_html_noise("Alert", "Verify account password")
        self.assertIn("<span></span>", body)
        self.assertIn("account".replace("account", "acc<span></span>ount"), body.lower())


class RobustnessEvaluationTests(unittest.TestCase):
    def test_evaluation_uses_only_provided_phishing_records_and_reports_changes(self):
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
            ]
        )
        pipeline.fit(
            [
                "subject: meeting body: routine agenda",
                "subject: report body: weekly project notes",
                "subject: urgent body: verify account password URL",
                "subject: alert body: click invoice URL",
            ],
            [0, 0, 1, 1],
        )
        records = [
            {
                "subject": "Urgent",
                "body": "Verify your account at [URL]",
                "text": "subject: urgent body: verify your account at [URL]",
            },
            {
                "subject": "Invoice alert",
                "body": "Click the link at [URL]",
                "text": "subject: invoice alert body: click the link at [URL]",
            },
        ]
        bundle = {"pipeline": pipeline, "threshold": 0.5}

        report = evaluate_robustness(bundle, records, TRANSFORMATIONS)

        self.assertEqual(report["original"]["evaluated_messages"], 2)
        self.assertEqual(set(report["transformations"]), set(TRANSFORMATIONS))
        for metrics in report["transformations"].values():
            self.assertEqual(metrics["evaluated_messages"], 2)
            self.assertIn("recall_change_vs_original", metrics)
            self.assertIn("high_critical_percent", metrics)
            self.assertIn("ml_false_negatives_remaining_high_critical_count", metrics)
            self.assertIn("dropped_high_critical_to_medium_low_count", metrics)


if __name__ == "__main__":
    unittest.main()
