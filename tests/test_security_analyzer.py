import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from risk_engine import analyze_email_risk, assess_risk  # noqa: E402
from security_analyzer import analyze_email  # noqa: E402


class SecurityAnalyzerTests(unittest.TestCase):
    def test_detects_urgency(self):
        result = analyze_email("Urgent", "Act now; this expires today.")

        self.assertIn("urgency", result["categories"])

    def test_detects_credential_request(self):
        result = analyze_email("Account", "Verify your account and enter your password.")

        self.assertIn("credentials", result["categories"])

    def test_detects_payment_and_invoice_language(self):
        result = analyze_email("Invoice", "The payment is overdue. Send your bank details.")

        self.assertIn("payment", result["categories"])

    def test_extracts_urls_and_detects_visible_domain_mismatch(self):
        result = analyze_email(
            "Link",
            '<a href="https://destination.example/login">https://visible.example</a>',
        )

        self.assertEqual(result["url_count"], 2)
        self.assertIn("https://destination.example/login", result["urls"])
        self.assertIn("url_domain_mismatch", result["categories"])

    def test_benign_email_has_no_indicators(self):
        result = analyze_email("Meeting notes", "Here are the notes from our weekly meeting.")

        self.assertEqual(result["indicator_count"], 0)
        self.assertEqual(result["categories"], [])
        self.assertEqual(result["urls"], [])

    def test_malformed_url_is_reported_without_crashing(self):
        result = analyze_email("Link", "Visit http://[invalid/path")

        self.assertIn("malformed_url", result["categories"])

    def test_combined_risk_scoring_keeps_inputs_visible(self):
        result = analyze_email_risk(
            "Urgent security alert",
            "Verify your account and click here immediately.",
            0.82,
        )

        self.assertEqual(result["ml_probability"], 0.82)
        self.assertEqual(result["ml_points"], 4)
        self.assertEqual(result["indicator_points"], 4)
        self.assertEqual(result["risk_points"], 8)
        self.assertEqual(result["risk_level"], "Critical")
        self.assertIn("not a calibrated probability", result["disclaimer"])

    def test_low_risk_score(self):
        analysis = analyze_email("Meeting", "The meeting starts at noon.")

        result = assess_risk(0.10, analysis)

        self.assertEqual(result["risk_points"], 0)
        self.assertEqual(result["risk_level"], "Low")


if __name__ == "__main__":
    unittest.main()
