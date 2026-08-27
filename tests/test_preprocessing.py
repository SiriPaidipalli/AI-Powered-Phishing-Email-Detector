import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from preprocessing import (  # noqa: E402
    DataValidationError,
    canonicalize_email,
    preprocess_csv,
    preprocess_email,
    validate_and_preprocess_record,
)


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_email_normalizes_html_urls_and_email_addresses(self):
        subject, body, text = preprocess_email(
            " <b>Security Alert</b> ",
            "Contact Help@Example.com at https://example.com/reset",
        )

        self.assertEqual(subject, "security alert")
        self.assertEqual(body, "contact EMAIL at URL")
        self.assertEqual(text, "subject: security alert body: contact EMAIL at URL")

    def test_duplicate_canonicalization_preserves_url_and_email_identity(self):
        first = canonicalize_email(
            "Alert", "Visit https://first.example or contact first@example.com"
        )
        second = canonicalize_email(
            "Alert", "Visit https://second.example or contact second@example.com"
        )

        self.assertNotEqual(first, second)

    def test_rejects_missing_subject_or_body(self):
        for record in (
            {"subject": "", "body": "Message", "label": "0"},
            {"subject": "Subject", "body": None, "label": "0"},
        ):
            with self.subTest(record=record):
                with self.assertRaisesRegex(DataValidationError, "missing or blank"):
                    validate_and_preprocess_record(record, 2)

    def test_rejects_invalid_label(self):
        with self.assertRaisesRegex(DataValidationError, "label must be 0 or 1"):
            validate_and_preprocess_record(
                {"subject": "Subject", "body": "Message", "label": "phishing"}, 2
            )

    def test_rejects_message_empty_after_preprocessing(self):
        with self.assertRaisesRegex(DataValidationError, "empty after preprocessing"):
            validate_and_preprocess_record(
                {"subject": "<br>", "body": "<style>hidden</style>", "label": "1"}, 2
            )

    def test_distinct_urls_do_not_count_as_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            self._write_csv(
                input_path,
                [
                    ["Security alert", "Visit https://first.example", "1"],
                    ["Security alert", "Visit https://second.example", "1"],
                ],
            )

            counts = preprocess_csv(input_path, output_path)

            self.assertEqual(counts["total"], 2)

    def test_rejects_canonical_duplicate_messages(self):
        error = self._run_csv(
            [
                ["<b>Security alert</b>", "Visit https://same.example", "1"],
                [" SECURITY ALERT ", "  Visit https://same.example  ", "1"],
            ]
        )
        self.assertIn("duplicates row 2", str(error))

    def test_rejects_conflicting_duplicate_labels(self):
        error = self._run_csv(
            [
                ["Security alert", "Reset your password", "1"],
                ["security alert", "Reset your password", "0"],
            ]
        )
        self.assertIn("different labels", str(error))

    def test_streams_valid_csv_to_canonical_output(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "nested" / "output.csv"
            self._write_csv(
                input_path,
                [
                    ["Account notice", "Review your account", "1"],
                    ["Meeting", "Agenda attached", "0"],
                ],
            )

            counts = preprocess_csv(input_path, output_path)

            self.assertEqual(
                counts, {"total": 2, "phishing": 1, "legitimate": 1}
            )
            with output_path.open(newline="", encoding="utf-8") as output:
                rows = list(csv.DictReader(output))
            self.assertEqual(rows[0]["text"], "subject: account notice body: review your account")

    def _run_csv(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            self._write_csv(input_path, rows)
            with self.assertRaises(DataValidationError) as context:
                preprocess_csv(input_path, output_path)
            self.assertFalse(output_path.exists())
            return context.exception

    @staticmethod
    def _write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["subject", "body", "label"])
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
