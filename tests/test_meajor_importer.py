import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from importers.meajor import import_meajor_csv  # noqa: E402
from preprocessing import DataValidationError  # noqa: E402


FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "meajor_sample.csv"


class MeajorImporterTests(unittest.TestCase):
    def test_maps_canonical_fields_and_preserves_other_fields_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            canonical_path = Path(directory) / "canonical.csv"
            metadata_path = Path(directory) / "metadata.jsonl"

            counts = import_meajor_csv(FIXTURE, canonical_path, metadata_path)

            self.assertEqual(
                counts, {"total": 2, "phishing": 1, "legitimate": 1}
            )
            with canonical_path.open(newline="", encoding="utf-8") as canonical_file:
                canonical_rows = list(csv.DictReader(canonical_file))
            self.assertEqual(
                list(canonical_rows[0]), ["subject", "body", "label"]
            )
            self.assertEqual(canonical_rows[1]["label"], "1")

            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = [json.loads(line) for line in metadata_file]
            self.assertEqual(metadata[0]["source_fields"]["source"], "trec5")
            self.assertEqual(metadata[1]["source_fields"]["url_count"], "1")
            self.assertNotIn("subject", metadata[0]["source_fields"])

    def test_rejects_unconfirmed_label_values_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "invalid.csv"
            canonical_path = Path(directory) / "canonical.csv"
            metadata_path = Path(directory) / "metadata.jsonl"
            input_path.write_text(
                "subject,body,label\nNotice,Message,phishing\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(DataValidationError, "expected 0 or 1"):
                import_meajor_csv(input_path, canonical_path, metadata_path)

            self.assertFalse(canonical_path.exists())
            self.assertFalse(metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
