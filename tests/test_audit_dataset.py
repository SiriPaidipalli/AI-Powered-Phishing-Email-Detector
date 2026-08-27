import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from audit_dataset import audit_dataset, write_summary  # noqa: E402


class DatasetAuditTests(unittest.TestCase):
    def test_reports_duplicates_conflicts_missing_values_and_sources(self):
        rows = [
            ["Notice", "Same body", "0"],
            ["Notice", "Same body", "0"],
            ["<b>Alert</b>", "Shared body", "0"],
            [" ALERT ", " shared  body ", "1"],
            ["", "Body without a subject", "1"],
            ["Link", "Visit https://first.example", "1"],
            ["Link", "Visit https://second.example", "1"],
        ]
        sources = ["trec5", "trec5", "trec6", "trec6", "nazario", "nazario", "nazario"]

        with tempfile.TemporaryDirectory() as directory:
            canonical_path = Path(directory) / "canonical.csv"
            metadata_path = Path(directory) / "metadata.jsonl"
            output_path = Path(directory) / "audit.json"
            with canonical_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["subject", "body", "label"])
                writer.writerows(rows)
            with metadata_path.open("w", encoding="utf-8") as metadata_file:
                for index, source in enumerate(sources, start=1):
                    metadata_file.write(
                        json.dumps(
                            {
                                "canonical_row": index,
                                "source_row": index + 1,
                                "source_fields": {"source": source},
                            }
                        )
                        + "\n"
                    )

            summary = audit_dataset(canonical_path, metadata_path)
            write_summary(summary, output_path)

            self.assertEqual(summary["records"]["total"], 7)
            self.assertEqual(summary["missing"]["blank_subject"], 1)
            self.assertEqual(
                summary["duplicates"]["exact"]["duplicate_records_beyond_first"], 1
            )
            self.assertEqual(
                summary["duplicates"]["canonicalized"]["duplicate_records_beyond_first"], 2
            )
            self.assertEqual(
                summary["duplicates"]["canonicalized"]["conflicting_label_groups"], 1
            )
            self.assertEqual(
                summary["duplicates"]["retained_after_recommended_deduplication"], 4
            )
            self.assertEqual(summary["sources"]["nazario"]["total"], 3)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
