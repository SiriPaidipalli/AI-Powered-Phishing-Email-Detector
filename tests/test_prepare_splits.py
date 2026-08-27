import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from prepare_splits import prepare_splits  # noqa: E402


class PrepareSplitsTests(unittest.TestCase):
    def test_deduplicates_filters_and_builds_source_disjoint_splits(self):
        rows = [
            ["Train", "Same message", "0"],
            [" TRAIN ", " same  message ", "0"],
            ["Conflict", "Shared", "0"],
            ["<b>Conflict</b>", "Shared", "1"],
            ["", "Missing subject", "0"],
            ["Development", "Dev body", "1"],
            ["Testing", "Test body", "0"],
        ]
        sources = ["trec5", "trec6", "trec5", "trec7", "trec5", "trec6", "trec7"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.csv"
            metadata_path = root / "metadata.jsonl"
            output_dir = root / "splits"
            self._write_inputs(canonical_path, metadata_path, rows, sources)

            summary = prepare_splits(canonical_path, metadata_path, output_dir)

            self.assertEqual(summary["deduplication"]["duplicate_groups"], 2)
            self.assertEqual(summary["deduplication"]["conflicting_label_groups"], 1)
            self.assertEqual(summary["deduplication"]["cross_source_duplicate_groups"], 2)
            self.assertEqual(summary["retained_total"], 3)
            self.assertEqual(
                summary["deduplication"]["groups_excluded_for_blank_subject_or_body"],
                1,
            )
            self.assertEqual(summary["splits"]["train"]["sources"], {"trec5": 1})
            self.assertEqual(summary["splits"]["dev"]["sources"], {"trec6": 1})
            self.assertEqual(summary["splits"]["test"]["sources"], {"trec7": 1})
            self.assertEqual(summary["splits"]["dev"]["phishing"], 1)
            self.assertEqual(summary["splits"]["test"]["benign"], 1)

            with (output_dir / "train.csv").open(
                "r", encoding="utf-8", newline=""
            ) as train_file:
                train_rows = list(csv.DictReader(train_file))
            self.assertEqual(train_rows[0]["subject"], "Train")
            self.assertIn("text", train_rows[0])

            with (output_dir / "train_metadata.jsonl").open(
                "r", encoding="utf-8"
            ) as metadata_file:
                retained_metadata = json.loads(metadata_file.readline())
            self.assertEqual(retained_metadata["canonical_row"], 1)
            self.assertEqual(retained_metadata["split"], "train")
            self.assertEqual(retained_metadata["split_row"], 1)

    def test_rejects_misaligned_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.csv"
            metadata_path = root / "metadata.jsonl"
            self._write_inputs(
                canonical_path,
                metadata_path,
                [["Subject", "Body", "0"]],
                ["trec5"],
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "canonical_row": 2,
                        "source_row": 2,
                        "source_fields": {"source": "trec5"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Metadata alignment error"):
                prepare_splits(canonical_path, metadata_path, root / "splits")

    @staticmethod
    def _write_inputs(canonical_path, metadata_path, rows, sources):
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


if __name__ == "__main__":
    unittest.main()
