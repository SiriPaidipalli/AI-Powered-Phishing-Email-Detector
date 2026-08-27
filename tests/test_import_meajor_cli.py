import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from import_meajor import main  # noqa: E402


FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "meajor_sample.csv"


class MeajorCliTests(unittest.TestCase):
    def test_cli_imports_fixture_and_prints_all_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "canonical.csv"
            metadata_output = Path(directory) / "metadata.jsonl"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--input",
                        str(FIXTURE),
                        "--output",
                        str(output),
                        "--metadata-output",
                        str(metadata_output),
                    ]
                )

            self.assertTrue(output.is_file())
            self.assertTrue(metadata_output.is_file())
            self.assertEqual(
                stdout.getvalue().splitlines(),
                [
                    "Benign rows: 1",
                    "Phishing rows: 1",
                    "Total rows: 2",
                    "Metadata rows: 2",
                ],
            )


if __name__ == "__main__":
    unittest.main()
