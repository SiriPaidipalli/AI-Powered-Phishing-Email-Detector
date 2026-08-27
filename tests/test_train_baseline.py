import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

try:
    import joblib
    import numpy as np

    from train_baseline import fit_and_select, select_threshold
except ImportError as error:
    raise unittest.SkipTest(f"ML dependencies are not installed: {error}")


SMALL_TFIDF_PARAMS = {"ngram_range": (1, 1), "min_df": 1}
SMALL_LOGISTIC_PARAMS = {
    "solver": "liblinear",
    "random_state": 42,
    "max_iter": 200,
}


class BaselineTrainingTests(unittest.TestCase):
    def test_vectorizer_is_fitted_only_on_training_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            dev = root / "dev.csv"
            self._write_split(train, [("trainonly benign", 0), ("trainonly phishing", 1)])
            self._write_split(dev, [("devonly benign", 0), ("devonly phishing", 1)])
            pipeline, *_ = fit_and_select(
                train, dev, SMALL_TFIDF_PARAMS, SMALL_LOGISTIC_PARAMS, minimum_precision=0.5
            )
            vocabulary = pipeline.named_steps["tfidf"].vocabulary_
            self.assertIn("trainonly", vocabulary)
            self.assertNotIn("devonly", vocabulary)

    def test_threshold_maximizes_recall_at_minimum_precision(self):
        labels = np.asarray([1, 0, 1, 0])
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])
        selection = select_threshold(labels, probabilities, minimum_precision=0.65)
        self.assertTrue(selection["constraint_met"])
        self.assertAlmostEqual(selection["threshold"], 0.7)
        self.assertAlmostEqual(selection["recall_at_selection"], 1.0)
        self.assertGreaterEqual(selection["precision_at_selection"], 0.65)

    def test_fit_and_select_does_not_require_or_read_a_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            dev = root / "dev.csv"
            self._write_split(train, [("safe note", 0), ("urgent link", 1)])
            self._write_split(dev, [("safe update", 0), ("urgent update", 1)])
            result = fit_and_select(
                train, dev, SMALL_TFIDF_PARAMS, SMALL_LOGISTIC_PARAMS, minimum_precision=0.5
            )
            self.assertEqual(result[-1], 2)
            self.assertFalse((root / "test.csv").exists())

    def test_saved_pipeline_can_run_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            dev = root / "dev.csv"
            model_path = root / "model.joblib"
            self._write_split(train, [("routine agenda", 0), ("urgent password", 1)])
            self._write_split(dev, [("routine update", 0), ("urgent account", 1)])
            pipeline, selection, *_ = fit_and_select(
                train, dev, SMALL_TFIDF_PARAMS, SMALL_LOGISTIC_PARAMS, minimum_precision=0.5
            )
            joblib.dump({"pipeline": pipeline, "threshold": selection["threshold"]}, model_path)
            bundle = joblib.load(model_path)
            probability = bundle["pipeline"].predict_proba(["urgent password"])[0, 1]
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    @staticmethod
    def _write_split(path, rows):
        with path.open("w", encoding="utf-8", newline="") as split_file:
            writer = csv.DictWriter(split_file, fieldnames=["text", "label"])
            writer.writeheader()
            for text, label in rows:
                writer.writerow({"text": text, "label": label})


if __name__ == "__main__":
    unittest.main()
