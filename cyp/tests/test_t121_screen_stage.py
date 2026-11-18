
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyp/scripts"))
from t121_screen_stage import (
    screening_split, native_screen_score, select_screen_epoch,
    fit_screen_stage,
)


class ScreenStageTests(unittest.TestCase):
    def test_split_reproducibility_and_whole_clusters(self):
        ids = ["A", "B", "C", "D", "E", "F"]
        smiles = ["CC", "CC", "c1ccccc1", "c1ccccc1", "O", "N"]
        first = screening_split(ids, smiles)
        second = screening_split(ids[::-1], smiles[::-1])
        self.assertEqual(first, second)
        self.assertEqual(first["A"], first["B"])
        self.assertEqual(first["C"], first["D"])
        partitions = {}
        for value in first.values():
            partitions.setdefault(value["cluster_id"], set()).add(
                value["partition"]
            )
        self.assertTrue(all(len(value) == 1 for value in partitions.values()))
        self.assertEqual(
            {value["partition"] for value in first.values()},
            {"train", "validation"},
        )

    def test_native_metric_manual_example(self):
        targets = np.zeros((2, 4))
        predictions = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
        weights = np.array([[1, 1, 1, 1], [3, 3, 3, 3]])
        score = native_screen_score(predictions, targets, weights)
        expected = [7.0, 13.0, 21.0, 31.0]
        np.testing.assert_allclose(score["per_enzyme"], expected)
        self.assertEqual(score["macro"], 18.0)
        # Native residuals must not be confused with standardized residuals.
        scaled = native_screen_score(predictions / 2, targets, weights)
        self.assertEqual(scaled["macro"], 4.5)

    def test_missing_labels_and_invalid_observed_weights(self):
        targets = np.zeros((2, 4))
        targets[0, 0] = np.nan
        weights = np.ones((2, 4))
        weights[0, 0] = np.nan
        score = native_screen_score(np.ones((2, 4)), targets, weights)
        self.assertEqual(score["macro"], 1.0)
        weights[1, 0] = 0
        with self.assertRaises(ValueError):
            native_screen_score(np.ones((2, 4)), targets, weights)

    def test_selection_minimum_and_earlier_exact_tie(self):
        records = [
            {"epoch": 3, "score": 2.0},
            {"epoch": 2, "score": 1.0},
            {"epoch": 1, "score": 1.0},
        ]
        self.assertEqual(select_screen_epoch(records)["epoch"], 1)

    def test_training_requires_authorization(self):
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            fit_screen_stage({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
