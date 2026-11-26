"""Tests for the CYP screening-head loading smoke (Tripwire-CYP-010).

Each test is tied to a concrete failure risk: a wrong or silently reordered CYP3A4 output
column, a tampered or mismatched checkpoint or selection record, a split file read wrongly
because of its CRLF line endings, a non-deterministic smoke molecule choice, a model that is
not on native log2fc units, an output directory being reused, and an accidental score.
Writing these tests is not a passing run.
"""
import json
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyp/scripts"))
import screen_head_smoke_cyp as smoke  # noqa: E402

PINS = Path(__file__).resolve().parents[2] / "cyp/artifacts/screen_heads_smoke/cyp_r1_pins.json"
EXPECTED = {
    "n_epoch_records": 3, "selected_epoch_zero_based": 1, "selected_score": 0.5,
    "selected_per_enzyme": [0.1, 0.2, 0.3, 0.4],
}


def selection_payload():
    records = [
        {"epoch": 0, "score": 0.9, "per_enzyme": [1, 1, 1, 1], "checkpoint": "a", "sha256": "h0"},
        {"epoch": 1, "score": 0.5, "per_enzyme": [0.1, 0.2, 0.3, 0.4], "checkpoint": "b", "sha256": "h1"},
        {"epoch": 2, "score": 0.7, "per_enzyme": [2, 2, 2, 2], "checkpoint": "c", "sha256": "h2"},
    ]
    return {"epoch_numbering": "zero_based", "records": records, "selected": records[1]}


class HashAndOutputTests(unittest.TestCase):
    def test_verify_hashed_files_detects_changed_and_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.txt"
            path.write_text("abc")
            good = {"role": {"path": str(path), "sha256": smoke.sha256_file(path)}}
            self.assertEqual(smoke.verify_hashed_files(good)["role"]["sha256"], smoke.sha256_file(path))
            bad = {"role": {"path": str(path), "sha256": "0" * 64}}
            with self.assertRaises(ValueError):
                smoke.verify_hashed_files(bad)
            missing = {"role": {"path": str(Path(tmp) / "none"), "sha256": "0" * 64}}
            with self.assertRaises(ValueError):
                smoke.verify_hashed_files(missing)

    def test_output_directory_is_never_reused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            smoke.new_output_dir(target)
            with self.assertRaises(FileExistsError):
                smoke.new_output_dir(target)


class ColumnOrderTests(unittest.TestCase):
    ORDER = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

    def test_correct_order_gives_cyp3a4_index_three(self):
        self.assertEqual(smoke.check_enzyme_order(self.ORDER, self.ORDER, self.ORDER), 3)

    def test_swapped_model_order_is_rejected(self):
        swapped = ("CYP1A2", "CYP2C9", "CYP3A4", "CYP2D6")
        with self.assertRaises(ValueError):
            smoke.check_enzyme_order(swapped, self.ORDER, self.ORDER)

    def test_contract_order_change_is_rejected(self):
        swapped = ("CYP2C9", "CYP1A2", "CYP2D6", "CYP3A4")
        with self.assertRaises(ValueError):
            smoke.check_enzyme_order(self.ORDER, swapped, self.ORDER)


class SplitFileTests(unittest.TestCase):
    def write(self, tmp, text, newline):
        path = Path(tmp) / "split.csv"
        with open(path, "w", newline="") as handle:
            handle.write(text.replace("\n", newline))
        return path

    def test_crlf_and_lf_files_read_identically(self):
        import tempfile
        text = "molecule_id,cluster_id,partition\nB,1,validation\nA,0,train\nC,2,train\n"
        with tempfile.TemporaryDirectory() as tmp:
            crlf = smoke.read_split_assignments(self.write(tmp, text, "\r\n"))
            lf = smoke.read_split_assignments(self.write(tmp, text, "\n"))
        self.assertEqual(crlf, lf)
        self.assertEqual(crlf["B"], "validation")
        self.assertEqual(smoke.partition_counts(crlf), {"train": 2, "validation": 1})

    def test_bad_header_and_duplicates_are_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                smoke.read_split_assignments(self.write(tmp, "a,b,c\n1,2,3\n", "\n"))
            with self.assertRaises(ValueError):
                smoke.read_split_assignments(self.write(
                    tmp, "molecule_id,cluster_id,partition\nA,0,train\nA,1,train\n", "\n"))

    def test_smoke_molecule_choice_is_deterministic_and_order_independent(self):
        assignments = {f"M{i:03d}": ("validation" if i % 5 == 0 else "train") for i in range(60)}
        first = smoke.select_smoke_ids(assignments, 4)
        shuffled = dict(reversed(list(assignments.items())))
        self.assertEqual(first, smoke.select_smoke_ids(shuffled, 4))
        self.assertEqual(first["train"], sorted(first["train"]))
        self.assertTrue(all(assignments[n] == "validation" for n in first["validation"]))
        with self.assertRaises(ValueError):
            smoke.select_smoke_ids(assignments, 50)


class SelectionRecordTests(unittest.TestCase):
    def test_consistent_records_pass(self):
        selection = selection_payload()
        smoke.verify_selection_records(selection, selection["selected"], EXPECTED, "h1")

    def test_wrong_checkpoint_hash_is_rejected(self):
        selection = selection_payload()
        with self.assertRaises(ValueError):
            smoke.verify_selection_records(selection, selection["selected"], EXPECTED, "other")

    def test_selected_record_differing_from_selection_is_rejected(self):
        selection = selection_payload()
        altered = dict(selection["selected"], score=0.4)
        with self.assertRaises(ValueError):
            smoke.verify_selection_records(selection, altered, EXPECTED, "h1")

    def test_selected_epoch_that_is_not_the_minimum_is_rejected(self):
        selection = selection_payload()
        selection["records"][0]["score"] = 0.1
        with self.assertRaises(ValueError):
            smoke.verify_selection_records(selection, selection["selected"], EXPECTED, "h1")

    def test_wrong_record_count_is_rejected(self):
        selection = selection_payload()
        selection["records"].pop()
        with self.assertRaises(ValueError):
            smoke.verify_selection_records(selection, selection["selected"], EXPECTED, "h1")


class OutputCheckTests(unittest.TestCase):
    def test_shape_finite_and_bound(self):
        stats = smoke.check_native_outputs(np.zeros((16, 4)), 16, 4, 10.0)
        self.assertEqual(stats["shape"], [16, 4])
        with self.assertRaises(ValueError):
            smoke.check_native_outputs(np.zeros((16, 3)), 16, 4, 10.0)
        bad = np.zeros((16, 4))
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            smoke.check_native_outputs(bad, 16, 4, 10.0)
        big = np.zeros((16, 4))
        big[3, 3] = 11.0
        with self.assertRaises(ValueError):
            smoke.check_native_outputs(big, 16, 4, 10.0)

    def test_checks_recorder_requires_every_check_to_pass(self):
        checks = smoke.Checks()
        self.assertFalse(checks.passed)  # nothing recorded is not a pass
        checks.record("a", True)
        self.assertTrue(checks.passed)
        checks.record("b", False)
        self.assertFalse(checks.passed)


class NativeHeadTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
            from chemprop import nn  # noqa: F401
        except Exception:
            self.skipTest("torch/chemprop unavailable")

    def build(self, with_unscale):
        from chemprop import models, nn
        from sklearn.preprocessing import StandardScaler

        transform = None
        if with_unscale:
            scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(20, 4)))
            transform = nn.transforms.UnscaleTransform.from_standard_scaler(scaler)
        predictor = nn.RegressionFFN(
            n_tasks=4, input_dim=16, hidden_dim=8, n_layers=1, output_transform=transform)
        return models.MPNN(
            message_passing=nn.BondMessagePassing(d_h=16, depth=2),
            agg=nn.MeanAggregation(), predictor=predictor, batch_norm=False)

    def test_identity_head_with_four_outputs_passes(self):
        info = smoke.assert_native_identity_head(self.build(False), 4)
        self.assertEqual(info["out_features"], 4)

    def test_scaled_head_is_rejected_as_not_native(self):
        with self.assertRaises(ValueError):
            smoke.assert_native_identity_head(self.build(True), 4)

    def test_wrong_output_count_is_rejected(self):
        with self.assertRaises(ValueError):
            smoke.assert_native_identity_head(self.build(False), 3)


class PinsFileTests(unittest.TestCase):
    def test_pins_file_is_consistent(self):
        pins = json.loads(PINS.read_text())
        self.assertEqual(pins["expected"]["enzyme_order"], list(smoke.CYP_ENZYMES))
        self.assertEqual(pins["expected"]["cyp3a4_index"], 3)
        self.assertIn("model_side", pins["pinned_files"])
        for entry in list(pins["pinned_files"].values()) + list(pins["run_files"].values()):
            self.assertEqual(len(entry["sha256"]), 64)
        self.assertTrue(pins["run_files"]["checkpoint"]["path"].endswith("epoch_016.ckpt"))
        self.assertEqual(sum(pins["expected"]["split_counts"].values()), 6947)


if __name__ == "__main__":
    unittest.main()
