"""Tests for screen_predict.py (Tripwire-CYP-010, D025).

Each test is tied to a concrete failure risk: identifiers being altered by type inference or
stripping, an invalid or blank input row being silently dropped or shifting later rows, duplicate
identifiers going unrecorded, a protected file being accepted as input, a mean computed from
fewer than three seeds, a manifest naming the wrong number of checkpoints, the inference model
being switched into training mode, batch size changing values, and an existing output directory
being reused. Writing these tests is not a passing run.
"""
import json
import sys
from pathlib import Path
import unittest
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyp/scripts"))
import screen_predict as sp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO / "cyp/artifacts/screen_predict_smoke/synthetic_input.csv"
PROTECTED = REPO / "cyp/artifacts/screen_predict/protected_inputs.json"


def write(tmp, name, text):
    path = Path(tmp) / name
    path.write_bytes(text.encode("utf-8"))
    return path


class InputReadingTests(unittest.TestCase):
    def test_ids_stay_literal_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "in.csv", "sample_id,smiles\n00042,CCO\n1E5,CCN\nNA,CCC\nnan,CCCl\n\"a, b\",CO\nws ,CN\n")
            rows = sp.read_input_rows(path, "sample_id", "smiles")
        self.assertEqual([r.id for r in rows], ["00042", "1E5", "NA", "nan", "a, b", "ws "])
        self.assertEqual([r.index for r in rows], list(range(6)))

    def test_bom_header_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "in.csv", "﻿id,note,smiles\n7,x,CCO\n")
            rows = sp.read_input_rows(path, "id", "smiles")
        self.assertEqual((rows[0].id, rows[0].smiles), ("7", "CCO"))

    def test_missing_column_duplicate_header_and_empty_file_stop_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sp.read_input_rows(write(tmp, "a.csv", "id,smiles\n1,CCO\n"), "id", "SMILES")
            with self.assertRaises(ValueError):
                sp.read_input_rows(write(tmp, "b.csv", "id,id,smiles\n1,2,CCO\n"), "id", "smiles")
            with self.assertRaises(ValueError):
                sp.read_input_rows(write(tmp, "c.csv", ""), "id", "smiles")
            with self.assertRaises(ValueError):
                sp.read_input_rows(write(tmp, "d.csv", "id,smiles\n"), "id", "smiles")
            with self.assertRaises(ValueError):
                sp.read_input_rows(write(tmp, "e.csv", "id,smiles\n1,CCO\n"), "id", "id")

    def test_blank_or_short_line_stops_instead_of_dropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sp.read_input_rows(write(tmp, "a.csv", "id,smiles\n1,CCO\n\n2,CCN\n"), "id", "smiles")
            with self.assertRaises(ValueError):
                sp.read_input_rows(write(tmp, "b.csv", "id,smiles\n1,CCO\n2\n"), "id", "smiles")


class ClassificationTests(unittest.TestCase):
    def rows(self, pairs):
        return [sp.InputRow(i, i + 2, a, b) for i, (a, b) in enumerate(pairs)]

    def test_statuses_reasons_and_row_positions(self):
        rows = self.rows([("a", "CCO"), ("", "CCN"), ("b", ""), ("   ", "CCC"), ("c", " CCO ")])
        info = sp.classify_rows(rows)
        self.assertEqual([i["status"] for i in info], [
            sp.STATUS_OK, sp.STATUS_MISSING_ID, sp.STATUS_EMPTY_SMILES, sp.STATUS_MISSING_ID, sp.STATUS_OK])
        self.assertIn("smiles_has_surrounding_whitespace", info[4]["warnings"])
        self.assertTrue(info[1]["reason"])

    def test_duplicates_recorded_literally_and_not_for_blank_ids(self):
        rows = self.rows([("00042", "CCO"), ("00042", "CCN"), ("42", "CCC"), ("", "C"), ("", "CC")])
        self.assertEqual(sp.duplicate_ids(rows), {"00042": [0, 1]})
        info = sp.classify_rows(rows)
        self.assertIn("duplicate_id", info[0]["warnings"])
        self.assertIn("duplicate_id", info[1]["warnings"])
        self.assertNotIn("duplicate_id", info[2]["warnings"])

    def test_whitespace_around_id_is_warned_not_changed(self):
        rows = self.rows([("x ", "CCO")])
        info = sp.classify_rows(rows)
        self.assertIn("id_has_surrounding_whitespace", info[0]["warnings"])
        self.assertEqual(rows[0].id, "x ")

    def test_synthetic_input_has_the_designed_edge_cases(self):
        rows = sp.read_input_rows(SYNTHETIC, "sample_id", "smiles")
        info = sp.classify_rows(rows)
        counts = sp.status_counts(info)
        self.assertEqual(counts[sp.STATUS_MISSING_ID], 1)
        self.assertEqual(counts[sp.STATUS_EMPTY_SMILES], 1)
        self.assertEqual(sp.duplicate_ids(rows), {"00042": [0, 8]})
        ids = [r.id for r in rows]
        for literal in ("00042", "1E5", "NA", "nan", "id with, comma", "trail-id ", "0x1F"):
            self.assertIn(literal, ids)


class ProtectedGuardTests(unittest.TestCase):
    def test_protected_file_hash_is_refused_and_others_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = write(tmp, "p.json", json.dumps({"entries": [
                {"sha256": "a" * 64, "path": "x.csv", "reason": "protected test"}]}))
            table = sp.load_protected_hashes(listing, sp.sha256_file(listing))
            with self.assertRaises(sp.ProtectedInputError):
                sp.check_input_not_protected("a" * 64, table)
            sp.check_input_not_protected("b" * 64, table)
            with self.assertRaises(ValueError):
                sp.load_protected_hashes(listing, "0" * 64)

    def test_real_protected_list_is_well_formed_and_excludes_the_synthetic_input(self):
        table = sp.load_protected_hashes(PROTECTED, sp.sha256_file(PROTECTED))
        self.assertGreaterEqual(len(table), 10)
        sp.check_input_not_protected(sp.sha256_file(SYNTHETIC), table)

    def test_known_protected_hashes_are_present(self):
        table = sp.load_protected_hashes(PROTECTED, sp.sha256_file(PROTECTED))
        for digest in (
            "a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9",  # CYP blinded test
            "e8a8eed3e81e4fb6c443cba2200a89f855f769541ce786508f7efc2b4edb517c",  # PXR blinded test
            "4f61be8ffdec767a61981ab89e3f06d28850603c3fa4ad0715a2183d4261a706",  # screening file
        ):
            self.assertIn(digest, table)


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.rows = [sp.InputRow(i, i + 2, name, smiles) for i, (name, smiles) in enumerate(
            [("a", "CCO"), ("", "CCN"), ("c", "CCC"), ("d", "bad")])]
        self.info = sp.classify_rows(self.rows)
        self.info[3]["status"], self.info[3]["reason"] = sp.STATUS_STRUCTURE, "bad structure"

    def values(self):
        cyp = {0: [0.1, 0.2, 0.3, 0.4], 2: [1.1, 1.2, 1.3, 1.4]}
        pxr = {seed: {0: 1.0 + k, 2: 2.0 + k} for k, seed in enumerate(sp.PXR_SEEDS)}
        return cyp, pxr

    def test_one_row_per_input_in_order_with_empty_cells_for_invalid_rows(self):
        out = sp.assemble_output_rows(self.rows, self.info, *self.values())
        self.assertEqual([r["input_row"] for r in out], [0, 1, 2, 3])
        self.assertEqual([r["status"] for r in out], [
            sp.STATUS_OK, sp.STATUS_MISSING_ID, sp.STATUS_OK, sp.STATUS_STRUCTURE])
        for index in (1, 3):
            for column in sp.CYP_COLUMNS + sp.PXR_SEED_COLUMNS + (sp.PXR_MEAN_COLUMN,):
                self.assertEqual(out[index][column], "")
        self.assertEqual(out[2][sp.CYP_COLUMNS[3]], repr(1.4))

    def test_mean_is_the_arithmetic_mean_of_the_three_retained_seeds(self):
        out = sp.assemble_output_rows(self.rows, self.info, *self.values())
        seeds = [float(out[0][c]) for c in sp.PXR_SEED_COLUMNS]
        self.assertEqual(seeds, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(out[0][sp.PXR_MEAN_COLUMN]), np.mean(seeds), places=14)

    def test_mean_requires_all_three_values(self):
        self.assertIsNone(sp.mean3((1.0, None, 3.0)))
        self.assertIsNone(sp.mean3((1.0, 2.0)))
        self.assertEqual(sp.mean3((1.0, 2.0, 6.0)), 3.0)

    def test_ok_row_without_predictions_or_invalid_row_with_predictions_is_an_error(self):
        cyp, pxr = self.values()
        del cyp[2]
        with self.assertRaises(ValueError):
            sp.assemble_output_rows(self.rows, self.info, cyp, pxr)
        cyp, pxr = self.values()
        cyp[1] = [0.0, 0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            sp.assemble_output_rows(self.rows, self.info, cyp, pxr)

    def test_non_finite_prediction_is_refused(self):
        with self.assertRaises(ValueError):
            sp.fmt_float(float("nan"))

    def test_csv_round_trip_keeps_literal_ids_and_blank_cells(self):
        rows = [sp.InputRow(0, 2, "00042", "CCO"), sp.InputRow(1, 3, "id, with comma", "CCN"), sp.InputRow(2, 4, "", "C")]
        info = sp.classify_rows(rows)
        cyp = {0: [0.1, 0.2, 0.3, 0.4], 1: [0.5, 0.6, 0.7, 0.8]}
        pxr = {seed: {0: 1.0, 1: 2.0} for seed in sp.PXR_SEEDS}
        out = sp.assemble_output_rows(rows, info, cyp, pxr)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o.csv"
            sp.write_output_csv(path, out)
            import csv
            with open(path, newline="") as handle:
                back = list(csv.DictReader(handle))
            with self.assertRaises(FileExistsError):
                sp.write_output_csv(path, out)
        self.assertEqual([r["input_id"] for r in back], ["00042", "id, with comma", ""])
        self.assertEqual(back[2]["status"], sp.STATUS_MISSING_ID)
        self.assertEqual(back[2][sp.PXR_MEAN_COLUMN], "")
        self.assertEqual(tuple(back[0].keys()), sp.OUTPUT_COLUMNS)


class ManifestTests(unittest.TestCase):
    def parts(self, checkpoints):
        return dict(
            script_sha256="s", input={}, checkpoints=checkpoints, output_csv_sha256="o", cyp={}, pxr={},
            code_provenance={}, pins={}, environment={}, run={}, timing_seconds={}, pxr_transform_check=None)

    def four(self):
        names = ["cyp_r1_epoch_016"] + [f"pxr_stage1_seed{s}" for s in sp.PXR_SEEDS]
        return {name: {"path": "p", "sha256": "h"} for name in names}

    def test_manifest_names_exactly_four_checkpoints(self):
        manifest = sp.build_manifest(**self.parts(self.four()))
        self.assertEqual(len(manifest["checkpoints"]), 4)
        self.assertTrue(manifest["not_predictive_validation"])
        self.assertFalse(manifest["stage2_potency_checkpoints_used"])
        self.assertFalse(manifest["labels_used"])
        self.assertIn("not assumed", manifest["join_note"])

    def test_wrong_checkpoint_set_is_rejected(self):
        extra = self.four()
        extra["pxr_stage2_seed20261125"] = {"path": "p", "sha256": "h"}
        with self.assertRaises(ValueError):
            sp.build_manifest(**self.parts(extra))
        short = self.four()
        short.pop("cyp_r1_epoch_016")
        with self.assertRaises(ValueError):
            sp.build_manifest(**self.parts(short))

    def test_missing_provenance_field_is_rejected(self):
        parts = self.parts(self.four())
        del parts["pins"]
        with self.assertRaises(ValueError):
            sp.build_manifest(**parts)


class OutputDirectoryTests(unittest.TestCase):
    def test_existing_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            sp.new_output_dir(target)
            with self.assertRaises(FileExistsError):
                sp.new_output_dir(target)

    def test_stage2_checkpoint_paths_are_refused(self):
        pins = {"run_files": {"checkpoint": {"path": __file__, "sha256": sp.sha256_file(__file__)}},
                "expected": {"selected_epoch_zero_based": 16}}
        pxr = {"seeds": {str(s): {"run_dir": "/x/stage2/y", "best_checkpoint_sha256": "h"} for s in sp.PXR_SEEDS}}
        # run_dir contains "stage2" in its path parts before "stage1/best.ckpt" is appended
        with self.assertRaises(ValueError):
            sp.collect_checkpoint_records(pins, pxr)


class ModelBehaviourTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
            from chemprop import nn  # noqa: F401
        except Exception:
            self.skipTest("torch/chemprop unavailable")

    def build(self, n_tasks, with_unscale):
        from chemprop import models, nn
        from sklearn.preprocessing import StandardScaler

        transform = None
        if with_unscale:
            rng = np.random.default_rng(0)
            scaler = StandardScaler().fit(rng.normal(loc=[0.3, 0.6, 1.2][:n_tasks], scale=[0.4, 0.5, 0.6][:n_tasks], size=(50, n_tasks)))
            transform = nn.transforms.UnscaleTransform.from_standard_scaler(scaler)
        predictor = nn.RegressionFFN(n_tasks=n_tasks, input_dim=16, hidden_dim=8, n_layers=1, output_transform=transform)
        model = models.MPNN(
            message_passing=nn.BondMessagePassing(d_h=16, depth=2), agg=nn.MeanAggregation(),
            predictor=predictor, batch_norm=False)
        model.eval()
        return model

    SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCCl", "CNC"]

    def test_batch_size_does_not_change_values_or_order(self):
        model = self.build(3, True)
        one = sp.predict_in_batches(model, self.SMILES, None, 1, 3)
        three = sp.predict_in_batches(model, self.SMILES, None, 3, 3)
        big = sp.predict_in_batches(model, self.SMILES, None, 64, 3)
        np.testing.assert_allclose(one, three, atol=1e-6)
        np.testing.assert_allclose(one, big, atol=1e-6)
        self.assertEqual(one.shape, (7, 3))

    def test_inference_does_not_toggle_the_model_into_training_mode(self):
        model = self.build(3, True)
        sp.predict_in_batches(model, self.SMILES, None, 4, 3)
        self.assertFalse(model.training)
        self.assertFalse(model.predictor.output_transform.training)

    def test_inference_refuses_a_model_in_training_mode(self):
        model = self.build(3, True)
        model.train()
        with self.assertRaises(ValueError):
            sp.predict_in_batches(model, self.SMILES, None, 4, 3)

    def test_separate_transform_check_leaves_the_inference_model_untouched(self):
        model = self.build(3, True)
        before = sp.predict_in_batches(model, self.SMILES, None, 7, 3)
        result = sp.verify_pxr_transform_on_copy(model, self.SMILES, None, 1e-5)
        after = sp.predict_in_batches(model, self.SMILES, None, 7, 3)
        self.assertTrue(result["inference_model_untouched"])
        self.assertLess(result["max_abs_difference"], 1e-5)
        self.assertFalse(model.training)
        self.assertFalse(model.predictor.output_transform.training)
        np.testing.assert_array_equal(before, after)

    def test_transform_check_rejects_a_model_with_no_unscale_transform(self):
        model = self.build(3, False)
        with self.assertRaises(ValueError):
            sp.verify_pxr_transform_on_copy(model, self.SMILES, None, 1e-5)

    def test_native_identity_head_check(self):
        sp.assert_native_identity_head(self.build(4, False), 4)
        with self.assertRaises(ValueError):
            sp.assert_native_identity_head(self.build(3, True), 3)
        with self.assertRaises(ValueError):
            sp.assert_native_identity_head(self.build(3, False), 4)

    def test_featurization_error_reports_bad_smiles_and_passes_good_ones(self):
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        featurizer = SimpleMoleculeMolGraphFeaturizer()
        self.assertIsNone(sp.featurization_error("CCO", featurizer))
        self.assertIsNone(sp.featurization_error("C[Hg]Cl", featurizer))
        for bad in ("not_a_smiles", "C1CC"):
            self.assertIsNotNone(sp.featurization_error(bad, featurizer))

    def test_scaler_buffers_are_read_from_the_state_dict(self):
        model = self.build(3, True)
        mean, scale = sp.scaler_from_state_dict(model.state_dict())
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(scale.shape, (3,))
        with self.assertRaises(ValueError):
            sp.scaler_from_state_dict(self.build(3, False).state_dict())


class SourceInvariantTests(unittest.TestCase):
    def test_inference_code_never_toggles_training_mode_outside_the_copy_check(self):
        source = (REPO / "cyp/scripts/screen_predict.py").read_text()
        inference = source.split("def predict_in_batches", 1)[1].split("def verify_pxr_transform_on_copy", 1)[0]
        self.assertNotIn(".train(", inference)
        self.assertNotIn("on_validation_model_eval", source)

    def test_stage2_checkpoints_are_never_named_as_inputs(self):
        source = (REPO / "cyp/scripts/screen_predict.py").read_text()
        self.assertNotIn("stage2/best.ckpt", source)
        self.assertNotIn('"stage2" / "best.ckpt"', source)


if __name__ == "__main__":
    unittest.main()
