"""Tests for t121_potency_predict.py (Tripwire-CYP-011, D028).

Each test is tied to a concrete failure risk: a member silently missing, duplicated or
substituted; a checkpoint whose hash differs from its record; an ensemble that is not the
equal-weight 150-member mean (or uses the wrong SD denominator); a native-unit path that
does not reproduce saved predictions and yet is aggregated; invalid input rows being
dropped or mis-aligned with predictions; a protected exact file or a hash-mismatched input
being accepted; output overwrite; and any training call entering the script. Fixtures are
synthetic (no project data, no checkpoints, no Chemprop model). The run test replaces only
the model-dependent functions. Run in tripwire_cyp002 (chemprop is imported for the
featurizer check).
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import itertools
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cyp/scripts"))

import t121_potency_predict as pp  # noqa: E402
import screen_predict as sp  # noqa: E402

TRIAL = 121


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities():
    return list(itertools.product(pp.REPETITIONS, pp.FOLDS, pp.SEEDS))


def build_run_root(root, mutate=None):
    """150 synthetic completed cells. mutate(cell_index, files) may corrupt one."""
    for cell, (rep, fold, seed) in enumerate(identities()):
        cell_dir = Path(root) / "trials" / f"trial_{TRIAL:03d}" / "cells" / f"cell_{cell:03d}"
        (cell_dir / "fit").mkdir(parents=True)
        ckpt = cell_dir / "fit" / "best.ckpt"
        ckpt.write_bytes(f"ckpt-{cell}".encode())
        (cell_dir / "COMPLETED.json").write_text("{}")
        plan = {"trial_index": TRIAL, "cell_index": cell, "repetition": rep, "fold": fold, "seed": seed}
        result = {"trial_index": TRIAL, "cell_index": cell, "checkpoint_sha256": sha(ckpt),
                  "split_manifest_sha256": "m" * 64}
        (cell_dir / "cell_plan.json").write_text(json.dumps(plan))
        (cell_dir / "result.json").write_text(json.dumps(result))
        if mutate:
            mutate(cell, cell_dir, plan, result)


class MemberVerification(unittest.TestCase):
    def test_good_pool_passes_with_exact_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_run_root(tmp)
            members = pp.verify_members(tmp, TRIAL, list(range(150)))
            self.assertEqual(len(members), 150)
            self.assertEqual({(m["repetition"], m["fold"], m["seed"]) for m in members},
                             set(identities()))

    def test_cell_count_must_be_exactly_150(self):
        with self.assertRaises(pp.PotencyStop):
            pp.expand_member_cells("0-148")
        with self.assertRaises(pp.PotencyStop):
            pp.expand_member_cells("0-150")
        self.assertEqual(len(pp.expand_member_cells("0-149")), 150)

    def test_checkpoint_hash_mismatch_stops(self):
        def mutate(cell, cell_dir, plan, result):
            if cell == 7:
                (cell_dir / "fit" / "best.ckpt").write_bytes(b"tampered")
        with tempfile.TemporaryDirectory() as tmp:
            build_run_root(tmp, mutate)
            with self.assertRaisesRegex(pp.PotencyStop, "hash"):
                pp.verify_members(tmp, TRIAL, list(range(150)))

    def test_duplicate_identity_stops_instead_of_substituting(self):
        def mutate(cell, cell_dir, plan, result):
            if cell == 9:
                plan.update(repetition=1, fold=1, seed=pp.SEEDS[0])
                (cell_dir / "cell_plan.json").write_text(json.dumps(plan))
        with tempfile.TemporaryDirectory() as tmp:
            build_run_root(tmp, mutate)
            with self.assertRaisesRegex(pp.PotencyStop, "Duplicate"):
                pp.verify_members(tmp, TRIAL, list(range(150)))

    def test_incomplete_cell_and_missing_checkpoint_stop(self):
        def mark(cell, cell_dir, plan, result):
            if cell == 3:
                (cell_dir / "INCOMPLETE.json").write_text("{}")
        def drop(cell, cell_dir, plan, result):
            if cell == 4:
                (cell_dir / "fit" / "best.ckpt").unlink()
        for mutate, text in ((mark, "INCOMPLETE"), (drop, "best.ckpt")):
            with tempfile.TemporaryDirectory() as tmp:
                build_run_root(tmp, mutate)
                with self.assertRaisesRegex(pp.PotencyStop, text):
                    pp.verify_members(tmp, TRIAL, list(range(150)))

    def test_plan_identity_mismatch_stops(self):
        def mutate(cell, cell_dir, plan, result):
            if cell == 2:
                plan["cell_index"] = 99
                (cell_dir / "cell_plan.json").write_text(json.dumps(plan))
        with tempfile.TemporaryDirectory() as tmp:
            build_run_root(tmp, mutate)
            with self.assertRaisesRegex(pp.PotencyStop, "frozen plan"):
                pp.verify_members(tmp, TRIAL, list(range(150)))


class EnsembleArithmetic(unittest.TestCase):
    def test_mean_is_equal_weight_and_sd_uses_ddof_one(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(150, 5, 4))
        mean, sd = pp.ensemble_mean_sd(matrix)
        np.testing.assert_allclose(mean, matrix.sum(axis=0) / 150, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sd, matrix.std(axis=0, ddof=1), rtol=0, atol=1e-12)
        self.assertFalse(np.allclose(sd, matrix.std(axis=0, ddof=0)))

    def test_wrong_member_count_or_nonfinite_stops(self):
        with self.assertRaises(pp.PotencyStop):
            pp.ensemble_mean_sd(np.zeros((149, 3, 4)))
        with self.assertRaises(pp.PotencyStop):
            pp.ensemble_mean_sd(np.zeros((150, 3, 3)))
        bad = np.zeros((150, 3, 4))
        bad[5, 1, 2] = np.nan
        with self.assertRaises(pp.PotencyStop):
            pp.ensemble_mean_sd(bad)


class ReferenceGate(unittest.TestCase):
    def write_reference(self, directory, cell, rows):
        path = Path(directory) / f"predictions_cell_{cell:03d}.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trial_index", "cell_index", "repetition", "fold", "seed", "isoform",
                             "Molecule_Name", "y_true", "y_pred", "conf_low", "conf_high"])
            for name, isoform, value in rows:
                writer.writerow([121, cell, 1, 1, 20261101, isoform, name, 5.0, repr(value), 4.0, 6.0])

    def test_isoform_columns_are_matched_not_assumed(self):
        native = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        ref = {("A", "CYP2D6"): 3.0, ("B", "CYP3A4"): 8.0}
        out = pp.compare_to_reference(["A", "B"], native, ref, 1e-6)
        self.assertTrue(out["within_tolerance"])
        self.assertEqual(out["n_compared"], 2)
        wrong = {("A", "CYP3A4"): 3.0}
        self.assertFalse(pp.compare_to_reference(["A", "B"], native, wrong, 1e-6)["within_tolerance"])

    def test_tolerance_boundary_and_unknown_molecule(self):
        native = np.array([[1.0, 2.0, 3.0, 4.0]])
        self.assertFalse(pp.compare_to_reference(["A"], native, {("A", "CYP1A2"): 1.001}, 1e-4)["within_tolerance"])
        with self.assertRaises(pp.PotencyStop):
            pp.compare_to_reference(["A"], native, {("Z", "CYP1A2"): 1.0}, 1e-4)

    def test_reference_duplicates_missing_and_wrong_cell_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_reference(tmp, 0, [("A", "CYP1A2", 1.0), ("A", "CYP1A2", 1.0)])
            with self.assertRaisesRegex(pp.PotencyStop, "duplicate"):
                pp.load_reference(tmp, 0)
            with self.assertRaisesRegex(pp.PotencyStop, "missing"):
                pp.load_reference(tmp, 1)
            self.write_reference(tmp, 2, [("A", "CYP1A2", 1.0)])
            self.assertEqual(pp.load_reference(tmp, 2), {("A", "CYP1A2"): 1.0})
            Path(tmp, "predictions_cell_003.csv").write_text(
                "trial_index,cell_index,repetition,fold,seed,isoform,Molecule_Name,y_true,y_pred,conf_low,conf_high\n"
                "121,4,1,1,20261101,CYP1A2,A,5,1.0,4,6\n")
            with self.assertRaisesRegex(pp.PotencyStop, "different cell"):
                pp.load_reference(tmp, 3)


class RowAssembly(unittest.TestCase):
    def test_values_follow_ok_rows_and_invalid_rows_are_kept_blank(self):
        rows = [sp.InputRow(0, 2, "A", "CCO"), sp.InputRow(1, 3, "", "CCN"),
                sp.InputRow(2, 4, "C", "C(C"), sp.InputRow(3, 5, "D", "CCC")]
        classification = sp.classify_rows(rows)
        mean = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        sd = mean / 10
        records = pp.ensemble_rows(rows, classification, {2: "bad structure"}, mean, sd)
        self.assertEqual([r["status"] for r in records],
                         ["ok", "invalid_missing_id", "invalid_structure", "ok"])
        self.assertEqual(records[0][pp.MEAN_COLUMNS[0]], "1.0")
        self.assertEqual(records[3][pp.MEAN_COLUMNS[0]], "5.0")
        self.assertEqual(records[3][pp.SD_COLUMNS[3]], "0.8")
        for blank in (1, 2):
            self.assertEqual(records[blank][pp.MEAN_COLUMNS[0]], "")

    def test_count_mismatch_stops(self):
        rows = [sp.InputRow(0, 2, "A", "CCO")]
        with self.assertRaises(pp.PotencyStop):
            pp.ensemble_rows(rows, sp.classify_rows(rows), {}, np.zeros((2, 4)), np.zeros((2, 4)))


def fake_native(cell, smiles):
    base = np.array([[len(s) + 0.1 * cell + 0.01 * j for j in range(4)] for s in smiles])
    return base


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.run_root = t / "cells_run"
        build_run_root(self.run_root)
        self.dataset = t / "dataset.csv"
        self.dataset.write_text("a\n1\n")
        self.split = t / "split.csv"
        self.split.write_text("b\n2\n")
        self.input = t / "input.csv"
        self.input.write_text("id,smi\nM1,CCO\nM2,\nM3,C(C\nM4,CCCC\n")
        self.protected = t / "protected.json"
        self.protected.write_text(json.dumps({"entries": [
            {"path": "x", "sha256": "f" * 64, "reason": "test"}]}))
        self.reference = t / "reference"
        self.reference.mkdir()
        self.validation = pd.DataFrame({"Molecule_Name": ["V1", "V2"], "SMILES": ["CC", "CCC"]})
        for cell in range(150):
            native = fake_native(cell, self.validation["SMILES"])
            with open(self.reference / f"predictions_cell_{cell:03d}.csv", "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["trial_index", "cell_index", "repetition", "fold", "seed", "isoform",
                                 "Molecule_Name", "y_true", "y_pred", "conf_low", "conf_high"])
                for i, name in enumerate(self.validation["Molecule_Name"]):
                    for j, iso in enumerate(pp.ISOFORMS):
                        writer.writerow([121, cell, 1, 1, 1, iso, name, 5, repr(float(native[i, j])), 4, 6])
        self.args = SimpleNamespace(
            authorize_inference=True, run_root=str(self.run_root), trial_index=TRIAL,
            dataset=str(self.dataset), dataset_sha256=sha(self.dataset),
            original_split=str(self.split), original_split_sha256=sha(self.split),
            split_manifest_root=str(t), executor_source_sha256="e" * 64,
            reference_dir=str(self.reference), reference_tolerance=1e-9,
            protected_hashes=str(self.protected), protected_hashes_sha256=sha(self.protected),
            input=str(self.input), input_sha256=sha(self.input), id_column="id",
            smiles_column="smi", out_dir=str(t / "out"), cells="0-149", batch_size=8,
            accelerator="cpu")
        self.orig = (pp.predict_native, pp.load_ops)
        self.current = {}
        def fake_predict(model, scaler, smiles, batch_size):
            return fake_native(self.current["cell"], smiles)
        pp.predict_native = fake_predict
        pp.load_ops = lambda: {}
        def builder(ops, member, args, root):
            self.current["cell"] = member["cell_index"]
            return object(), object(), self.validation
        self.builder = builder

    def tearDown(self):
        pp.predict_native, pp.load_ops = self.orig
        self.tmp.cleanup()

    def test_full_run_writes_exact_mean_and_manifest(self):
        manifest = pp.run(self.args, ops={}, builder=self.builder)
        out = Path(self.args.out_dir)
        records = list(csv.DictReader(open(out / pp.OUT_ENSEMBLE)))
        self.assertEqual([r["status"] for r in records],
                         ["ok", "invalid_empty_smiles", "invalid_structure", "ok"])
        expected = np.mean([fake_native(c, ["CCO", "CCCC"]) for c in range(150)], axis=0)
        self.assertAlmostEqual(float(records[0][pp.MEAN_COLUMNS[0]]), expected[0, 0], places=12)
        self.assertAlmostEqual(float(records[3][pp.MEAN_COLUMNS[3]]), expected[1, 3], places=12)
        members = list(csv.DictReader(open(out / pp.OUT_MEMBERS)))
        self.assertEqual(len(members), 150 * 2)
        self.assertEqual(manifest["n_members"], 150)
        self.assertEqual(len(manifest["reference_gates"]), 150)
        self.assertFalse(manifest["training_performed"])
        self.assertIn("not held-out", manifest["deployment_note"])
        self.assertEqual(manifest["outputs"][pp.OUT_ENSEMBLE], sha(out / pp.OUT_ENSEMBLE))

    def test_reference_mismatch_stops_and_writes_no_ensemble(self):
        path = self.reference / "predictions_cell_005.csv"
        text = path.read_text().replace("\n121,5,1,1,1,CYP1A2,V1,5,", "\n121,5,1,1,1,CYP1A2,V1,5,9", 1)
        path.write_text(text)
        with self.assertRaises(pp.PotencyStop):
            pp.run(self.args, ops={}, builder=self.builder)
        out = Path(self.args.out_dir)
        self.assertTrue((out / "STOPPED_ON_REFERENCE_MISMATCH.json").exists())
        self.assertFalse((out / pp.OUT_ENSEMBLE).exists())

    def test_guards_stop_before_any_output_or_model(self):
        self.args.authorize_inference = False
        with self.assertRaises(pp.PotencyStop):
            pp.run(self.args, ops={}, builder=self.builder)
        self.assertFalse(Path(self.args.out_dir).exists())
        self.args.authorize_inference = True
        self.args.input_sha256 = "0" * 64
        with self.assertRaisesRegex(pp.PotencyStop, "input hash mismatch"):
            pp.run(self.args, ops={}, builder=self.builder)

    def test_protected_exact_file_is_refused(self):
        self.protected.write_text(json.dumps({"entries": [
            {"path": "p", "sha256": sha(self.input), "reason": "protected"}]}))
        self.args.protected_hashes_sha256 = sha(self.protected)
        with self.assertRaises(sp.ProtectedInputError):
            pp.run(self.args, ops={}, builder=self.builder)

    def test_existing_output_directory_is_refused(self):
        Path(self.args.out_dir).mkdir()
        with self.assertRaises(FileExistsError):
            pp.run(self.args, ops={}, builder=self.builder)


class TimingOnly(EndToEnd):
    """--timing-only-cell: one member end to end, a timing record only, gate still enforced."""

    def test_writes_only_a_timing_record_and_no_aggregate_files(self):
        self.args.timing_only_cell = 3
        record = pp.run(self.args, ops={}, builder=self.builder)
        out = Path(self.args.out_dir)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["cyp_potency_timing.json"])
        self.assertEqual(record["mode"], "timing_only")
        self.assertEqual(record["cell_index"], 3)
        self.assertFalse(record["ensemble_files_written"])
        self.assertEqual(record["input_molecules_predicted"], 2)
        self.assertAlmostEqual(record["extrapolated_150_member_seconds"],
                               record["seconds"]["member_total"] * 150, places=9)
        self.assertTrue(record["reference_gate"]["within_tolerance"])

    def test_gate_is_still_enforced_for_the_timed_member(self):
        self.args.timing_only_cell = 5
        path = self.reference / "predictions_cell_005.csv"
        path.write_text(path.read_text().replace("\n121,5,1,1,1,CYP1A2,V1,5,", "\n121,5,1,1,1,CYP1A2,V1,5,9", 1))
        with self.assertRaises(pp.PotencyStop):
            pp.run(self.args, ops={}, builder=self.builder)
        out = Path(self.args.out_dir)
        self.assertTrue((out / "STOPPED_ON_REFERENCE_MISMATCH.json").exists())
        self.assertFalse((out / "cyp_potency_timing.json").exists())

    def test_timed_cell_must_be_among_the_150_and_hash_checked(self):
        self.args.timing_only_cell = 150
        with self.assertRaises(pp.PotencyStop):
            pp.run(self.args, ops={}, builder=self.builder)
        self.args.out_dir = str(Path(self.args.out_dir).parent / "out2")
        self.args.timing_only_cell = 2
        (Path(self.args.run_root) / "trials" / f"trial_{TRIAL:03d}" / "cells" / "cell_002" / "fit" / "best.ckpt").write_bytes(b"x")
        with self.assertRaisesRegex(pp.PotencyStop, "hash"):
            pp.run(self.args, ops={}, builder=self.builder)

    def test_cells_argument_must_still_name_exactly_150(self):
        self.args.timing_only_cell = 3
        self.args.cells = "0-148"
        with self.assertRaises(pp.PotencyStop):
            pp.run(self.args, ops={}, builder=self.builder)


class RealModelSeam(unittest.TestCase):
    """predict_native against a tiny REAL Chemprop model with a REAL UnscaleTransform.

    Regression for the CoreHPC reference-gate failure (max difference 5.07): the output
    transform already unscales in eval mode, so unscaling again doubled the scaling. The
    fake-based tests never reached this path.
    """

    SMILES = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCC"]

    def build(self, with_transform=True, scaler_shift=0.0):
        import torch
        from chemprop import models, nn
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(0)
        data = np.random.default_rng(0).normal(loc=5.0, scale=2.0, size=(60, 4))
        scaler = StandardScaler().fit(data)
        transform = nn.transforms.UnscaleTransform.from_standard_scaler(scaler) if with_transform else None
        predictor = nn.RegressionFFN(n_tasks=4, input_dim=16, hidden_dim=16, n_layers=1,
                                     output_transform=transform)
        model = models.MPNN(nn.BondMessagePassing(d_h=16, depth=2), nn.NormAggregation(), predictor,
                            batch_norm=False)
        if scaler_shift:
            scaler.mean_ = scaler.mean_ + scaler_shift
        return model.eval(), scaler

    def reference_forward(self, model):
        import torch
        from chemprop import data

        points = [data.MoleculeDatapoint.from_smi(s) for s in self.SMILES]
        loader = data.build_dataloader(
            data.MoleculeDataset(points), batch_size=2, num_workers=0, shuffle=False, drop_last=False)
        with torch.inference_mode():
            return np.concatenate(
                [model(b[0], b[1], b[2]).detach().numpy().astype(np.float64) for b in loader], axis=0)

    def test_matches_eval_forward_and_is_not_unscaled_twice(self):
        model, scaler = self.build()
        got = pp.predict_native(model, scaler, self.SMILES, 3)
        expected = self.reference_forward(model)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
        twice = expected * scaler.scale_ + scaler.mean_
        self.assertGreater(float(np.max(np.abs(twice - expected))), 1.0)
        self.assertFalse(np.allclose(got, twice, atol=1e-3))

    def test_batch_size_does_not_change_the_values_beyond_float32_noise(self):
        model, scaler = self.build()
        a = pp.predict_native(model, scaler, self.SMILES, 1)
        b = pp.predict_native(model, scaler, self.SMILES, 5)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)

    def test_guards_refuse_a_missing_transform_a_mismatched_scaler_and_train_mode(self):
        model, scaler = self.build(with_transform=False)
        with self.assertRaisesRegex(pp.PotencyStop, "not UnscaleTransform"):
            pp.predict_native(model, scaler, self.SMILES, 2)
        model, scaler = self.build(scaler_shift=0.5)
        with self.assertRaisesRegex(pp.PotencyStop, "differs from the cell scaler"):
            pp.predict_native(model, scaler, self.SMILES, 2)
        model, scaler = self.build()
        model.train()
        with self.assertRaisesRegex(pp.PotencyStop, "eval mode"):
            pp.predict_native(model, scaler, self.SMILES, 2)


class SourceInvariants(unittest.TestCase):
    def test_no_training_or_optimizer_calls(self):
        text = (REPO / "cyp/scripts/t121_potency_predict.py").read_text()
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        for forbidden in ("trainer.fit", ".fit(", "configure_optimizers", "backward(", ".train()",
                          "train_step", "scale_ +", "* scale"):
            self.assertNotIn(forbidden, code, forbidden)
        self.assertIn("inference_mode", code)
        self.assertIn("model.eval()", code)


if __name__ == "__main__":
    unittest.main()
