"""Tests for pxr002_build_predict_frame.py and pxr002_predict_frame.py (Tripwire-CYP-011).

Each test is tied to a concrete failure risk: a mapping table that differs from the reviewed
one, duplicate or empty ids, the wrong mapping column used as the PXR id or SMILES, a
non-finite descriptor being imputed or dropped, an artifact that was edited after hashing, a
plan with a different variant or seed set, a mean over fewer than five seeds or mixed
variants, a failed validation gate still producing predictions, the primary and secondary
columns being swapped, and output overwrite. Fixtures are synthetic; model loading and
descriptor computation are replaced by fakes (no checkpoints, no project data). Run in the
repository root with the Chemprop environment (tripwire_cyp002).
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pxr.scripts import pxr002_build_predict_frame as bld  # noqa: E402
from pxr.scripts import pxr002_dataset as ds  # noqa: E402
from pxr.scripts import pxr002_predict_frame as pf  # noqa: E402

HEADER = ["OCNT_ID", "cyp_molecule_name", "pxr_row_id", "pxr_role", "pxr_eligibility_reason",
          "cyp_tiers", "cyp_smiles", "pxr_source_smiles", "pxr_smiles_source", "smiles_string_equal"]


def write_mapping(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for ocnt, pxr_id, role, smiles in rows:
            writer.writerow([ocnt, ocnt, pxr_id, role, "x", "train", "CCO", smiles, "TRAIN.csv", "True"])


def fake_descriptors(smiles, allow_nonfinite=False):
    base = np.array([[float(len(s)) + 0.001 * j for j in range(ds.N_DESCRIPTORS)] for s in smiles])
    return base


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.mapping = self.t / "map.csv"
        write_mapping(self.mapping, [("OCNT-1", "OADMET-9", "primary_validation", "CCO"),
                                     ("OCNT-2", "OCNT-2", "aux_only", "CCCC")])
        self.sha = ds.sha256(self.mapping)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pxr_row_id_and_pxr_smiles_columns_are_used_not_ocnt(self):
        molecules = bld.read_mapping_molecules(self.mapping, self.sha)
        self.assertEqual(molecules, [("OADMET-9", "CCO"), ("OCNT-2", "CCCC")])

    def test_hash_mismatch_duplicates_and_blanks_stop(self):
        with self.assertRaises(bld.BuildStop):
            bld.read_mapping_molecules(self.mapping, "0" * 64)
        write_mapping(self.mapping, [("A", "R1", "primary_validation", "CCO"),
                                     ("B", "R1", "primary_validation", "CCC")])
        with self.assertRaisesRegex(bld.BuildStop, "Duplicate"):
            bld.read_mapping_molecules(self.mapping, ds.sha256(self.mapping))
        write_mapping(self.mapping, [("A", "R1", "primary_validation", "")])
        with self.assertRaisesRegex(bld.BuildStop, "empty"):
            bld.read_mapping_molecules(self.mapping, ds.sha256(self.mapping))

    def test_non_finite_descriptor_stops_and_names_rows(self):
        def bad(smiles, allow_nonfinite=False):
            matrix = fake_descriptors(smiles)
            matrix[1, 3] = np.nan
            return matrix
        with self.assertRaisesRegex(bld.BuildStop, "OCNT-2"):
            bld.build(self.mapping, self.sha, self.t / "out", compute=bad)
        self.assertFalse((self.t / "out").exists())

    def test_artifact_is_hash_listed_label_free_and_not_overwritten(self):
        report = bld.build(self.mapping, self.sha, self.t / "out", compute=fake_descriptors)
        self.assertEqual(report["molecules"], 2)
        listed = ds.verify_artifact(self.t / "out")
        self.assertEqual(set(listed), {bld.PREDICT_FRAME, bld.PREDICT_DESCRIPTORS})
        frame = pd.read_csv(self.t / "out" / bld.PREDICT_FRAME, dtype=str)
        self.assertEqual(list(frame.columns), ["row_id", "SMILES"])
        with self.assertRaises(bld.BuildStop):
            bld.build(self.mapping, self.sha, self.t / "out", compute=fake_descriptors)


class PlanAndFrameTests(unittest.TestCase):
    def good_plan(self):
        return {"configuration": pf.CONFIGURATION, "variants": ["multitask", "aux_off"], "seeds": list(pf.SEEDS)}

    def test_plan_must_be_exactly_the_ten_approved_models(self):
        pf.check_plan(self.good_plan())
        for key, value in (("configuration", "other"), ("variants", ["multitask"]),
                           ("variants", ["multitask", "aux_off", "extra"]),
                           ("seeds", list(pf.SEEDS[:4])), ("seeds", list(pf.SEEDS) + [1])):
            plan = self.good_plan()
            plan[key] = value
            with self.assertRaises(pf.PredictStop):
                pf.check_plan(plan)

    def test_frame_loader_verifies_hash_pin_columns_and_descriptors(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            mapping = t / "m.csv"
            write_mapping(mapping, [("A", "R1", "primary_validation", "CCO")])
            bld.build(mapping, ds.sha256(mapping), t / "art", compute=fake_descriptors)
            sums = ds.sha256(t / "art" / "SHA256SUMS")
            frame, x, listed = pf.load_predict_frame(t / "art", sums)
            self.assertEqual(frame["row_id"].tolist(), ["R1"])
            self.assertEqual(x.shape, (1, ds.N_DESCRIPTORS))
            with self.assertRaises(pf.PredictStop):
                pf.load_predict_frame(t / "art", "0" * 64)
            (t / "art" / bld.PREDICT_FRAME).write_text("row_id,SMILES\nR1,CCC\n")
            with self.assertRaises(RuntimeError):  # SHA256SUMS no longer matches the edited frame
                pf.load_predict_frame(t / "art", sums)


class MeanTests(unittest.TestCase):
    def frame(self, drop=None, duplicate=False):
        rows = []
        for row in ("R1", "R2"):
            for variant, offset in (("multitask", 0.0), ("aux_off", 10.0)):
                for i, seed in enumerate(pf.SEEDS):
                    rows.append({"row_id": row, "variant": variant, "seed": seed,
                                 "prediction": offset + i + (0.5 if row == "R2" else 0.0)})
        df = pd.DataFrame(rows)
        if drop is not None:
            df = df.drop(index=drop)
        if duplicate:
            df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        return df

    def test_primary_is_multitask_secondary_is_aux_off(self):
        means = pf.build_means(self.frame())
        self.assertEqual(list(means.columns), ["row_id", pf.PRIMARY_COLUMN, pf.SECONDARY_COLUMN])
        r1, r2 = means.iloc[0], means.iloc[1]
        self.assertAlmostEqual(float(r1[pf.PRIMARY_COLUMN]), 2.0, places=12)
        self.assertAlmostEqual(float(r1[pf.SECONDARY_COLUMN]), 12.0, places=12)
        self.assertAlmostEqual(float(r2[pf.PRIMARY_COLUMN]), 2.5, places=12)

    def test_missing_or_duplicate_seed_predictions_stop(self):
        with self.assertRaises(pf.PredictStop):
            pf.build_means(self.frame(drop=3))
        with self.assertRaises(pf.PredictStop):
            pf.build_means(self.frame(duplicate=True))

    def test_non_finite_seed_value_stops(self):
        with self.assertRaises(pf.PredictStop):
            pf.five_seed_mean([1, 2, 3, 4, float("nan")])
        with self.assertRaises(pf.PredictStop):
            pf.five_seed_mean([1, 2, 3, 4])


class RunWithFakes(unittest.TestCase):
    """The run loop with fake model loading, prediction and artifact; no checkpoints."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.t = t
        mapping = t / "m.csv"
        write_mapping(mapping, [("A", "R1", "primary_validation", "CCO"), ("B", "R2", "aux_only", "CCCC")])
        bld.build(mapping, ds.sha256(mapping), t / "art", compute=fake_descriptors)
        root = t / "final"
        root.mkdir()
        (root / "final_plan.json").write_text(json.dumps(
            {"configuration": pf.CONFIGURATION, "variants": list(pf.VARIANTS), "seeds": list(pf.SEEDS)}))
        for variant in pf.VARIANTS:
            for seed in pf.SEEDS:
                run = root / f"{variant}_seed{seed}"
                run.mkdir()
                (run / "metrics.json").write_text(json.dumps({"best_primary_validation_mae": 0.5}))
                (run / "best.ckpt").write_bytes(f"{variant}{seed}".encode())
        self.args = SimpleNamespace(
            artifact_dir=str(t / "v2"), predict_dir=str(t / "art"),
            predict_sums_sha256=ds.sha256(t / "art" / "SHA256SUMS"), final_root=str(root),
            final_plan_sha256=ds.sha256(root / "final_plan.json"), out_dir=str(t / "out"), accelerator="cpu")
        self.validation = pd.DataFrame({"SMILES": ["CC", "CCC"], "pEC50": [4.0, 5.0]})
        self.val_x = np.zeros((2, ds.N_DESCRIPTORS))
        self.orig = ds.load_artifact if hasattr(ds, "load_artifact") else None
        ds.load_artifact = lambda directory: {"validation": self.validation, "val_x": self.val_x, "hashes": {"v": "h"}}
        self.gate_bias = 0.0

    def tearDown(self):
        ds.load_artifact = self.orig
        self.tmp.cleanup()

    def fakes(self):
        state = {"seed": None}

        def loader(ckpt, config, xd_dim):
            state["seed"] = int(Path(ckpt).read_bytes().decode().replace("multitask", "").replace("aux_off", ""))
            state["variant"] = "multitask" if b"multitask" in Path(ckpt).read_bytes() else "aux_off"
            return state

        def predictor(model, smiles, descriptors, accelerator):
            if len(smiles) == 2 and smiles == ["CC", "CCC"]:  # validation gate molecules
                return np.array([4.0, 5.0]) + 0.5 - 0.5 + self.gate_bias  # MAE == gate_bias
            offset = 0.0 if model["variant"] == "multitask" else 10.0
            return np.array([offset + (model["seed"] - pf.SEEDS[0]) + 0.01 * k for k in range(len(smiles))])

        return loader, predictor

    def test_successful_run_writes_means_and_manifest(self):
        self.gate_bias = 0.5
        loader, predictor = self.fakes()
        # gate: recorded 0.5, replicated MAE = 0.5 -> passes
        status = pf.run_predict_frame(self.args, loader=loader, predictor=predictor)
        self.assertEqual(status, 0)
        out = Path(self.args.out_dir)
        means = pd.read_csv(out / pf.OUT_MEANS)
        self.assertAlmostEqual(means.loc[0, pf.PRIMARY_COLUMN], 2.0, places=12)
        self.assertAlmostEqual(means.loc[0, pf.SECONDARY_COLUMN], 12.0, places=12)
        self.assertEqual(len(pd.read_csv(out / pf.OUT_SEEDS)), 2 * 10)
        manifest = json.loads((out / pf.OUT_MANIFEST).read_text())
        self.assertFalse(manifest["training_performed"])
        self.assertEqual(len(manifest["gates"]), 10)
        self.assertEqual(len(manifest["checkpoint_sha256"]), 10)

    def test_failed_gate_stops_before_any_prediction_output(self):
        self.gate_bias = 0.9  # replicated MAE 0.9 vs recorded 0.5 -> fails
        loader, predictor = self.fakes()
        status = pf.run_predict_frame(self.args, loader=loader, predictor=predictor)
        self.assertEqual(status, 1)
        out = Path(self.args.out_dir)
        self.assertTrue((out / "gate_failure.json").exists())
        self.assertFalse((out / pf.OUT_MEANS).exists())

    def test_plan_hash_mismatch_and_existing_output_stop(self):
        loader, predictor = self.fakes()
        self.args.final_plan_sha256 = "0" * 64
        with self.assertRaises(pf.PredictStop):
            pf.run_predict_frame(self.args, loader=loader, predictor=predictor)
        self.args.final_plan_sha256 = ds.sha256(Path(self.args.final_root) / "final_plan.json")
        Path(self.args.out_dir).mkdir()
        with self.assertRaises(FileExistsError):
            pf.run_predict_frame(self.args, loader=loader, predictor=predictor)


class TimingOnly(RunWithFakes):
    """--timing-only-checkpoint: one model, a timing record only, gate still enforced."""

    def test_timing_target_must_be_one_approved_model(self):
        self.assertEqual(pf.parse_timing_target("aux_off:20261123"), ("aux_off", 20261123))
        for bad in ("multitask", "multitask:1", "other:20261120", "multitask:abc", ":20261120"):
            with self.assertRaises(pf.PredictStop):
                pf.parse_timing_target(bad)

    def test_writes_only_a_timing_record(self):
        self.gate_bias = 0.5
        self.args.timing_only_checkpoint = "multitask:20261120"
        loader, predictor = self.fakes()
        self.assertEqual(pf.run_predict_frame(self.args, loader=loader, predictor=predictor), 0)
        out = Path(self.args.out_dir)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["pxr_potency_timing.json"])
        record = json.loads((out / "pxr_potency_timing.json").read_text())
        self.assertEqual(record["checkpoint"], "multitask_seed20261120")
        self.assertFalse(record["means_written"])
        self.assertEqual(record["input_molecules_predicted"], 2)
        self.assertAlmostEqual(record["extrapolated_10_model_seconds"],
                               record["seconds"]["member_total_including_load_gate_and_prediction"] * 10, places=9)

    def test_gate_is_still_enforced(self):
        self.gate_bias = 0.9
        self.args.timing_only_checkpoint = "aux_off:20261124"
        loader, predictor = self.fakes()
        self.assertEqual(pf.run_predict_frame(self.args, loader=loader, predictor=predictor), 1)
        out = Path(self.args.out_dir)
        self.assertTrue((out / "gate_failure.json").exists())
        self.assertFalse((out / "pxr_potency_timing.json").exists())


class SourceInvariants(unittest.TestCase):
    def test_no_training_calls_or_label_reads_for_the_frame(self):
        for name in ("pxr002_predict_frame.py", "pxr002_build_predict_frame.py"):
            text = (REPO / "pxr/scripts" / name).read_text()
            code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            for forbidden in ("trainer.fit", ".fit(", "configure_optimizers", "backward("):
                self.assertNotIn(forbidden, code, f"{name}: {forbidden}")
        builder = (REPO / "pxr/scripts/pxr002_build_predict_frame.py").read_text()
        self.assertNotIn("pEC50", builder.split('"""', 2)[2])


if __name__ == "__main__":
    unittest.main()
