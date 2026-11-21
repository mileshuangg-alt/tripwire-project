"""Tests for t121_validation_inference.

The end-to-end tests use the real Chemprop model classes, loaders, scorer and
Lightning checkpoints on CPU with a tiny synthetic fold (synthetic names and
targets; no project data, no protected molecules). Run in the environment that
has chemprop and lightning (tripwire_cyp002).
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for directory in (
    REPO / "cyp/scripts",
    REPO / "qualification/t121_executor_source_20261117",
    REPO,
):
    sys.path.insert(0, str(directory))

import t121_validation_inference as inf

SMILES = [
    "CCO", "CCN", "CCC", "CCCC", "c1ccccc1", "c1ccccc1O", "c1ccccc1N",
    "CC(=O)O", "CC(=O)N", "CCOC", "CCCO", "CCCN", "c1ccncc1", "C1CCCCC1",
    "CC(C)C", "CC(C)O", "CCS", "CCCl", "c1ccccc1C", "C1CCOC1",
    "CCOC(=O)C", "CN(C)C", "c1ccc2ccccc2c1", "OCCO",
]
SAMPLED = {
    "batch_size": 8, "dropout": 0.0, "ffn_hidden_dim": 300, "ffn_num_layers": 1,
    "final_lr_ratio": 0.1, "init_lr_ratio": 0.1, "max_lr": 0.001,
    "message_hidden_dim": 300, "message_passing_depth": 2, "warmup_epochs": 1,
}
SEEDS = (20261101, 20261102, 20261103)


def sha(path):
    return inf.sha256_file(path)


class MetricTests(unittest.TestCase):
    def test_metrics_match_frozen_scorer_and_definitions(self):
        from st_rae import st_rae_metric

        rng = np.random.default_rng(0)
        y = rng.normal(5.5, 1.0, 200)
        low = y - rng.uniform(0.0, 0.8, 200)
        high = y + rng.uniform(0.0, 0.8, 200)
        pred = y + rng.normal(0.2, 0.7, 200)
        m = inf.isoform_metrics(y, pred, low, high)
        self.assertAlmostEqual(m["st_rae"], st_rae_metric(y, pred, low, high), places=12)
        self.assertAlmostEqual(m["plain_mae"], np.mean(np.abs(pred - y)), places=12)
        self.assertAlmostEqual(m["signed_bias"], np.mean(pred - y), places=12)
        self.assertLessEqual(m["soft_mae"], m["plain_mae"])

    def test_zero_error_inside_band_and_edge_distance_outside(self):
        err = inf.soft_error(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]),
                             np.array([1.5, 2.5, 0.25]))
        np.testing.assert_allclose(err, [0.0, 0.5, 0.75])

    def test_errors_are_kept_individually_not_averaged(self):
        # two predictions for the same compound value: +0.5 and -0.5 stay separate errors
        m = inf.isoform_metrics([1.0, 2.0], [1.5, 1.5], [0.9, 1.9], [1.1, 2.1])
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["signed_bias"], 0.0)
        self.assertAlmostEqual(m["plain_mae"], 0.5)
        self.assertAlmostEqual(m["soft_mae"], 0.4)

    def test_parse_cells(self):
        self.assertEqual(inf.parse_cells("0,3-5"), [0, 3, 4, 5])
        with self.assertRaises(ValueError):
            inf.parse_cells("1,1")


def build_fixture(root, *, include_test_in_fold=False):
    """Write a dataset, split, manifest, one trained CPU cell and its records."""
    import pandas as pd
    import torch
    from lightning import pytorch as pl

    import cyp007_cell_fit as fit
    import t121_weighted_training as weighted
    from cyp002_validation import build_validation_callbacks

    rng = np.random.default_rng(1)
    names = [f"SYN-{i:03d}" for i in range(len(SMILES))]
    frame = {"Molecule_Name": names, "SMILES": SMILES}
    for task in inf.TASKS:
        y = rng.normal(5.5, 1.0, len(names))
        y[rng.random(len(names)) < 0.25] = np.nan
        frame[task] = y
        width = rng.uniform(0.2, 0.9, len(names))
        frame[f"{task}_conf_low"] = y - width / 2
        frame[f"{task}_conf_high"] = y + width / 2
        frame[f"{task}_std"] = width / 4
    df = pd.DataFrame(frame)
    # every molecule needs at least one label in each partition for scoring
    for i in range(len(names)):
        if df.loc[i, list(inf.TASKS)].isna().all():
            df.loc[i, inf.TASKS[0]] = 5.0
            df.loc[i, f"{inf.TASKS[0]}_conf_low"] = 4.8
            df.loc[i, f"{inf.TASKS[0]}_conf_high"] = 5.2
    validation_names = names[:6]
    training_names = names[6:]
    if include_test_in_fold:
        split = ["train"] * len(names)
        split[0] = "test"
    else:
        split = ["validation" if n in validation_names else "train" for n in names]
    split_df = pd.DataFrame({
        "Molecule_Name": names, "cluster_id": range(len(names)), "split": split,
    })
    dataset_path = root / "dataset.csv"
    split_path = root / "split.csv"
    df.to_csv(dataset_path, index=False)
    split_df.to_csv(split_path, index=False)

    manifest = {
        "repetition": 1, "fold": 1, "training_seeds": list(SEEDS),
        "training_names": training_names, "validation_names": validation_names,
        "training_count": len(training_names), "validation_count": len(validation_names),
    }
    manifest_root = root / "manifests"
    manifest_root.mkdir()
    manifest_path = manifest_root / "rep_01_fold_01.json"
    manifest_path.write_text(json.dumps(manifest))

    run_root = root / "run"
    trial_dir, cell_dir = fit._cell_paths(run_root, 121, 0)
    cell_dir.mkdir(parents=True)
    (cell_dir / "cell_plan.json").write_text(json.dumps({
        "trial_index": 121, "cell_index": 0, "repetition": 1, "fold": 1, "seed": SEEDS[0],
    }))
    (trial_dir / "proposal.json").write_text(json.dumps({
        "sampled_config": SAMPLED, "config_name": "D007-T121",
    }))

    routed = fit.route_d007_fold(
        dataframe=df, original_split_dataframe=split_df,
        split_manifest=manifest, training_seed=SEEDS[0],
    ) if not include_test_in_fold else None
    if routed is None:
        return {"dataset": dataset_path, "split": split_path, "manifest_root": manifest_root,
                "run_root": run_root}
    prefit = weighted.build_updated_t121_pic50_prefit_state(
        training_dataframe=routed["training_dataframe"],
        validation_dataframe=routed["validation_dataframe"],
        sampled_config=SAMPLED, config_name="D007-T121", training_seed=SEEDS[0],
    )
    ckpt_dir = cell_dir / "fit"
    ckpt_dir.mkdir()
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, deterministic=True, max_epochs=3, logger=False,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=build_validation_callbacks(ckpt_dir, patience=5),
    )
    trainer.fit(prefit["model"], prefit["train_loader"], prefit["validation_loader"])
    history = prefit["model"].validation_history
    best = min(history, key=lambda item: item["macro_st_rae"])
    ckpt = ckpt_dir / "best.ckpt"
    (cell_dir / "result.json").write_text(json.dumps({
        "trial_index": 121, "cell_index": 0, "repetition": 1, "fold": 1, "seed": SEEDS[0],
        "best_validation": best, "macro_st_rae": best["macro_st_rae"],
        "checkpoint_sha256": sha(ckpt), "split_manifest_sha256": sha(manifest_path),
    }))
    (cell_dir / "COMPLETED.json").write_text(json.dumps({"status": "completed"}))
    return {"dataset": dataset_path, "split": split_path, "manifest_root": manifest_root,
            "run_root": run_root, "cell_dir": cell_dir, "ckpt": ckpt}


class EndToEndCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fx = build_fixture(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def call(self, output_dir, **overrides):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        kwargs = dict(
            run_root=self.fx["run_root"], trial_index=121, cell_index=0,
            dataset_path=self.fx["dataset"], original_split_path=self.fx["split"],
            split_manifest_root=self.fx["manifest_root"], output_dir=output_dir,
            tolerance=1e-6, accelerator="cpu", authorize_inference=True,
        )
        kwargs.update(overrides)
        return inf.infer_cell(**kwargs)

    def test_reproduces_original_scores_and_saves_identified_rows(self):
        with tempfile.TemporaryDirectory() as out:
            record = self.call(out)
            self.assertEqual(record["status"], "reproduced")
            self.assertLessEqual(record["comparison_frozen_scorer_path"]["max_abs_diff"], 1e-6)
            # independent recomputation from the saved per-compound rows
            self.assertLessEqual(
                record["comparison_recomputed_from_saved_rows"]["max_abs_diff"], 1e-6)
            import pandas as pd
            saved = pd.read_csv(Path(out) / "predictions_cell_000.csv")
            self.assertEqual(
                list(saved.columns),
                list(inf.PREDICTION_COLUMNS))
            self.assertTrue((saved.seed == SEEDS[0]).all())
            self.assertEqual(set(saved.isoform), set(inf.ISOFORMS))
            self.assertEqual(len(saved), record["prediction_rows"])
            self.assertTrue(set(saved.Molecule_Name) <= {f"SYN-{i:03d}" for i in range(6)})
            # labels in the file are the dataset's exact labels
            ds = pd.read_csv(self.fx["dataset"]).set_index("Molecule_Name")
            for _, r in saved.iterrows():
                self.assertEqual(r.y_true, ds.loc[r.Molecule_Name, f"{r.isoform}{inf.TASK_SUFFIX}"])
            # the training artifacts are untouched and the checkpoint hash is unchanged
            self.assertEqual(sha(self.fx["ckpt"]),
                             json.loads((self.fx["cell_dir"] / "result.json").read_text())["checkpoint_sha256"])

    def test_not_authorized_raises(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(RuntimeError):
                self.call(out, authorize_inference=False)

    def test_checkpoint_hash_mismatch_refused(self):
        result_path = self.fx["cell_dir"] / "result.json"
        original = result_path.read_text()
        try:
            payload = json.loads(original)
            payload["checkpoint_sha256"] = "0" * 64
            result_path.write_text(json.dumps(payload))
            with tempfile.TemporaryDirectory() as out:
                with self.assertRaises(RuntimeError):
                    self.call(out)
        finally:
            result_path.write_text(original)

    def test_mismatch_is_recorded_and_stops_without_patching(self):
        result_path = self.fx["cell_dir"] / "result.json"
        original = result_path.read_text()
        try:
            payload = json.loads(original)
            payload["best_validation"]["macro_st_rae"] += 0.01
            result_path.write_text(json.dumps(payload))
            with tempfile.TemporaryDirectory() as out:
                record = self.call(out)
                self.assertEqual(record["status"], "MISMATCH")
                self.assertGreater(record["comparison_frozen_scorer_path"]["macro_abs_diff"], 1e-3)
        finally:
            result_path.write_text(original)

    def test_run_refuses_existing_output_dir_and_stops_on_mismatch(self):
        args = SimpleNamespace(
            run_root=self.fx["run_root"], trial_index=121, cells="0",
            dataset=self.fx["dataset"], original_split=self.fx["split"],
            split_manifest_root=self.fx["manifest_root"], tolerance=1e-6,
            accelerator="cpu", authorize_inference=True,
        )
        with tempfile.TemporaryDirectory() as parent:
            args.output_dir = Path(parent) / "new_run"
            summary = inf.run(args)
            self.assertEqual(summary["cells"], [0])
            self.assertTrue((args.output_dir / "completion.json").is_file())
            with self.assertRaises(FileExistsError):
                inf.run(args)  # never reuses an output directory

            # a mismatch halts the loop and writes the stop record
            def mismatching(**kwargs):
                record = inf.infer_cell(**kwargs)
                record["status"] = "MISMATCH"
                record["comparison_frozen_scorer_path"]["max_abs_diff"] = 1.0
                return record
            args.output_dir = Path(parent) / "stopped"
            with self.assertRaises(SystemExit):
                inf.run(args, infer=mismatching)
            self.assertTrue((args.output_dir / "STOPPED_ON_MISMATCH.json").is_file())
            self.assertFalse((args.output_dir / "completion.json").exists())

    def test_protected_test_molecule_in_fold_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = build_fixture(Path(tmp), include_test_in_fold=True)
            import cyp007_cell_fit as fit
            import pandas as pd
            manifest = json.loads((fx["manifest_root"] / "rep_01_fold_01.json").read_text())
            with self.assertRaises(RuntimeError):
                fit.route_d007_fold(
                    dataframe=pd.read_csv(fx["dataset"]),
                    original_split_dataframe=pd.read_csv(fx["split"]),
                    split_manifest=manifest, training_seed=SEEDS[0],
                )


if __name__ == "__main__":
    unittest.main()
