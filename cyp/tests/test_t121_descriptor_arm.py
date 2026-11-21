"""D021 descriptor arm: cache, builders, alignment, transfer, and the graph-only guard.

Uses the real Chemprop model classes, loaders and Lightning checkpoints on CPU
with tiny synthetic data (synthetic names and labels; no project data and no
protected molecules). Run in the environment that has chemprop and lightning
(tripwire_cyp002).
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import tempfile
import types
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for directory in (REPO / "cyp/scripts", REPO / "qualification/t121_executor_source_20261117", REPO):
    sys.path.insert(0, str(directory))

import t121_descriptor_cache as dc

SMILES = [
    "CCO", "CCN", "CCC", "CCCC", "c1ccccc1", "c1ccccc1O", "c1ccccc1N",
    "CC(=O)O", "CC(=O)N", "CCOC", "CCCO", "CCCN", "c1ccncc1", "C1CCCCC1",
    "CC(C)C", "CC(C)O", "CCS", "CCCl", "c1ccccc1C", "C1CCOC1",
    "CCOC(=O)C", "CN(C)C", "c1ccc2ccccc2c1", "OCCO",
]
TASKS = [f"{i}_pIC50_direct_inhibition" for i in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")]
SAMPLED = {
    "batch_size": 8, "dropout": 0.0, "ffn_hidden_dim": 300, "ffn_num_layers": 1,
    "final_lr_ratio": 0.1, "init_lr_ratio": 0.1, "max_lr": 0.001,
    "message_hidden_dim": 300, "message_passing_depth": 2, "warmup_epochs": 1,
}
SEED = 20261101
# SHA-256 of the graph-only pIC50 prefit (model state, loader order, X_d absent)
# and of the graph-only screening prefit, computed 2026-11-19 with the committed
# code before the D021 edits and reproduced with the edited code (descriptor_dim 0).
GOLDEN_PIC50 = "0ad000be7e61f9e7865b51e466494d6f408d3c8fcae8bdcf5b22e6032ebdfab4"
GOLDEN_SCREEN = "2a5116f8d05b303093d6f1f32709753461e9d1f227aa7e93a049fd4cfa2a2244"


def frames():
    import pandas as pd

    rng = np.random.default_rng(1)
    names = [f"SYN-{i:03d}" for i in range(len(SMILES))]
    frame = {"Molecule_Name": names, "SMILES": SMILES}
    for task in TASKS:
        y = rng.normal(5.5, 1.0, len(names))
        y[rng.random(len(names)) < 0.25] = np.nan
        width = rng.uniform(0.2, 0.9, len(names))
        frame[task] = y
        frame[task + "_conf_low"] = y - width / 2
        frame[task + "_conf_high"] = y + width / 2
    df = pd.DataFrame(frame)
    for i in range(len(names)):
        if df.loc[i, TASKS].isna().all():
            df.loc[i, TASKS[0]] = 5.0
            df.loc[i, TASKS[0] + "_conf_low"] = 4.8
            df.loc[i, TASKS[0] + "_conf_high"] = 5.2
    return df.iloc[6:].copy(), df.iloc[:6].copy()


def build_cache(root):
    sets = {"fixture": SMILES}
    manifest = dc.build_cache(sets, root / "cache", sources={"fixture": "synthetic"})
    return dc.DescriptorCache.load(
        root / "cache", expected_manifest_sha256=dc.sha256_file(root / "cache" / dc.MANIFEST_NAME)
    ), manifest


def state_fingerprint(state, *, screen=False):
    import torch  # noqa: F401

    digest = hashlib.sha256()
    state_dict = state["model"].state_dict()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    order = []
    for batch in state["train_loader"]:
        order.append(list(batch["molecule_ids"]))
        digest.update(np.asarray(batch["chemprop_batch"].X_d is None).tobytes())
        if screen:
            digest.update(batch["raw_observation_weights"].numpy().tobytes())
        else:
            digest.update(batch["chemprop_batch"].Y.numpy().tobytes())
    digest.update(json.dumps(order).encode())
    return digest.hexdigest()


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cache, cls.manifest = build_cache(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_196_columns_equal_the_registry_output_minus_the_four_positions(self):
        from chemprop.featurizers import MoleculeFeaturizerRegistry
        from rdkit import Chem

        featurizer = MoleculeFeaturizerRegistry[dc.FEATURIZER_NAME]()
        self.assertEqual(self.cache.dim, 196)
        self.assertEqual(dc.DROPPED_POSITIONS, (39, 41, 43, 45))
        for smiles in SMILES[:5]:
            raw = np.asarray(featurizer(Chem.MolFromSmiles(smiles)), dtype=np.float64)
            self.assertEqual(raw.shape, (200,))
            np.testing.assert_array_equal(
                self.cache.lookup(smiles), np.delete(raw, [39, 41, 43, 45])
            )
        self.assertGreaterEqual(self.manifest["value_min"], 0.0)
        self.assertLessEqual(self.manifest["value_max"], 1.0)
        self.assertEqual(self.manifest["molecules"], len(SMILES))

    def test_missing_smiles_raises_instead_of_computing(self):
        with self.assertRaisesRegex(KeyError, "not in the frozen descriptor cache"):
            self.cache.lookup("CCCCCCCCCC")

    def test_load_refuses_wrong_manifest_hash_and_tampered_npz(self):
        manifest_path = self.root / "cache" / dc.MANIFEST_NAME
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            dc.DescriptorCache.load(self.root / "cache", expected_manifest_sha256="0" * 64)
        with tempfile.TemporaryDirectory() as other:
            copy = Path(other) / "cache"
            copy.mkdir()
            (copy / dc.MANIFEST_NAME).write_bytes(manifest_path.read_bytes())
            data = bytearray((self.root / "cache" / dc.NPZ_NAME).read_bytes())
            data[-20] ^= 0xFF
            (copy / dc.NPZ_NAME).write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "npz hash"):
                dc.DescriptorCache.load(copy, expected_manifest_sha256=dc.sha256_file(manifest_path))

    def test_output_directory_is_never_reused(self):
        with self.assertRaises(FileExistsError):
            dc.build_cache({"fixture": SMILES[:2]}, self.root / "cache", sources={})

    def test_non_finite_after_the_drop_raises_but_dropped_positions_may_be_non_finite(self):
        def featurizer(molecule):
            vector = np.full(200, 0.5)
            vector[[39, 41, 43, 45]] = np.nan  # the known metal-compound failure, dropped
            return vector

        matrix = dc.compute_matrix(["CCO"], featurizer=featurizer)
        self.assertEqual(matrix.shape, (1, 196))
        self.assertTrue(np.isfinite(matrix).all())

        def bad(molecule):
            vector = np.full(200, 0.5)
            vector[7] = np.inf
            return vector

        with self.assertRaisesRegex(ValueError, "no imputation, no exclusion"):
            dc.compute_matrix(["CCO"], featurizer=bad)

    def test_attach_columns_are_named_in_position_order_and_aligned_by_smiles(self):
        train, _ = frames()
        frame = dc.attach_descriptor_columns(train, self.cache)
        columns = [c for c in frame.columns if c.startswith("xd__")]
        self.assertEqual(columns, sorted(columns))
        self.assertEqual(len(columns), 196)
        for index in (0, 5, 17):
            np.testing.assert_array_equal(
                frame.iloc[index][columns].to_numpy(dtype=np.float64),
                self.cache.lookup(frame.iloc[index]["SMILES"]),
            )
        with self.assertRaises(ValueError):
            dc.attach_descriptor_columns(frame, self.cache)  # already has descriptor columns


class GraphOnlyGuardTests(unittest.TestCase):
    """D021 gate: the graph-only (arm R) construction is unchanged."""

    def test_pic50_prefit_is_bit_identical_to_the_pre_edit_golden(self):
        import t121_weighted_training as weighted

        train, validation = frames()
        for kwargs in ({}, {"descriptor_dim": 0}):
            state = weighted.build_updated_t121_pic50_prefit_state(
                training_dataframe=train, validation_dataframe=validation,
                sampled_config=SAMPLED, config_name="D007-T121", training_seed=SEED, **kwargs,
            )
            self.assertEqual(state_fingerprint(state), GOLDEN_PIC50)

    def test_screening_prefit_is_bit_identical_to_the_pre_edit_golden(self):
        import t121_weighted_training as weighted

        with stub_screen_loader():
            state = weighted.build_t121_screen_prefit_state(
                screen_path="x", identity_path="x", similarity_path="x", contract_path="x",
                sampled_config={**SAMPLED, "batch_size": 4}, config_name="D007-T121",
                training_seed=SEED,
            )
        self.assertEqual(state_fingerprint(state, screen=True), GOLDEN_SCREEN)

    def test_descriptor_columns_in_a_graph_only_build_are_refused(self):
        import t121_weighted_training as weighted

        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = build_cache(Path(tmp))
            train, validation = frames()
            train = dc.attach_descriptor_columns(train, cache)
            validation = dc.attach_descriptor_columns(validation, cache)
            with self.assertRaisesRegex(ValueError, "descriptor width 196 != descriptor_dim 0"):
                weighted.build_updated_t121_pic50_prefit_state(
                    training_dataframe=train, validation_dataframe=validation,
                    sampled_config=SAMPLED, config_name="D007-T121", training_seed=SEED,
                )


def screen_assays():
    ids = [f"SCR-{i:03d}" for i in range(12)]
    smiles = SMILES[:12]
    rng = np.random.default_rng(3)
    assays = []
    for e, enzyme in enumerate(("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")):
        pick = [i for i in range(len(ids)) if (i + e) % 3 != 0]
        assays.append(SimpleNamespace(
            assay_id=f"screen:{enzyme}",
            molecule_ids=tuple(ids[i] for i in pick), smiles=tuple(smiles[i] for i in pick),
            y=tuple(rng.normal(0, 1, len(pick)).tolist()),
            observation_weights=tuple(rng.uniform(0.5, 2.0, len(pick)).tolist()),
        ))
    return assays


class stub_screen_loader:
    """Replace the frozen adjudicated-screen loader with synthetic assays."""

    def __enter__(self):
        self.previous = sys.modules.get("actfound_model_side")
        stub = types.ModuleType("actfound_model_side")
        stub.load_adjudicated_screen_assays = lambda *a, **k: screen_assays()
        sys.modules["actfound_model_side"] = stub

    def __exit__(self, *exc):
        if self.previous is None:
            sys.modules.pop("actfound_model_side", None)
        else:
            sys.modules["actfound_model_side"] = self.previous


class DescriptorArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cache, _ = build_cache(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pic50_state(self):
        import t121_weighted_training as weighted

        train, validation = frames()
        train = dc.attach_descriptor_columns(train, self.cache)
        validation = dc.attach_descriptor_columns(validation, self.cache)
        state = weighted.build_updated_t121_pic50_prefit_state(
            training_dataframe=train, validation_dataframe=validation, sampled_config=SAMPLED,
            config_name="D007-T121", training_seed=SEED, descriptor_dim=196,
        )
        return state, train, validation

    def screen_state(self, cache):
        import t121_weighted_training as weighted

        with stub_screen_loader():
            return weighted.build_t121_screen_prefit_state(
                screen_path="x", identity_path="x", similarity_path="x", contract_path="x",
                sampled_config={**SAMPLED, "batch_size": 4}, config_name="D007-T121",
                training_seed=SEED, descriptor_cache=cache,
            )

    @staticmethod
    def first_layer_inputs(model):
        import torch

        return [m.in_features for m in model.predictor.modules() if isinstance(m, torch.nn.Linear)][0]

    def test_ffn_input_is_message_width_plus_196(self):
        state, _, _ = self.pic50_state()
        self.assertEqual(self.first_layer_inputs(state["model"]), 300 + 196)
        self.assertEqual(state["descriptor_dim"], 196)
        self.assertEqual(self.first_layer_inputs(self.screen_state(self.cache)["model"]), 300 + 196)

    def test_every_batch_row_carries_its_own_molecules_descriptors_under_shuffling(self):
        state, train, validation = self.pic50_state()
        smiles = dict(zip(train["Molecule_Name"], train["SMILES"]))
        seen = 0
        for _ in range(2):  # two shuffled epochs
            for batch in state["train_loader"]:
                x_d = batch["chemprop_batch"].X_d.numpy()
                for row, name in zip(x_d, batch["molecule_ids"]):
                    np.testing.assert_allclose(row, self.cache.lookup(smiles[name]), atol=1e-6)
                    seen += 1
        self.assertEqual(seen, 2 * len(train))
        # validation loader is unshuffled: rows follow the dataframe order
        rows = np.vstack([b[2].numpy() if isinstance(b, tuple) else b.X_d.numpy()
                          for b in state["validation_loader"]])
        expected = np.vstack([self.cache.lookup(s) for s in validation["SMILES"]])
        np.testing.assert_allclose(rows, expected, atol=1e-6)

    def test_screening_datapoints_carry_cached_descriptors_and_a_missing_smiles_raises(self):
        state = self.screen_state(self.cache)
        dataset = state["training_dataset"]
        assays = screen_assays()
        by_id = {i: s for a in assays for i, s in zip(a.molecule_ids, a.smiles)}
        for index, name in enumerate(dataset.molecule_ids):
            np.testing.assert_allclose(
                dataset.molecular_dataset[index].x_d, self.cache.lookup(by_id[name]), atol=1e-6
            )
        partial = dc.DescriptorCache(["CCO"], self.cache.lookup("CCO")[None, :])
        with self.assertRaisesRegex(ValueError, "Chemprop preparation failed"):
            self.screen_state(partial)

    def test_transfer_works_with_matching_descriptor_layers_and_refuses_a_graph_only_source(self):
        import t121_weighted_training as weighted

        pic50, _, _ = self.pic50_state()
        screen = self.screen_state(self.cache)
        report = weighted.transfer_screen_initialization(screen["model"], pic50["model"])
        self.assertTrue(report["transferred_keys"])
        keys = [k for k in report["transferred_keys"] if "predictor" in k and k.endswith("weight")]
        self.assertTrue(keys)  # the widened first FFN layer is among the transferred state
        after = pic50["model"].state_dict()
        source = screen["model"].state_dict()
        for key in keys:
            self.assertTrue(np.array_equal(after[key].numpy(), source[key].numpy()))
        graph_only = self.screen_state(None)
        with self.assertRaisesRegex(ValueError, "Incompatible shared-state shapes"):
            weighted.transfer_screen_initialization(graph_only["model"], pic50["model"])

    def test_fit_checkpoint_reload_and_graph_only_checkpoint_refusal_on_cpu(self):
        from lightning import pytorch as pl

        import cyp007_cell_fit as fit
        import t121_weighted_training as weighted
        from cyp002_validation import build_validation_callbacks

        state, train, validation = self.pic50_state()
        ckpt_dir = self.root / "fit_ckpt"
        ckpt_dir.mkdir()
        trainer = pl.Trainer(
            accelerator="cpu", devices=1, deterministic=True, max_epochs=3, logger=False,
            enable_progress_bar=False, enable_model_summary=False,
            callbacks=build_validation_callbacks(ckpt_dir, patience=5),
        )
        trainer.fit(state["model"], state["train_loader"], state["validation_loader"])
        history = state["model"].validation_history
        self.assertTrue(history and all(np.isfinite(h["macro_st_rae"]) for h in history))
        import torch

        checkpoint = torch.load(ckpt_dir / "best.ckpt", map_location="cpu", weights_only=False)
        fresh, _, _ = self.pic50_state()
        fresh["model"].load_state_dict(checkpoint["state_dict"], strict=True)
        # a graph-only model cannot load the descriptor checkpoint, and vice versa
        graph = weighted.build_updated_t121_pic50_prefit_state(
            training_dataframe=frames()[0], validation_dataframe=frames()[1],
            sampled_config=SAMPLED, config_name="D007-T121", training_seed=SEED,
        )
        with self.assertRaises(RuntimeError):
            graph["model"].load_state_dict(checkpoint["state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()
