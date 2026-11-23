"""D022 CheMeleon arm: frozen-file checks, encoder injection, learning-rate groups,
stage transfer and the graph-only guard.

Uses a randomly initialised stand-in checkpoint with the real CheMeleon shapes
(d_h 2048, depth 6): the pretrained tensors are not needed to test wiring, and
the real file's hash is checked against the frozen config separately. Synthetic
molecules and labels only. Run in tripwire_cyp002.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for directory in (REPO / "cyp/scripts", REPO / "qualification/t121_executor_source_20261117", REPO, REPO / "cyp/tests"):
    sys.path.insert(0, str(directory))

import t121_chemeleon as ch
from test_t121_descriptor_arm import SEED, frames, stub_screen_loader

FROZEN = REPO / "cyp/artifacts/d022_chemeleon/chemeleon_config.json"
REAL_WEIGHTS = REPO / "pxr/artifacts/pxr004_weights/chemeleon_mp.pt"
# trial_121 sampled configuration (artifacts/d009_closure_workspace/d009_t121_comparator_20261109/proposal.json)
SAMPLED = {
    "batch_size": 16, "dropout": 0.4, "ffn_hidden_dim": 2000, "ffn_num_layers": 1,
    "final_lr_ratio": 0.24427727346003839, "init_lr_ratio": 0.017323283256612185,
    "max_lr": 0.0008245400010426866, "message_hidden_dim": 600,
    "message_passing_depth": 4, "warmup_epochs": 10,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(root, *, bad_hp=None):
    """Stand-in weights with the real shapes plus a config pinned to them."""
    import torch
    from chemprop import nn

    config = json.loads(FROZEN.read_text())
    hp = dict(config["encoder"])
    torch.manual_seed(0)
    encoder = nn.BondMessagePassing(**hp)
    ckpt_hp = dict(hp, **(bad_hp or {}))
    weights = root / "chemeleon_mp.pt"
    torch.save({"hyper_parameters": ckpt_hp, "state_dict": encoder.state_dict()}, weights)
    config["weights"]["sha256"] = sha(weights)
    config_path = root / "chemeleon_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path, weights, encoder.state_dict()


class FrozenFileTests(unittest.TestCase):
    def test_frozen_config_matches_d022_and_the_real_weights_hash(self):
        config = json.loads(FROZEN.read_text())
        self.assertEqual(config["aggregation"], "mean")
        self.assertEqual(config["encoder"]["dropout"], 0.0)
        self.assertEqual(config["head"]["dropout"], 0.4)
        self.assertEqual(config["batch_size"], 16)
        self.assertEqual(config["encoder_lr_ratio"], 0.1)
        self.assertEqual(config["head_lr"], {
            "init": 1.4283739994469765e-05, "peak": 0.0008245400010426866,
            "final": 0.00020141638331344468, "warmup_epochs": 10,
        })
        if REAL_WEIGHTS.exists():
            self.assertEqual(sha(REAL_WEIGHTS), config["weights"]["sha256"])

    def test_the_updated_t121_recipe_equals_the_frozen_file(self):
        from cyp007_infrastructure import config_from_trial

        with tempfile.TemporaryDirectory() as tmp:
            cfg, w, _ = make_fixture(Path(tmp))
            arm = ch.ChemeleonArm.load(cfg, w, expected_config_sha256=sha(cfg))
            ch.check_recipe(arm, config_from_trial(name="x", sampled=SAMPLED, seed=SEED))
            with self.assertRaisesRegex(ValueError, "recipe differs"):
                ch.check_recipe(arm, config_from_trial(name="x", sampled={**SAMPLED, "dropout": 0.1}, seed=SEED))

    def test_load_refuses_wrong_hashes_and_wrong_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, w, _ = make_fixture(root)
            with self.assertRaisesRegex(ValueError, "config hash"):
                ch.ChemeleonArm.load(cfg, w, expected_config_sha256="0" * 64)
            data = bytearray(w.read_bytes())
            data[-9] ^= 0xFF
            w.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "weights hash"):
                ch.ChemeleonArm.load(cfg, w, expected_config_sha256=sha(cfg))
        with tempfile.TemporaryDirectory() as tmp:
            cfg, w, _ = make_fixture(Path(tmp), bad_hp={"depth": 3})
            with self.assertRaisesRegex(ValueError, "depth"):
                ch.ChemeleonArm.load(cfg, w, expected_config_sha256=sha(cfg))


class ArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cfg, w, cls.encoder_state = make_fixture(root)
        cls.arm = ch.ChemeleonArm.load(cfg, w, expected_config_sha256=sha(cfg))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pic50_state(self, **kw):
        import t121_weighted_training as weighted

        train, validation = frames()
        return weighted.build_updated_t121_pic50_prefit_state(
            training_dataframe=train, validation_dataframe=validation,
            sampled_config=SAMPLED, config_name="D007-T121", training_seed=SEED, **kw,
        )

    def screen_state(self, **kw):
        import t121_weighted_training as weighted

        with stub_screen_loader():
            return weighted.build_t121_screen_prefit_state(
                screen_path="x", identity_path="x", similarity_path="x", contract_path="x",
                sampled_config=SAMPLED, config_name="D007-T121", training_seed=SEED, **kw,
            )

    def check_architecture(self, model):
        import torch
        from chemprop import nn

        self.assertIsInstance(model.agg, nn.MeanAggregation)
        self.assertNotIsInstance(model.agg, nn.NormAggregation)
        self.assertEqual(model.message_passing.output_dim, 2048)
        self.assertEqual(model.message_passing.depth, 6)
        self.assertEqual(model.message_passing.dropout.p, 0.0)
        for key, value in model.message_passing.state_dict().items():
            self.assertTrue(torch.equal(value, self.encoder_state[key]), key)
        linears = [m for m in model.predictor.modules() if isinstance(m, torch.nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in linears], [(2048, 2000), (2000, 4)])
        drops = [m.p for m in model.predictor.modules() if isinstance(m, torch.nn.Dropout)]
        self.assertTrue(drops and all(p == 0.4 for p in drops))
        self.assertEqual(model.encoder_lr_ratio, 0.1)

    def test_pic50_and_screening_builds_have_the_agreed_architecture(self):
        pic50 = self.pic50_state(chemeleon=self.arm)
        self.check_architecture(pic50["model"])
        self.assertEqual(pic50["train_loader"].batch_sampler.batch_size, 16)
        self.assertEqual(pic50["chemeleon"]["weights_sha256"], self.arm.weights_sha256)
        screen = self.screen_state(chemeleon=self.arm)
        self.check_architecture(screen["model"])
        self.assertIsNone(screen["model"].predictor.output_transform.__class__.__dict__.get("mean_"))

    def test_encoder_survives_the_stage_transfer_and_the_head_projection_is_fresh(self):
        import torch
        import t121_weighted_training as weighted

        screen = self.screen_state(chemeleon=self.arm)["model"]
        with torch.no_grad():  # as if screening had changed the encoder
            for p in screen.message_passing.parameters():
                p.add_(0.01)
        pic50 = self.pic50_state(chemeleon=self.arm)["model"]
        projection = "predictor.ffn.1.2"
        fresh_projection = {
            k: v.clone() for k, v in pic50.state_dict().items()
            if k.startswith(projection + ".") or k.startswith("predictor.output_transform.")
        }
        self.assertEqual(
            sorted(k for k in fresh_projection if k.startswith(projection)),
            [projection + ".bias", projection + ".weight"],
        )
        report = weighted.transfer_screen_initialization(screen, pic50)
        self.assertEqual(report["final_projection"], projection)
        self.assertFalse(report["optimizer_state_transferred"])
        self.assertFalse(report["scheduler_state_transferred"])
        for key, value in screen.message_passing.state_dict().items():
            self.assertTrue(torch.equal(pic50.message_passing.state_dict()[key], value))
        for key, value in fresh_projection.items():
            self.assertTrue(torch.equal(pic50.state_dict()[key], value), key)
        self.assertEqual(pic50.encoder_lr_ratio, 0.1)

    def test_a_chemeleon_screen_cannot_transfer_into_a_graph_only_pic50_stage(self):
        import t121_weighted_training as weighted

        with self.assertRaises(ValueError):
            weighted.transfer_screen_initialization(
                self.screen_state(chemeleon=self.arm)["model"], self.pic50_state()["model"]
            )

    def test_descriptors_are_refused_with_chemeleon(self):
        with self.assertRaisesRegex(ValueError, "no descriptors"):
            self.pic50_state(chemeleon=self.arm, descriptor_dim=196)
        with self.assertRaisesRegex(ValueError, "no descriptors"):
            self.screen_state(chemeleon=self.arm, descriptor_cache=object())

    def test_head_initialisation_is_seeded_and_repeatable(self):
        import torch

        a = self.screen_state(chemeleon=self.arm)["model"]
        b = self.screen_state(chemeleon=self.arm)["model"]
        for key, value in a.predictor.state_dict().items():
            self.assertTrue(torch.equal(value, b.predictor.state_dict()[key]), key)

    def test_learning_rates_are_exactly_one_tenth_for_the_encoder_at_every_step(self):
        import torch
        from lightning import pytorch as pl
        from chemprop import nn

        records = []

        class Record(pl.Callback):
            def on_train_batch_start(self, trainer, module, batch, batch_idx):
                groups = trainer.optimizers[0].param_groups
                records.append((groups[0]["lr"], groups[1]["lr"], len(groups)))

        state = self.screen_state(chemeleon=self.arm)
        trainer = pl.Trainer(
            accelerator="cpu", devices=1, max_epochs=12, callbacks=[Record()],
            enable_checkpointing=False, logger=False, enable_progress_bar=False,
            num_sanity_val_steps=0, deterministic=True,
        )
        trainer.fit(state["model"], train_dataloaders=state["train_loader"])
        steps_per_epoch = len(state["train_loader"])
        self.assertEqual(len(records), 12 * steps_per_epoch)
        self.assertTrue(all(n == 2 for _, _, n in records))
        for head, encoder, _ in records:
            self.assertAlmostEqual(encoder / head, 0.1, places=12)
        c = state["config"]
        self.assertEqual(records[0][0], c.initial_lr)
        self.assertEqual(records[0][1], 0.1 * c.initial_lr)
        warm = c.warmup_epochs * steps_per_epoch
        self.assertAlmostEqual(records[warm][0], c.max_lr, delta=1e-12)
        self.assertAlmostEqual(records[warm][1], 0.1 * c.max_lr, delta=1e-13)

        # head schedule equals Chemprop's stock schedule for the same step counts
        stock = self.screen_state()
        stock_records = []

        class RecordStock(pl.Callback):
            def on_train_batch_start(self, trainer, module, batch, batch_idx):
                stock_records.append([g["lr"] for g in trainer.optimizers[0].param_groups])

        trainer = pl.Trainer(
            accelerator="cpu", devices=1, max_epochs=12, callbacks=[RecordStock()],
            enable_checkpointing=False, logger=False, enable_progress_bar=False,
            num_sanity_val_steps=0, deterministic=True,
        )
        trainer.fit(stock["model"], train_dataloaders=stock["train_loader"])
        self.assertTrue(all(len(r) == 1 for r in stock_records))
        np.testing.assert_allclose([r[0] for r in stock_records], [r[0] for r in records], rtol=1e-12)

    def test_without_chemeleon_the_optimizer_is_chemprops_single_group(self):
        from lightning import pytorch as pl

        n_groups = []

        class Record(pl.Callback):
            def on_train_batch_start(self, trainer, module, batch, batch_idx):
                n_groups.append(len(trainer.optimizers[0].param_groups))

        state = self.screen_state()
        self.assertIsNone(state["model"].encoder_lr_ratio)
        trainer = pl.Trainer(
            accelerator="cpu", devices=1, max_epochs=1, callbacks=[Record()],
            enable_checkpointing=False, logger=False, enable_progress_bar=False,
            num_sanity_val_steps=0, deterministic=True,
        )
        trainer.fit(state["model"], train_dataloaders=state["train_loader"])
        self.assertEqual(set(n_groups), {1})


if __name__ == "__main__":
    unittest.main()
