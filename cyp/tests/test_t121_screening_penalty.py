"""D024 screening penalty: rules, join, loss equivalence and the training path.

Synthetic names, structures, labels and screening values only (no project data,
no protected molecules). The model tests use the real Chemprop classes, the
updated-T121 builder and Lightning on CPU; run in the environment that has
chemprop and lightning (tripwire_cyp002).
"""
from pathlib import Path
import copy
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXECUTOR = REPO / "qualification/t121_executor_source_20261117"
if not EXECUTOR.is_dir():  # deployed package layout
    EXECUTOR = REPO / "executor_source"
for directory in (REPO / "cyp/scripts", EXECUTOR, REPO):
    sys.path.insert(0, str(directory))

import t121_screening_penalty as sp

CONFIGS = REPO / "cyp/artifacts/d024_penalty"
ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TASKS = [f"{i}_pIC50_direct_inhibition" for i in ENZYMES]
SMILES = [
    "CCO", "CCN", "CCC", "CCCC", "c1ccccc1", "c1ccccc1O", "c1ccccc1N",
    "CC(=O)O", "CC(=O)N", "CCOC", "CCCO", "CCCN", "c1ccncc1", "C1CCCCC1",
    "CC(C)C", "CC(C)O", "CCS", "CCCl", "c1ccccc1C", "C1CCOC1",
    "CCOC(=O)C", "CN(C)C", "c1ccc2ccccc2c1", "OCCO",
]
NAMES = [f"SYN-{i:03d}" for i in range(len(SMILES))]
SAMPLED = {
    "batch_size": 8, "dropout": 0.4, "ffn_hidden_dim": 300, "ffn_num_layers": 1,
    "final_lr_ratio": 0.1, "init_lr_ratio": 0.1, "max_lr": 0.001,
    "message_hidden_dim": 300, "message_passing_depth": 2, "warmup_epochs": 1,
}
SEED = 20261101
PRE_EDIT_COMMIT = "9bcf355"
PRE_EDIT_PATH = "cyp/scripts/t121_weighted_training.py"
PRE_EDIT_SHA256 = "50414196e14dd1846480749d328b7fcb9c680ca08cfed4f7cf47f4b6fa67c49e"
EXECUTOR_SHA256 = "625a64fae191573b5a860cdba9a6ae9456d17bc6cd5b0337cc0b7cff1df1a00b"


def arm(name, **overrides):
    config = json.loads((CONFIGS / f"penalty_{name}.json").read_text())
    config.update(overrides)
    return sp.PenaltyArm(config, "test")


def frames():
    """(training, validation) frames; the last 18 molecules train."""
    import pandas as pd

    rng = np.random.default_rng(1)
    frame = {"Molecule_Name": NAMES, "SMILES": SMILES}
    for task in TASKS:
        y = rng.normal(5.5, 1.0, len(NAMES))
        y[rng.random(len(NAMES)) < 0.25] = np.nan
        width = rng.uniform(0.2, 0.9, len(NAMES))
        frame[task] = y
        frame[task + "_conf_low"] = y - width / 2
        frame[task + "_conf_high"] = y + width / 2
    df = pd.DataFrame(frame)
    for i in range(len(NAMES)):
        if df.loc[i, TASKS].isna().all():
            df.loc[i, TASKS[0]] = 5.0
            df.loc[i, TASKS[0] + "_conf_low"] = 4.8
            df.loc[i, TASKS[0] + "_conf_high"] = 5.2
    df.index = df["Molecule_Name"]
    return df.iloc[6:].copy(), df.iloc[:6].copy()


def default_screen_rows(concentration="5e-05"):
    rng = np.random.default_rng(7)
    return [
        {"Molecule_Name": name, "SMILES": smiles, "enzyme": enzyme,
         "concentration_M": concentration, "log2fc_estimate": repr(float(rng.normal(-1, 1)))}
        for name, smiles in zip(NAMES, SMILES) for enzyme in ENZYMES
    ]


def default_identity_rows():
    return [
        {"molecule_name": name, "smiles": smiles, "class": "full_key_match", "tiers": "train"}
        for name, smiles in zip(NAMES, SMILES)
    ]


def table_bytes(rows, delimiter):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()  # CRLF, as in the project files


def input_bytes(screen_rows=None, identity_rows=None):
    screen = table_bytes(screen_rows or default_screen_rows(), ",")
    identity = table_bytes(identity_rows or default_identity_rows(), "\t")
    contract = json.dumps({"sha256": {
        "screen": hashlib.sha256(screen).hexdigest(),
        "identity": hashlib.sha256(identity).hexdigest(),
    }}).encode()
    return screen, identity, contract


def parsed(screen_rows=None, identity_rows=None, penalty_arm=None):
    return sp.parse_screen(*input_bytes(screen_rows, identity_rows), penalty_arm or arm("q3"))


def write_inputs(directory, **kwargs):
    paths = {}
    for key, data in zip(("screen_path", "identity_path", "contract_path"), input_bytes(**kwargs)):
        paths[key] = Path(directory) / key
        paths[key].write_bytes(data)
    paths["similarity_path"] = Path(directory) / "unused"
    return paths


class ConfigTests(unittest.TestCase):
    def test_frozen_configs_hold_the_recorded_settings(self):
        q3, rank = arm("q3"), arm("rank")
        self.assertEqual((q3.rule, q3.strength, q3.config["quantile"]), ("q3_cutoff", 1.0, 0.75))
        self.assertEqual((rank.rule, rank.strength), ("response_rank", 1.0))
        evaluation = json.loads((CONFIGS / "evaluation.json").read_text())
        self.assertEqual(evaluation["arm_labels"]["q3"], q3.arm_label)
        self.assertEqual(evaluation["arm_labels"]["rank"], rank.arm_label)
        self.assertEqual(list(q3.config["tasks"]), TASKS)
        self.assertEqual(q3.config["screen"], rank.config["screen"])
        self.assertEqual(q3.config["identity"], rank.config["identity"])

    def test_lambda_must_be_set_and_valid(self):
        config = json.loads((CONFIGS / "penalty_q3.json").read_text())
        del config["lambda"]
        with self.assertRaisesRegex(ValueError, "lacks"):
            sp.PenaltyArm(config, "x")
        for bad in (None, -1.0, float("nan"), True, "1"):
            with self.assertRaisesRegex(ValueError, "lambda"):
                arm("q3", **{"lambda": bad})
        self.assertEqual(arm("q3", **{"lambda": 0}).strength, 0.0)

    def test_load_pins_the_config_hash(self):
        path = CONFIGS / "penalty_rank.json"
        sp.PenaltyArm.load(path, expected_sha256=sp.sha256_file(path))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            sp.PenaltyArm.load(path, expected_sha256="0" * 64)


class WeightTests(unittest.TestCase):
    def test_quantile_is_linear_interpolation(self):
        self.assertEqual(sp.linear_quantile([4, 1, 3, 2], 0.75), 3.25)
        rng = np.random.default_rng(0)
        for n in (2, 3, 7, 50, 1001):
            values = rng.normal(0, 1, n)
            self.assertAlmostEqual(
                sp.linear_quantile(values, 0.75), float(np.percentile(values, 75)), places=12
            )

    def test_cutoff_flags_strictly_above_q3_with_realized_fraction(self):
        values = [0.0, 1.0, 2.0, 2.0, 2.0, 5.0]  # Q3 = 2, tied values not flagged
        weights, summary = sp.population_weights(values, arm("q3"))
        self.assertEqual(summary["q3"], 2.0)
        self.assertEqual(weights, [0.0, 0.0, 0.0, 0.0, 0.0, 6.0])
        self.assertEqual(summary["flagged"], 1)
        self.assertAlmostEqual(summary["flagged_fraction"], 1 / 6)
        self.assertAlmostEqual(summary["mean_weight"], 1.0, places=12)

    def test_rank_uses_average_ranks_and_mean_one(self):
        weights, summary = sp.population_weights([10.0, 30.0, 20.0, 20.0], arm("rank"))
        np.testing.assert_allclose(weights, [0.0, 2.0, 1.0, 1.0], atol=1e-12)
        self.assertEqual(summary["flagged"], 3)
        values = np.random.default_rng(2).normal(0, 1, 501).round(1)
        weights, summary = sp.population_weights(values, arm("rank"))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=12)

    def test_degenerate_populations_switch_the_penalty_off(self):
        for name in ("q3", "rank"):
            for values, reason in (
                ([], "fewer_than_two_eligible"), ([1.0], "fewer_than_two_eligible"),
                ([2.0, 2.0, 2.0], "all_responses_equal"),
            ):
                weights, summary = sp.population_weights(values, arm(name))
                self.assertEqual((summary["active"], summary["reason"]), (False, reason))
                self.assertEqual(weights, [0.0] * len(values))

    def test_nothing_above_q3_disables_only_the_cutoff_arm(self):
        values = [1.0, 2.0, 2.0, 2.0, 2.0]
        weights, summary = sp.population_weights(values, arm("q3"))
        self.assertEqual(summary["reason"], "nothing_strictly_above_q3")
        self.assertEqual(weights, [0.0] * 5)
        weights, summary = sp.population_weights(values, arm("rank"))
        self.assertTrue(summary["active"])
        self.assertGreater(sum(weights), 0)


class JoinTests(unittest.TestCase):
    def test_each_status_gets_no_penalty_unless_verified(self):
        screen_rows = default_screen_rows()
        identity_rows = default_identity_rows()
        screen_rows = [r for r in screen_rows if r["Molecule_Name"] != "SYN-006"]
        for row in screen_rows:
            if row["Molecule_Name"] == "SYN-007":
                row["log2fc_estimate"] = ""
            if row["Molecule_Name"] == "SYN-009":
                row["SMILES"] = "CCCCCC"  # differs from its adjudication
        identity_rows[8]["class"] = "no_identity_link"
        del identity_rows[10]
        extra = [dict(r) for r in screen_rows if r["Molecule_Name"] == "SYN-011"]
        conflicting = [dict(r, log2fc_estimate="9.0") for r in screen_rows if r["Molecule_Name"] == "SYN-012"]
        screen = parsed(screen_rows + extra + conflicting, identity_rows)
        a = arm("q3")
        expected = {
            "SYN-006": "no_screening_row", "SYN-007": "missing_value",
            "SYN-008": "unverified_identity", "SYN-009": "unverified_identity",
            "SYN-010": "unverified_identity", "SYN-011": "ok", "SYN-012": "conflicting_records",
        }
        for name, status in expected.items():
            self.assertEqual(sp.lookup(screen, a, name, "CYP1A2")[0], status, name)
        self.assertEqual(sp.lookup(screen, a, "SYN-011", "CYP1A2")[3], 1)  # collapsed
        train, _ = frames()
        rows, summary = sp.build_population(train, screen, a)
        for row in rows:
            if row["status"] != "ok":
                self.assertEqual(row["weight"], 0.0)
        entry = summary["enzymes"]["CYP1A2"]
        self.assertEqual(entry["population"], entry["status_counts"]["ok"])
        self.assertEqual(sum(entry["status_counts"].values()), entry["labelled_training_molecules"])

    def test_protected_tier_in_the_training_population_raises(self):
        identity_rows = default_identity_rows()
        identity_rows[8]["tiers"] = "internal_test"
        train, _ = frames()
        train.loc["SYN-008", TASKS[0]] = 5.0
        with self.assertRaisesRegex(ValueError, "Protected-tier"):
            sp.build_population(train, parsed(identity_rows=identity_rows), arm("q3"))

    def test_second_condition_or_wrong_hash_stops(self):
        rows = default_screen_rows()
        rows[0]["concentration_M"] = "1e-05"
        with self.assertRaisesRegex(ValueError, "condition key must be updated"):
            parsed(rows)
        screen, identity, contract = input_bytes()
        with self.assertRaisesRegex(ValueError, "contract's SHA-256"):
            sp.parse_screen(screen + b" ", identity, contract, arm("q3"))

    def test_training_smiles_difference_is_reported_and_stops(self):
        train, _ = frames()
        train.loc["SYN-010", "SMILES"] = "CCCCCCC"
        train.loc["SYN-010", TASKS[0]] = 5.0
        _, summary = sp.build_population(train, parsed(), arm("q3"))
        self.assertIn("SYN-010", {d["Molecule_Name"] for d in summary["training_smiles_differences"]})
        with self.assertRaisesRegex(ValueError, "differ from the training SMILES"):
            sp.require_no_smiles_differences(summary)

    def test_only_training_fold_molecules_shape_the_weights(self):
        train, validation = frames()
        rows = default_screen_rows()
        held_out = set(validation["Molecule_Name"])
        for name in ("q3", "rank"):
            a = arm(name)
            base_rows, base_summary = sp.build_population(train, parsed(rows), a)
            extreme = [
                dict(r, log2fc_estimate="999.0") if r["Molecule_Name"] in held_out else r for r in rows
            ]
            dropped = [r for r in rows if r["Molecule_Name"] not in held_out]
            for changed in (extreme, dropped):
                new_rows, new_summary = sp.build_population(train, parsed(changed), a)
                self.assertEqual(sp.weights_text(new_rows), sp.weights_text(base_rows))
                self.assertEqual(new_summary, base_summary)
            shuffled, _ = sp.build_population(train.iloc[::-1], parsed(rows), a)
            self.assertEqual(sp.weights_text(shuffled), sp.weights_text(base_rows))
            for enzyme, entry in base_summary["enzymes"].items():
                ok = [r["weight"] for r in base_rows if r["enzyme"] == enzyme and r["status"] == "ok"]
                self.assertAlmostEqual(float(np.mean(ok)), 1.0, places=12)

    def test_unlabelled_cells_get_no_row_and_an_empty_enzyme_disables_only_itself(self):
        train, _ = frames()
        train[TASKS[2]] = np.nan
        rows, summary = sp.build_population(train, parsed(), arm("rank"))
        self.assertFalse([r for r in rows if r["enzyme"] == "CYP2D6"])
        self.assertEqual(summary["enzymes"]["CYP2D6"]["reason"], "no_labelled_training_molecules")
        self.assertTrue(summary["enzymes"]["CYP1A2"]["active"])
        self.assertEqual(
            len([r for r in rows if r["enzyme"] == "CYP1A2"]), int(train[TASKS[0]].notna().sum())
        )


class LossTests(unittest.TestCase):
    def batch(self):
        import torch

        nan = float("nan")
        predictions = torch.tensor([[1.0, 0.0, 2.0, -1.0], [0.5, 3.0, 0.0, 1.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 0.0, nan, 0.0], [0.0, 0.0, nan, nan]])
        weights = torch.tensor([[2.0, 1.0, 5.0, 1.0], [1.0, 0.0, 5.0, 5.0]])
        upper = torch.tensor([[0.5, 0.0, nan, 0.0], [1.0, 1.0, nan, nan]])
        return predictions, targets, weights, upper

    def test_cells_gradients_and_the_bound(self):
        import torch

        predictions, targets, weights, upper = self.batch()
        cells = sp.penalty_cells(predictions, targets, weights, upper)
        expected = torch.tensor([[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(cells.detach(), expected)  # at the bound: zero
        cells.sum().backward()
        torch.testing.assert_close(
            predictions.grad, torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        )

    def test_standardized_and_native_excess_agree(self):
        rng = np.random.default_rng(5)
        mean, scale = np.array([5.0, 4.5, 5.5, 6.0]), np.array([0.5, 1.0, 1.5, 2.0])
        high, prediction = rng.normal(5, 1, (20, 4)), rng.normal(5, 1, (20, 4))
        upper = sp.standardized_upper_bounds(high, mean, scale)
        np.testing.assert_allclose(upper, (high - mean) / scale)
        standardized = np.maximum((prediction - mean) / scale - upper, 0)
        np.testing.assert_allclose(standardized * scale, np.maximum(prediction - high, 0), atol=1e-12)

    def table(self, strength, path=None):
        _, _, weights, upper = self.batch()
        return sp.ScreeningPenalty(
            strength=strength, molecule_ids=["a", "b"], weights=weights.numpy(),
            upper=upper.numpy(), enzymes=ENZYMES, history_path=path,
        )

    def apply(self, penalty, base_value, raw_scale=1.0):
        import torch

        predictions, targets, _, _ = self.batch()
        raw = torch.where(targets.isfinite(), torch.full_like(targets, raw_scale), torch.zeros_like(targets))
        raw[0, 0] = raw[0, 0] * 3
        return penalty.apply(
            torch.tensor(base_value), predictions, targets, raw, torch.ones(2, 1), ["a", "b"]
        )

    def test_denominator_is_the_observed_cell_count_and_ci_weights_do_not_enter(self):
        # five observed cells; P = 0.5 / 5
        first = self.apply(self.table(1.0), 2.0)
        self.assertAlmostEqual(float(first), 2.1, places=6)
        self.assertEqual(float(self.apply(self.table(1.0), 2.0, raw_scale=7.0)), float(first))
        self.assertAlmostEqual(float(self.apply(self.table(0.5), 2.0)), 2.05, places=6)
        self.assertEqual(float(self.apply(self.table(0.0), 2.0)), 2.0)

    def test_unknown_molecule_raises(self):
        import torch

        predictions, targets, _, _ = self.batch()
        with self.assertRaisesRegex(ValueError, "absent from the penalty table"):
            self.table(1.0).apply(
                torch.tensor(1.0), predictions, targets, torch.ones_like(targets),
                torch.ones(2, 1), ["a", "zzz"],
            )

    def test_epoch_record_fields_and_appended_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / sp.HISTORY_FILE
            penalty = self.table(1.0, path)
            for epoch in (0, 1):
                self.apply(penalty, 2.0)
                self.apply(penalty, 4.0)
                penalty.flush_epoch(epoch)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records], [0, 1])
            record = records[0]
            self.assertEqual((record["batches"], record["observed_cells"]), (2, 10))
            self.assertAlmostEqual(record["base_loss"], 3.0)
            self.assertAlmostEqual(record["penalty_unscaled"], 0.1, places=6)
            self.assertAlmostEqual(record["penalty_scaled"], 0.1, places=6)
            first = record["per_enzyme"]["CYP1A2"]
            self.assertEqual(
                (first["observed_cells"], first["flagged_cells"], first["active_cells"]), (4, 4, 2)
            )
            self.assertAlmostEqual(first["penalty_sum"], 1.0, places=6)
            self.assertEqual(record["per_enzyme"]["CYP2D6"]["observed_cells"], 0)
            with self.assertRaisesRegex(RuntimeError, "No training batches"):
                penalty.flush_epoch(2)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def prefit(self):
        import t121_weighted_training as weighted

        train, validation = frames()
        state = weighted.build_updated_t121_pic50_prefit_state(
            training_dataframe=train, validation_dataframe=validation,
            sampled_config=SAMPLED, config_name="D007-T121", training_seed=SEED,
        )
        return state, train, validation

    def attached(self, state, train, directory, penalty_arm, **inputs):
        return sp.attach_penalty(
            prefit=state, training_dataframe=train,
            screen_inputs=write_inputs(directory, **inputs), arm=penalty_arm, cell_dir=directory,
        )

    @staticmethod
    def step(function, model, batch):
        import torch

        model = copy.deepcopy(model)
        model.train()
        torch.manual_seed(11)
        loss = function(model, batch, 0)
        loss.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return loss.detach().clone(), gradients

    def pre_edit_module(self):
        try:
            source = subprocess.run(
                ["git", "-C", str(REPO), "show", f"{PRE_EDIT_COMMIT}:{PRE_EDIT_PATH}"],
                check=True, capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("pre-edit commit not available (deployed package has no git history)")
        self.assertEqual(hashlib.sha256(source).hexdigest(), PRE_EDIT_SHA256)
        module = types.ModuleType("t121_weighted_training_pre_edit")
        exec(compile(source, PRE_EDIT_PATH, "exec"), module.__dict__)
        return module

    def test_disabled_and_zero_lambda_equal_the_pre_edit_loss_and_gradients(self):
        import torch
        import t121_weighted_training as weighted

        old = self.pre_edit_module().T121WeightedTrainingMPNN.training_step
        new = weighted.T121WeightedTrainingMPNN.training_step
        for name in ("q3", "rank"):
            with self.subTest(arm=name), tempfile.TemporaryDirectory() as tmp:
                state, train, _ = self.prefit()
                batch = next(iter(state["train_loader"]))
                reference_loss, reference_grads = self.step(old, state["model"], batch)
                disabled = self.step(new, state["model"], batch)
                self.attached(state, train, tmp, arm(name, **{"lambda": 0.0}))
                zero = self.step(new, state["model"], batch)
                for loss, grads in (disabled, zero):
                    self.assertTrue(torch.equal(loss, reference_loss))
                    self.assertEqual(set(grads), set(reference_grads))
                    for key, value in grads.items():
                        self.assertTrue(torch.equal(value, reference_grads[key]), key)

    def test_unit_lambda_adds_exactly_the_defined_penalty(self):
        import torch
        import t121_weighted_training as weighted

        step = weighted.T121WeightedTrainingMPNN.training_step
        for name in ("q3", "rank"):
            with self.subTest(arm=name), tempfile.TemporaryDirectory() as tmp:
                state, train, _ = self.prefit()
                batch = next(iter(state["train_loader"]))
                base, base_grads = self.step(step, state["model"], batch)
                self.attached(state, train, tmp, arm(name))
                total, grads = self.step(step, state["model"], batch)

                bmg, V_d, X_d, targets = batch["chemprop_batch"][:4]
                model = copy.deepcopy(state["model"])
                model.train()
                torch.manual_seed(11)
                with torch.no_grad():
                    predictions = model.predictor.train_step(model.fingerprint(bmg, V_d, X_d))
                # Rebuild the penalty from native values, independently of the table.
                with (Path(tmp) / sp.WEIGHTS_FILE).open(newline="") as handle:
                    weights = {(r["Molecule_Name"], r["enzyme"]): float(r["weight"])
                               for r in csv.DictReader(handle)}
                scaler, total_penalty, observed = state["scaler"], 0.0, 0
                for i, molecule in enumerate(batch["molecule_ids"]):
                    for j, (task, enzyme) in enumerate(zip(TASKS, ENZYMES)):
                        if not np.isfinite(train.loc[molecule, task]):
                            continue
                        observed += 1
                        upper = (train.loc[molecule, task + "_conf_high"] - scaler.mean_[j]) / scaler.scale_[j]
                        excess = max(float(predictions[i, j]) - upper, 0.0)
                        total_penalty += weights[(molecule, enzyme)] * excess ** 2
                        # loader targets are in the same scaler units as the bound
                        self.assertAlmostEqual(
                            float(targets[i, j]),
                            (train.loc[molecule, task] - scaler.mean_[j]) / scaler.scale_[j], places=5,
                        )
                self.assertGreater(total_penalty, 0)
                self.assertAlmostEqual(float(total - base), total_penalty / observed, places=5)
                self.assertTrue(any(not torch.equal(grads[k], base_grads[k]) for k in grads))

    def test_attach_writes_records_keeps_model_state_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = self.prefit()
            before = list(state["model"].state_dict())
            report = self.attached(state, train, tmp, arm("q3"))
            self.assertEqual(list(state["model"].state_dict()), before)
            self.assertEqual(report["weights_sha256"], sp.sha256_file(Path(tmp) / sp.WEIGHTS_FILE))
            saved = json.loads((Path(tmp) / sp.POPULATION_FILE).read_text())
            self.assertEqual(set(saved["enzymes"]), set(ENZYMES))
            fresh, _, _ = self.prefit()
            fresh["model"].load_state_dict(state["model"].state_dict(), strict=True)
            self.assertIsNone(fresh["model"].screening_penalty)
            with self.assertRaises(FileExistsError):
                self.attached(state, train, tmp, arm("q3"))

    def test_attach_stops_on_smiles_difference_after_writing_the_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = self.prefit()
            identity = default_identity_rows()
            screen = default_screen_rows()
            index = NAMES.index(train["Molecule_Name"].iloc[0])
            identity[index]["smiles"] = "CCCCCCCC"
            for row in screen:
                if row["Molecule_Name"] == NAMES[index]:
                    row["SMILES"] = "CCCCCCCC"
            with self.assertRaisesRegex(ValueError, "differ from the training SMILES"):
                self.attached(state, train, tmp, arm("q3"), screen_rows=screen, identity_rows=identity)
            self.assertTrue((Path(tmp) / sp.POPULATION_FILE).exists())
            self.assertIsNone(state["model"].screening_penalty)

    def fit(self, state, directory, extra_callbacks=()):
        from lightning import pytorch as pl
        from cyp002_validation import build_validation_callbacks

        trainer = pl.Trainer(
            accelerator="cpu", devices=1, deterministic=True, max_epochs=2, logger=False,
            enable_progress_bar=False, enable_model_summary=False, default_root_dir=directory,
            callbacks=build_validation_callbacks(Path(directory) / "fit", patience=5)
            + list(extra_callbacks),
        )
        trainer.fit(state["model"], state["train_loader"], state["validation_loader"])
        return trainer

    def test_one_fit_per_arm_updates_weights_and_logs_every_epoch(self):
        import torch

        for name in ("q3", "rank"):
            with self.subTest(arm=name), tempfile.TemporaryDirectory() as tmp:
                state, train, _ = self.prefit()
                self.attached(state, train, tmp, arm(name))
                before = copy.deepcopy(state["model"].state_dict())
                self.fit(state, tmp)
                after = state["model"].state_dict()
                self.assertTrue(any(not torch.equal(before[k], after[k].cpu()) for k in before))
                self.assertTrue(all(torch.isfinite(v).all() for v in after.values()))
                lines = (Path(tmp) / sp.HISTORY_FILE).read_text().splitlines()
                records = [json.loads(line) for line in lines]
                self.assertEqual(len(records), len(state["model"].validation_history))
                self.assertEqual([r["epoch"] for r in records], [0, 1])
                observed = int(train[TASKS].notna().to_numpy().sum())
                for record in records:
                    self.assertEqual(record["lambda"], 1.0)
                    self.assertEqual(record["observed_cells"], observed)
                    self.assertAlmostEqual(record["penalty_scaled"], record["penalty_unscaled"])
                    self.assertEqual(set(record["per_enzyme"]), set(ENZYMES))
                    base = sum(e["base_sum"] for e in record["per_enzyme"].values())
                    self.assertAlmostEqual(base / observed, record["base_loss"], places=4)

    def test_epoch_log_survives_a_failure_mid_run(self):
        from lightning import pytorch as pl

        class Fail(pl.Callback):
            def on_train_epoch_start(self, trainer, module):
                if trainer.current_epoch == 1:
                    raise RuntimeError("injected failure")

        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = self.prefit()
            self.attached(state, train, tmp, arm("rank"))
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.fit(state, tmp, extra_callbacks=[Fail()])
            self.assertEqual(len((Path(tmp) / sp.HISTORY_FILE).read_text().splitlines()), 1)

    def test_validation_hands_the_scorer_standardized_predictions_that_it_unscales(self):
        """Records which branch of the frozen scorer's scaling rule runs."""
        import torch
        from lightning import pytorch as pl
        import cyp002_validation as validation_module

        state, _, validation = self.prefit()
        model, loader = state["model"], state["validation_loader"]
        captured = {}
        original = validation_module.compute_validation_macro_st_rae

        def recording(y_true, y_pred, conf_low, conf_high, task_names):
            captured.update({t: np.array(y_pred[t]) for t in task_names})
            return original(y_true, y_pred, conf_low, conf_high, task_names)

        validation_module.compute_validation_macro_st_rae = recording
        try:
            pl.Trainer(
                accelerator="cpu", devices=1, deterministic=True, logger=False,
                enable_progress_bar=False, enable_model_summary=False,
                enable_checkpointing=False,
            ).validate(model, dataloaders=loader, verbose=False)
        finally:
            validation_module.compute_validation_macro_st_rae = original
        collected = torch.cat(model._validation_predictions).numpy()
        model.eval()
        with torch.no_grad():
            standardized = torch.cat([
                model.predictor.ffn(model.fingerprint(b.bmg, b.V_d, b.X_d)) for b in loader
            ]).numpy()
        np.testing.assert_allclose(collected, standardized, atol=1e-5)
        scaler = state["scaler"]
        for j, task in enumerate(TASKS):
            present = validation[task].notna().to_numpy()
            np.testing.assert_allclose(
                captured[task], standardized[present, j] * scaler.scale_[j] + scaler.mean_[j], atol=1e-4
            )

    def test_screening_model_has_no_penalty_hooks_and_the_executor_is_unchanged(self):
        import t121_weighted_training as weighted

        screen = weighted.T121ScreenTrainingMPNN
        self.assertFalse(hasattr(screen, "screening_penalty"))
        self.assertNotIn("on_train_epoch_end", screen.__dict__)
        self.assertEqual(sp.sha256_file(EXECUTOR / "cyp007_cell_fit.py"), EXECUTOR_SHA256)


class PreviewTests(unittest.TestCase):
    def test_preview_writes_every_fold_and_flags_disabled_cases(self):
        train, _ = frames()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "rep_01_fold_00.json"
            manifest.write_text(json.dumps({"repetition": 1, "fold": 0, "training_seeds": [SEED]}))
            arms = [arm("q3"), arm("rank")]

            def run(directory, frame):
                return sp.preview(
                    arms=arms, screens={a.arm: parsed(penalty_arm=a) for a in arms},
                    dataframe=None, split_dataframe=None, manifests=[manifest],
                    route_fold=lambda **kwargs: {"training_dataframe": frame},
                    output_dir=root / directory,
                )

            report = run("clean", train)
            self.assertFalse(report["stop"])
            self.assertIn("CYP2D6", sp.preview_table(report))
            for entry in report["folds"]:
                path = root / "clean" / entry["arm"] / f"rep_01_fold_00_{sp.WEIGHTS_FILE}"
                self.assertEqual(sp.sha256_file(path), entry["weights_sha256"])
            broken = train.copy()
            broken[TASKS[1]] = np.nan
            report = run("disabled", broken)
            self.assertTrue(report["stop"])
            self.assertEqual({d["enzyme"] for d in report["disabled"]}, {"CYP2C9"})


if __name__ == "__main__":
    unittest.main()
