"""Two further penalty forms on the D024 rank weights: a two-sided squared
confidence band and a Huber (upper side only) penalty.

    two_sided_squared  P = sum w * (relu(y - U)^2 + relu(L - y)^2) / N_observed
    upper_huber        P = sum w * H(relu(y - U)) / N_observed,
                       H(e) = e^2 for e <= delta, else 2*delta*e - delta^2

delta is, per enzyme and training fold, the median standardized confidence
half-width over observed training cells.

Written ahead of the implementation: tests of the new forms, their configs and
the aggregator's new arm names fail until it exists. Tests marked "guard" hold
the D024 behaviour and pass before and after.

Synthetic names, structures, labels and screening values only. The model tests
use the real Chemprop classes on CPU (environment tripwire_cyp002).
"""
from pathlib import Path
import contextlib
import copy
import csv
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_t121_screening_penalty as d024  # noqa: E402  (also sets the import paths)
import test_t121_d024_aggregate as d024_agg  # noqa: E402

sp = d024.sp
agg = d024_agg.agg
TASKS, ENZYMES = d024.TASKS, d024.ENZYMES

D024_CONFIGS = d024.CONFIGS
NEW_CONFIGS = d024.REPO / "cyp/artifacts/d029_penalty"
D024_CONFIG_SHA256 = {
    "penalty_q3.json": "ddfe1f01e853e990fabd3ac60e0e7fec5b4bb4e998800b46440b50a4d372af7b",
    "penalty_rank.json": "a20552a33392891dcd1e9eac4af178825c7fc476a7b596004e964f9fc58edefe",
    "evaluation.json": "8e7aabecdaa364f33faf0f59a9292e97a6455d9fcaa9c94dfec34863fd669c77",
    "coverage.json": "98935366c996d0d9d7237be471eb9483bd80a8cc6c93558c753249f585b11727",
}
# The penalty module as the D024 comparison ran it.
D024_COMMIT = "99a3301"
D024_MODULE_PATH = "cyp/scripts/t121_screening_penalty.py"
D024_MODULE_SHA256 = "14e4fd110ce3f718726782b15c97e23513a8772f99c29aff2dbf2a9616e98aa2"

FORMS = ("two_sided_squared", "upper_huber")
ARM_NAMES = {"two_sided_squared": "twosided", "upper_huber": "huber"}
ARM_LABELS = {
    "unchanged": "updated_t121_from_screen_checkpoint",
    "twosided": "updated_t121_rank_twosided_penalty_from_screen_checkpoint",
    "huber": "updated_t121_rank_huber_penalty_from_screen_checkpoint",
}


def read_config(path):
    return json.loads(Path(path).read_text())


def form_arm(form, /, **overrides):
    """A new-form arm built on the frozen rank settings."""
    config = read_config(D024_CONFIGS / "penalty_rank.json")
    name = ARM_NAMES[form]
    config.update(arm=name, arm_label=ARM_LABELS[name], form=form)
    config.update(overrides)
    return sp.PenaltyArm(config, "test")


def prefit():
    import t121_weighted_training as weighted

    train, validation = d024.frames()
    state = weighted.build_updated_t121_pic50_prefit_state(
        training_dataframe=train, validation_dataframe=validation,
        sampled_config=d024.SAMPLED, config_name="D007-T121", training_seed=d024.SEED,
    )
    return state, train, validation


def attach(state, train, directory, arm, module=sp, **inputs):
    return module.attach_penalty(
        prefit=state, training_dataframe=train,
        screen_inputs=d024.write_inputs(directory, **inputs), arm=arm, cell_dir=directory,
    )


def committed_module(commit, path, digest, name):
    try:
        source = subprocess.run(
            ["git", "-C", str(d024.REPO), "show", f"{commit}:{path}"],
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        raise unittest.SkipTest("reference commit not available (deployed package has no git history)")
    if sp.sha256_bytes(source) != digest:
        raise AssertionError(f"{commit}:{path} does not have its recorded SHA-256.")
    module = types.ModuleType(name)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def native_deltas(train, scale):
    """Independent delta: median half-width of observed cells in scaler units."""
    result = []
    for j, task in enumerate(TASKS):
        observed = train[task].notna().to_numpy()
        half = (train[task + "_conf_high"] - train[task + "_conf_low"]).to_numpy()[observed] / 2.0
        result.append(float(np.median(half)) / float(scale[j]))
    return np.array(result)


def first_observed(frame, task):
    return frame.index[frame[task].notna().to_numpy()][0]


# --------------------------------------------------------------------------- config
class ConfigTests(unittest.TestCase):
    def test_form_is_read_from_the_config_and_absent_means_the_d024_form(self):
        for name in ("q3", "rank"):
            self.assertEqual(d024.arm(name).form, "upper_squared")
        for form in FORMS:
            self.assertEqual(form_arm(form).form, form)
        with self.assertRaisesRegex(ValueError, "form"):
            form_arm("two_sided_squared", form="something_else")

    def test_guard_d024_configs_are_byte_identical(self):
        for name, digest in D024_CONFIG_SHA256.items():
            self.assertEqual(sp.sha256_file(D024_CONFIGS / name), digest, name)

    def test_frozen_new_arm_configs_differ_from_rank_only_in_identity_and_form(self):
        own = {"decision", "arm", "arm_label", "form"}
        rank = read_config(D024_CONFIGS / "penalty_rank.json")
        for form in FORMS:
            name = ARM_NAMES[form]
            path = NEW_CONFIGS / f"penalty_rank_{name}.json"
            config = read_config(path)
            self.assertEqual(
                {k: v for k, v in config.items() if k not in own},
                {k: v for k, v in rank.items() if k not in own},
            )
            self.assertEqual(
                (config["form"], config["arm"], config["arm_label"]),
                (form, name, ARM_LABELS[name]),
            )
            self.assertEqual((config["rule"], config["lambda"]), ("response_rank", 1.0))
            loaded = sp.PenaltyArm.load(path, expected_sha256=sp.sha256_file(path))
            self.assertEqual(loaded.form, form)

    def test_frozen_new_evaluation_config_keeps_the_d024_settings(self):
        own = {"decision", "arm_labels", "tie_preference"}
        old = read_config(D024_CONFIGS / "evaluation.json")
        new = read_config(NEW_CONFIGS / "evaluation.json")
        self.assertEqual(
            {k: v for k, v in new.items() if k not in own},
            {k: v for k, v in old.items() if k not in own},
        )
        self.assertEqual(new["arm_labels"], ARM_LABELS)
        self.assertIn(new["tie_preference"], ("twosided", "huber"))


# --------------------------------------------------------------------------- loss
class TwoSidedLossTests(unittest.TestCase):
    def batch(self):
        import torch

        nan = float("nan")
        predictions = torch.tensor(
            [[1.5, -1.5, 0.0, 0.5], [-0.5, 2.5, 7.0, -3.5]], requires_grad=True
        )
        targets = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, nan, 0.0]])
        weights = torch.tensor([[2.0, 1.0, 1.0, 1.0], [1.0, 0.0, 5.0, 0.5]])
        upper = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, nan, 0.5]])
        lower = torch.tensor([[-0.5, -0.5, -0.5, -0.5], [-0.5, -0.5, nan, -0.5]])
        return predictions, targets, weights, upper, lower

    def test_cells_and_gradients_on_both_sides(self):
        import torch

        predictions, targets, weights, upper, lower = self.batch()
        cells = sp.penalty_cells(predictions, targets, weights, upper, lower=lower)
        # above U, below L, inside, at U; at L, zero weight, unobserved, below L
        expected = torch.tensor([[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.5]])
        torch.testing.assert_close(cells.detach(), expected)
        cells.sum().backward()
        torch.testing.assert_close(
            predictions.grad,
            torch.tensor([[4.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.0, -3.0]]),
        )

    def test_without_the_lower_bound_only_the_upper_side_remains(self):
        import torch

        predictions, targets, weights, upper, _ = self.batch()
        cells = sp.penalty_cells(predictions, targets, weights, upper, lower=None)
        expected = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(cells.detach(), expected)

    def test_non_finite_lower_bound_on_an_observed_cell_raises(self):
        predictions, targets, weights, upper, lower = self.batch()
        lower[0, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            sp.penalty_cells(predictions, targets, weights, upper, lower=lower)

    def test_denominator_lambda_and_epoch_record(self):
        import torch

        _, targets, weights, upper, lower = self.batch()

        def table(strength):
            return sp.ScreeningPenalty(
                strength=strength, molecule_ids=["a", "b"], weights=weights.numpy(),
                upper=upper.numpy(), lower=lower.numpy(), enzymes=ENZYMES,
            )

        def apply(penalty, raw_scale=1.0):
            predictions = self.batch()[0]
            raw = torch.where(
                targets.isfinite(), torch.full_like(targets, raw_scale), torch.zeros_like(targets)
            )
            return penalty.apply(
                torch.tensor(2.0), predictions, targets, raw, torch.ones(2, 1), ["a", "b"]
            )

        # seven observed cells, cell sum 7.5
        penalty = table(1.0)
        self.assertAlmostEqual(float(apply(penalty)), 2.0 + 7.5 / 7, places=6)
        self.assertEqual(float(apply(table(1.0), raw_scale=7.0)), float(apply(table(1.0))))
        self.assertAlmostEqual(float(apply(table(0.5))), 2.0 + 0.5 * 7.5 / 7, places=6)
        self.assertEqual(float(apply(table(0.0))), 2.0)
        record = penalty.epoch_record(0)
        self.assertEqual(record["observed_cells"], 7)
        self.assertAlmostEqual(record["penalty_unscaled"], 7.5 / 7, places=6)
        last = record["per_enzyme"]["CYP3A4"]
        self.assertEqual((last["active_cells"], last["observed_cells"]), (1, 2))
        self.assertAlmostEqual(last["penalty_sum"], 4.5, places=6)


class HuberLossTests(unittest.TestCase):
    DELTA = [1.0, 0.5, 2.0, 1.0]

    def batch(self):
        import torch

        nan = float("nan")
        predictions = torch.tensor(
            [[0.5, 0.5, -1.0, 1.0], [3.0, 2.0, 7.0, 4.0]], requires_grad=True
        )
        targets = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, nan, 0.0]])
        weights = torch.tensor([[2.0, 1.0, 1.0, 1.0], [1.0, 0.0, 5.0, 0.5]])
        upper = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, nan, 0.0]])
        return predictions, targets, weights, upper

    def test_cells_equal_twice_the_torch_huber_loss_per_enzyme(self):
        import torch

        predictions, targets, weights, upper = self.batch()
        delta = torch.tensor(self.DELTA)
        cells = sp.penalty_cells(predictions, targets, weights, upper, delta=delta)
        # e below delta, at delta, zero, at delta; above, zero weight, unobserved, above
        hand = torch.tensor([[0.5, 0.25, 0.0, 1.0], [5.0, 0.0, 0.0, 3.5]])
        torch.testing.assert_close(cells.detach(), hand)
        excess = torch.relu(predictions.detach() - torch.nan_to_num(upper))
        for j, value in enumerate(self.DELTA):
            huber = torch.nn.functional.huber_loss(
                excess[:, j], torch.zeros(2), reduction="none", delta=value
            )
            reference = torch.where(targets[:, j].isfinite(), weights[:, j] * 2 * huber, 0.0)
            torch.testing.assert_close(cells.detach()[:, j], reference)
        cells.sum().backward()
        torch.testing.assert_close(
            predictions.grad,
            torch.tensor([[2.0, 1.0, 0.0, 2.0], [2.0, 0.0, 0.0, 1.0]]),
        )

    def test_value_and_gradient_are_continuous_at_delta(self):
        import torch

        delta, step = 0.7, 1e-7
        predictions = torch.tensor(
            [[delta - step], [delta], [delta + step]], dtype=torch.float64, requires_grad=True
        )
        zeros, ones = torch.zeros(3, 1, dtype=torch.float64), torch.ones(3, 1, dtype=torch.float64)
        cells = sp.penalty_cells(
            predictions, zeros, ones, zeros, delta=torch.tensor([delta], dtype=torch.float64)
        )
        self.assertAlmostEqual(float(cells[1, 0].detach()), delta ** 2, places=12)
        values = cells.detach().ravel().numpy()
        np.testing.assert_allclose(values, delta ** 2, atol=1e-6)
        self.assertTrue(np.all(np.diff(values) > 0))
        cells.sum().backward()
        np.testing.assert_allclose(predictions.grad.ravel().numpy(), 2 * delta, atol=1e-6)

    def test_a_delta_above_every_excess_is_the_squared_penalty(self):
        import torch

        predictions, targets, weights, upper = self.batch()
        squared = sp.penalty_cells(predictions, targets, weights, upper)
        huber = sp.penalty_cells(
            predictions, targets, weights, upper, delta=torch.full((4,), 1e9)
        )
        self.assertTrue(torch.equal(squared, huber))

    def test_table_rejects_bad_delta_and_mixed_forms(self):
        import torch

        _, targets, weights, upper = self.batch()
        base = dict(
            strength=1.0, molecule_ids=["a", "b"], weights=weights.numpy(),
            upper=upper.numpy(), enzymes=ENZYMES,
        )
        for bad in ([1.0, 0.0, 1.0, 1.0], [1.0, float("nan"), 1.0, 1.0],
                    [1.0, float("inf"), 1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [1.0, 1.0]):
            with self.subTest(delta=bad), self.assertRaisesRegex(ValueError, "delta"):
                sp.ScreeningPenalty(**base, delta=np.array(bad))
        with self.assertRaisesRegex(ValueError, "shape"):
            sp.ScreeningPenalty(**base, lower=np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "lower.*delta|delta.*lower"):
            sp.ScreeningPenalty(**base, lower=np.zeros((2, 4)), delta=np.ones(4))
        predictions = self.batch()[0]
        with self.assertRaisesRegex(ValueError, "lower.*delta|delta.*lower"):
            sp.penalty_cells(
                predictions, targets, weights, upper,
                lower=torch.zeros_like(upper), delta=torch.ones(4),
            )

    def test_denominator_is_the_observed_cell_count(self):
        import torch

        _, targets, weights, upper = self.batch()
        penalty = sp.ScreeningPenalty(
            strength=1.0, molecule_ids=["a", "b"], weights=weights.numpy(),
            upper=upper.numpy(), delta=np.array(self.DELTA), enzymes=ENZYMES,
        )
        total = penalty.apply(
            torch.tensor(2.0), self.batch()[0], targets,
            torch.where(targets.isfinite(), 3.0, 0.0), torch.ones(2, 1), ["a", "b"],
        )
        self.assertAlmostEqual(float(total), 2.0 + 10.25 / 7, places=6)


# --------------------------------------------------------------------------- delta
class DeltaTests(unittest.TestCase):
    SCALE = np.array([0.5, 1.0, 2.0, 4.0])

    def bounds(self):
        low = np.zeros((5, 4))
        half = np.array([
            [0.1, 0.5, 0.3, 1.0], [0.2, 0.5, 0.3, 2.0], [0.3, 0.5, 0.3, 3.0],
            [0.4, 0.5, 0.9, 4.0], [99.0, 0.5, 0.9, 5.0],
        ])
        observed = np.ones((5, 4), dtype=bool)
        observed[4, 0] = False
        return low, 2 * half, observed

    def deltas(self, low, high, observed):
        return sp.standardized_deltas(low, high, observed, self.SCALE, ENZYMES)

    def test_median_half_width_of_observed_cells_in_scaler_units(self):
        low, high, observed = self.bounds()
        np.testing.assert_allclose(
            self.deltas(low, high, observed),
            np.array([0.25, 0.5, 0.3, 3.0]) / self.SCALE,
        )
        high[4, 0] = np.nan  # unobserved cells are ignored
        np.testing.assert_allclose(self.deltas(low, high, observed)[0], 0.25 / 0.5)

    def test_missing_non_finite_and_zero_delta_raise(self):
        low, high, observed = self.bounds()
        none = observed.copy()
        none[:, 1] = False
        with self.assertRaisesRegex(ValueError, "CYP2C9.*missing"):
            self.deltas(low, high, none)
        for value in (np.nan, np.inf):
            broken = high.copy()
            broken[0, 1] = value  # one bad cell is not skipped over
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "CYP2C9.*non-finite"):
                self.deltas(low, broken, observed)
        flat = high.copy()
        flat[:3, 1] = 0.0
        with self.assertRaisesRegex(ValueError, "CYP2C9.*zero"):
            self.deltas(low, flat, observed)
        inverted = high.copy()
        inverted[:, 1] = -1.0
        with self.assertRaisesRegex(ValueError, "CYP2C9"):
            self.deltas(low, inverted, observed)


# --------------------------------------------------------------------------- attach
class AttachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def test_table_holds_bounds_and_delta_in_fold_scaler_units(self):
        for form in FORMS:
            with self.subTest(form=form), tempfile.TemporaryDirectory() as tmp:
                state, train, _ = prefit()
                before = list(state["model"].state_dict())
                report = attach(state, train, tmp, form_arm(form))
                self.assertEqual(list(state["model"].state_dict()), before)
                self.assertEqual(report["bound_stop_reasons"], [])
                penalty, scaler = state["model"].screening_penalty, state["scaler"]
                names = [str(v) for v in train["Molecule_Name"]]
                rows = [penalty.index[n] for n in names]
                for j, task in enumerate(TASKS):
                    observed = train[task].notna().to_numpy()
                    columns = [("upper", "_conf_high")]
                    if form == "two_sided_squared":
                        columns.append(("lower", "_conf_low"))
                    for attribute, suffix in columns:
                        table = getattr(penalty, attribute)[rows, j].numpy()
                        native = train[task + suffix].to_numpy(dtype=np.float64)
                        np.testing.assert_allclose(
                            table[observed],
                            (native[observed] - scaler.mean_[j]) / scaler.scale_[j],
                        )
                if form == "two_sided_squared":
                    self.assertIsNone(penalty.delta)
                    continue
                self.assertIsNone(penalty.lower)
                expected = native_deltas(train, scaler.scale_)
                np.testing.assert_allclose(penalty.delta.numpy(), expected)
                saved = json.loads((Path(tmp) / sp.POPULATION_FILE).read_text())
                for j, (task, enzyme) in enumerate(zip(TASKS, ENZYMES)):
                    entry = saved["delta"][enzyme]
                    self.assertAlmostEqual(entry["standardized_delta"], expected[j])
                    self.assertEqual(entry["observed_cells"], int(train[task].notna().sum()))

    def test_delta_uses_every_observed_cell_not_only_screened_ones(self):
        screen = d024.default_screen_rows()
        dropped = set(d024.NAMES[6::2])
        partial = [r for r in screen if r["Molecule_Name"] not in dropped]
        deltas = []
        for rows in (screen, partial):
            with tempfile.TemporaryDirectory() as tmp:
                state, train, _ = prefit()
                attach(state, train, tmp, form_arm("upper_huber"), screen_rows=rows)
                deltas.append(state["model"].screening_penalty.delta.numpy())
        np.testing.assert_array_equal(deltas[0], deltas[1])

    def test_new_forms_use_exactly_the_rank_weights(self):
        files = []
        for arm in (d024.arm("rank"), *(form_arm(f) for f in FORMS)):
            with tempfile.TemporaryDirectory() as tmp:
                state, train, _ = prefit()
                attach(state, train, tmp, arm)
                files.append((Path(tmp) / sp.WEIGHTS_FILE).read_bytes())
                penalty = state["model"].screening_penalty
                files.append(penalty.weights.numpy().tobytes())
        self.assertEqual(files[0::2], [files[0]] * 3)
        self.assertEqual(files[1::2], [files[1]] * 3)

    def check_stop(self, form, change):
        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = prefit()
            broken = train.copy()
            change(broken)
            with self.assertRaises(ValueError):
                attach(state, broken, tmp, form_arm(form))
            self.assertTrue((Path(tmp) / sp.WEIGHTS_FILE).exists())
            saved = json.loads((Path(tmp) / sp.POPULATION_FILE).read_text())
            self.assertTrue(saved["bound_stop_reasons"])
            self.assertIn("CYP2C9", " ".join(saved["bound_stop_reasons"]))
            self.assertIsNone(state["model"].screening_penalty)

    def test_two_sided_stops_on_missing_or_incompatible_bounds(self):
        task = TASKS[1]

        def missing(frame):
            frame.loc[first_observed(frame, task), task + "_conf_low"] = np.nan

        def infinite(frame):
            frame.loc[first_observed(frame, task), task + "_conf_low"] = -np.inf

        def equal(frame):
            name = first_observed(frame, task)
            frame.loc[name, task + "_conf_low"] = frame.loc[name, task + "_conf_high"]

        def inverted(frame):
            name = first_observed(frame, task)
            frame.loc[name, task + "_conf_low"] = frame.loc[name, task + "_conf_high"] + 0.1

        for change in (missing, infinite, equal, inverted):
            with self.subTest(case=change.__name__):
                self.check_stop("two_sided_squared", change)

    def test_huber_stops_on_missing_non_finite_or_zero_delta(self):
        task = TASKS[1]

        def missing_bound(frame):
            frame.loc[first_observed(frame, task), task + "_conf_low"] = np.nan

        def no_observed_cells(frame):
            frame[task] = np.nan

        def zero(frame):
            frame[task + "_conf_low"] = frame[task + "_conf_high"]

        for change in (missing_bound, no_observed_cells, zero):
            with self.subTest(case=change.__name__):
                self.check_stop("upper_huber", change)

    def test_guard_d024_arms_do_not_read_the_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = prefit()
            changed = train.copy()
            for task in TASKS:
                changed[task + "_conf_low"] = np.nan
            attach(state, changed, tmp, d024.arm("rank"))
            self.assertIsNotNone(state["model"].screening_penalty)


# --------------------------------------------------------------------------- training
class TrainingStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def assert_same(self, first, second):
        import torch

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(set(first[1]), set(second[1]))
        for key, value in first[1].items():
            self.assertTrue(torch.equal(value, second[1][key]), key)

    def test_zero_lambda_equals_the_pre_penalty_loss_and_gradients(self):
        import t121_weighted_training as weighted

        old = committed_module(
            d024.PRE_EDIT_COMMIT, d024.PRE_EDIT_PATH, d024.PRE_EDIT_SHA256, "weighted_pre_edit"
        ).T121WeightedTrainingMPNN.training_step
        new = weighted.T121WeightedTrainingMPNN.training_step
        for form in FORMS:
            with self.subTest(form=form), tempfile.TemporaryDirectory() as tmp:
                state, train, _ = prefit()
                batch = next(iter(state["train_loader"]))
                reference = d024.ModelTests.step(old, state["model"], batch)
                arm = form_arm(form, **{"lambda": 0.0})
                self.assertEqual(arm.form, form)
                attach(state, train, tmp, arm)
                penalty = state["model"].screening_penalty
                extra = penalty.lower if form == "two_sided_squared" else penalty.delta
                self.assertIsNotNone(extra)  # the new form is really in the path
                self.assert_same(d024.ModelTests.step(new, state["model"], batch), reference)

    def test_inactive_extra_terms_reproduce_the_d024_rank_arm(self):
        import torch
        import t121_weighted_training as weighted

        old = committed_module(D024_COMMIT, D024_MODULE_PATH, D024_MODULE_SHA256, "penalty_d024")
        step = weighted.T121WeightedTrainingMPNN.training_step
        config = read_config(D024_CONFIGS / "penalty_rank.json")

        def remove_lower(penalty):
            penalty.lower = None

        def lower_far_below(penalty):
            penalty.lower = torch.full_like(penalty.lower, -1e6)

        def delta_far_above(penalty):
            penalty.delta = torch.full_like(penalty.delta, 1e9)

        cases = (
            ("two_sided_squared", remove_lower),
            ("two_sided_squared", lower_far_below),
            ("upper_huber", delta_far_above),
        )
        for form, change in cases:
            with self.subTest(case=change.__name__), tempfile.TemporaryDirectory() as a, \
                    tempfile.TemporaryDirectory() as b:
                state, train, _ = prefit()
                batch = next(iter(state["train_loader"]))
                attach(state, train, a, old.PenaltyArm(config, "test"), module=old)
                reference = d024.ModelTests.step(step, state["model"], batch)
                attach(state, train, b, form_arm(form))
                change(state["model"].screening_penalty)
                self.assert_same(d024.ModelTests.step(step, state["model"], batch), reference)

    def test_unit_lambda_adds_exactly_the_defined_penalty(self):
        import torch
        import t121_weighted_training as weighted

        step = weighted.T121WeightedTrainingMPNN.training_step
        for form in FORMS:
            with self.subTest(form=form), tempfile.TemporaryDirectory() as tmp:
                state, train, _ = prefit()
                batch = next(iter(state["train_loader"]))
                base, base_grads = d024.ModelTests.step(step, state["model"], batch)
                attach(state, train, tmp, form_arm(form))
                total, grads = d024.ModelTests.step(step, state["model"], batch)

                bmg, V_d, X_d = batch["chemprop_batch"][:3]
                model = copy.deepcopy(state["model"])
                model.train()
                torch.manual_seed(11)
                with torch.no_grad():
                    predictions = model.predictor.train_step(model.fingerprint(bmg, V_d, X_d))
                with (Path(tmp) / sp.WEIGHTS_FILE).open(newline="") as handle:
                    weights = {(r["Molecule_Name"], r["enzyme"]): float(r["weight"])
                               for r in csv.DictReader(handle)}
                scaler = state["scaler"]
                delta = native_deltas(train, scaler.scale_)
                upper_sum, lower_sum, observed = 0.0, 0.0, 0
                for i, molecule in enumerate(batch["molecule_ids"]):
                    for j, (task, enzyme) in enumerate(zip(TASKS, ENZYMES)):
                        if not np.isfinite(train.loc[molecule, task]):
                            continue
                        observed += 1
                        weight = weights[(molecule, enzyme)]
                        value = float(predictions[i, j])
                        upper = (train.loc[molecule, task + "_conf_high"] - scaler.mean_[j]) / scaler.scale_[j]
                        lower = (train.loc[molecule, task + "_conf_low"] - scaler.mean_[j]) / scaler.scale_[j]
                        excess = max(value - upper, 0.0)
                        if form == "two_sided_squared":
                            upper_sum += weight * excess ** 2
                            lower_sum += weight * max(lower - value, 0.0) ** 2
                        elif excess <= delta[j]:
                            upper_sum += weight * excess ** 2
                        else:
                            upper_sum += weight * (2 * delta[j] * excess - delta[j] ** 2)
                self.assertGreater(upper_sum, 0)
                if form == "two_sided_squared":
                    self.assertGreater(lower_sum, 0)
                self.assertAlmostEqual(
                    float(total - base), (upper_sum + lower_sum) / observed, places=5
                )
                self.assertTrue(any(not torch.equal(grads[k], base_grads[k]) for k in grads))


# --------------------------------------------------------------------------- preview
class PreviewTests(unittest.TestCase):
    def preview(self, frame, directory):
        arms = [d024.arm("rank"), *(form_arm(f) for f in FORMS)]
        root = Path(directory)
        manifest = root / "rep_01_fold_00.json"
        manifest.write_text(json.dumps(
            {"repetition": 1, "fold": 0, "training_seeds": [d024.SEED]}
        ))
        return sp.preview(
            arms=arms, screens={a.arm: d024.parsed(penalty_arm=a) for a in arms},
            dataframe=None, split_dataframe=None, manifests=[manifest],
            route_fold=lambda **kwargs: {"training_dataframe": frame},
            output_dir=root / "out",
        )

    def test_clean_fold_passes_with_identical_weights_for_all_rank_based_arms(self):
        train, _ = d024.frames()
        with tempfile.TemporaryDirectory() as tmp:
            report = self.preview(train, tmp)
        self.assertEqual(report["stopped_arms"], [])
        self.assertFalse(report["stop"])
        self.assertEqual(len({f["weights_sha256"] for f in report["folds"]}), 1)

    def test_weights_are_compared_with_a_reference_preview(self):
        train, _ = d024.frames()
        with tempfile.TemporaryDirectory() as tmp:
            report = self.preview(train, tmp)
        reference = copy.deepcopy(report)
        self.assertEqual(sp.weight_mismatches(report, reference, "rank"), [])
        for entry in reference["folds"]:
            if entry["arm"] == "rank":
                entry["weights_sha256"] = "0" * 64
        mismatches = sp.weight_mismatches(report, reference, "rank")
        self.assertEqual(sorted(m["arm"] for m in mismatches), ["huber", "rank", "twosided"])
        reference["folds"] = [f for f in report["folds"] if f["arm"] != "rank"]
        with self.assertRaisesRegex(ValueError, "no folds for arm"):
            sp.weight_mismatches(report, reference, "rank")

    def test_each_bound_failure_stops_only_the_affected_arms(self):
        task = TASKS[1]

        def no_observed_cells(frame):
            frame[task] = np.nan

        def one_inverted_cell(frame):
            name = first_observed(frame, task)
            frame.loc[name, task + "_conf_low"] = frame.loc[name, task + "_conf_high"] + 0.1

        def one_missing_lower(frame):
            frame.loc[first_observed(frame, task), task + "_conf_low"] = np.nan

        cases = (
            (no_observed_cells, ["huber"]),
            (one_inverted_cell, ["twosided"]),
            (one_missing_lower, ["huber", "twosided"]),
        )
        train, _ = d024.frames()
        for change, stopped in cases:
            with self.subTest(case=change.__name__), tempfile.TemporaryDirectory() as tmp:
                broken = train.copy()
                change(broken)
                report = self.preview(broken, tmp)
                self.assertTrue(report["stop"])
                self.assertEqual(report["stopped_arms"], stopped)
                self.assertEqual(sorted({s["arm"] for s in report["bound_stops"]}), stopped)
                self.assertEqual({s["fold"] for s in report["bound_stops"]}, {"rep_01_fold_00"})


# --------------------------------------------------------------------------- aggregation
NEW_EVALUATION = {
    **d024_agg.CONFIG, "arm_labels": ARM_LABELS, "tie_preference": "twosided",
}


class RuleTests(unittest.TestCase):
    def selected(self, twosided, huber, **overrides):
        return agg.decide(
            {"unchanged": d024_agg.metrics(), "twosided": twosided, "huber": huber},
            {**NEW_EVALUATION, **overrides},
        )["selected"]

    def test_lower_macro_wins_ties_follow_the_config_and_no_pass_keeps_unchanged(self):
        base = d024_agg.metrics()
        better = d024_agg.metrics(0.69, 0.49, 0.49)
        best = d024_agg.metrics(0.68, 0.49, 0.49)
        self.assertEqual(self.selected(better, best), "huber")
        self.assertEqual(self.selected(best, better), "twosided")
        for preference in ("twosided", "huber"):
            self.assertEqual(
                self.selected(better, copy.deepcopy(better), tie_preference=preference), preference
            )
        self.assertEqual(self.selected(base, copy.deepcopy(base)), "unchanged")

    def test_one_enzyme_failing_a_tail_clause_blocks_the_arm(self):
        best = d024_agg.metrics(0.68, 0.49, 0.49)
        best["tails"]["CYP2D6"]["over"]["mae"] = 0.5  # not lower for one enzyme
        better = d024_agg.metrics(0.69, 0.49, 0.49)
        self.assertEqual(self.selected(best, better), "huber")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_path = self.root / "evaluation.json"
        self.config_path.write_text(json.dumps({**NEW_EVALUATION, "baseline_o150": 0.70}))
        for arm, value, shift in (
            ("unchanged", 0.70, 0.0), ("twosided", 0.69, 0.1), ("huber", 0.71, 0.05),
        ):
            self.write_arm(arm, value, shift)
        self.sums = self.root / "sums.txt"
        self.sums.write_text("".join(
            f"{d024_agg.sha(p.read_bytes())}  {p.name}\n"
            for p in sorted((self.root / "pred_unchanged").iterdir())
        ))

    def write_arm(self, arm, value, shift):
        predictions = self.root / f"pred_{arm}"
        predictions.mkdir()
        rows = d024_agg.prediction_rows(shift)
        for index in range(150):
            cell = self.root / f"cells_{arm}" / f"cell_{index:03d}"
            cell.mkdir(parents=True)
            checkpoint = d024_agg.sha(f"{arm}{index}".encode())
            text = json.dumps({
                "cell_index": index, "repetition": 1 + index // 30, "fold": (index // 3) % 10,
                "seed": d024_agg.SEEDS[index % 3], "status": "completed",
                "split_manifest_sha256": "m", "screen_checkpoint_sha256": "s" * 64,
                "macro_st_rae": value, "arm": ARM_LABELS[arm], "checkpoint_sha256": checkpoint,
                "best_validation": {
                    "macro_st_rae": value,
                    "per_task": {i + NEW_EVALUATION["task_suffix"]: value for i in ENZYMES},
                },
            }).encode()
            (cell / "result.json").write_bytes(text)
            (cell / "COMPLETED.json").write_text(
                json.dumps({"status": "completed", "result_sha256": d024_agg.sha(text)})
            )
            mine = [(k, v) for k, v in rows.items() if k[0] == index]
            with (predictions / f"predictions_cell_{index:03d}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["cell_index", "isoform", "Molecule_Name",
                                 "y_true", "y_pred", "conf_low", "conf_high"])
                for (_, isoform, molecule), values in mine:
                    writer.writerow([index, isoform, molecule, *map(repr, values)])
            (predictions / f"reproduction_cell_{index:03d}.json").write_text(json.dumps({
                "cell_index": index, "status": "reproduced",
                "checkpoint_sha256": checkpoint, "prediction_rows": len(mine),
            }))

    def run_main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            agg.main([str(a) for a in args])

    def evaluate(self, arm_names):
        cohorts = self.root / "cohorts.json"
        coverage = self.root / "coverage.json"
        baseline = ("--baseline-predictions", self.root / "pred_unchanged",
                    "--sha256sums", self.sums, "--config", self.config_path)
        self.run_main("cohorts", *baseline, "--output", cohorts)
        d024_agg.screen_tar(self.root / "screen.tar.gz")
        self.run_main("coverage", *baseline,
                      "--coverage-config", d024_agg.COVERAGE_CONFIG_PATH,
                      "--penalty-config", d024_agg.PENALTY_CONFIG_PATH,
                      "--screen-tar", self.root / "screen.tar.gz", "--output", coverage)
        arguments = ["evaluate", "--config", self.config_path,
                     "--cohorts", cohorts, "--cohorts-sha256", d024_agg.sha(cohorts.read_bytes()),
                     "--coverage", coverage, "--coverage-sha256", d024_agg.sha(coverage.read_bytes()),
                     "--output", self.root / "report.json"]
        for name, directory in arm_names.items():
            arguments += ["--cells", f"{name}={self.root / f'cells_{directory}'}",
                          "--predictions", f"{name}={self.root / f'pred_{directory}'}"]
        self.run_main(*arguments)
        return json.loads((self.root / "report.json").read_text())

    def test_evaluate_reads_arm_names_from_the_config(self):
        report = self.evaluate({a: a for a in ARM_LABELS})
        self.assertEqual(set(report["metrics"]), set(ARM_LABELS))
        self.assertEqual(set(report["decision"]["arms"]), {"twosided", "huber"})
        self.assertEqual(report["decision"]["selected"], "twosided")
        self.assertEqual(set(report["coverage_groups"]["groups"]), set(ARM_LABELS))
        self.assertEqual(
            set(report["differences_from_unchanged"]), {"twosided", "huber"}
        )

    def test_arm_names_outside_the_config_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Bad or duplicate|must name exactly"):
            self.evaluate({"unchanged": "unchanged", "q3": "twosided", "rank": "huber"})


if __name__ == "__main__":
    unittest.main()
