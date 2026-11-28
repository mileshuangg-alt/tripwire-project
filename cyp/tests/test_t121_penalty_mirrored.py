"""D030 label-rank mirrored band penalty.

    P = sum over observed training cells of
            2(1 - r) * relu(y - U)^2 + 2r * relu(L - y)^2,  / observed cells
r = (average ascending rank of the measured label - 1) / (n - 1) per enzyme over
the labelled training-fold molecules.

The D029 two-sided arm is the reference: with the D029 rank weights on both
sides, the upper term, the lower term and their sum must equal it exactly, in
loss and gradients. The reference module is the pinned D029 source.

Synthetic names, structures, labels and screening values only.
"""
from pathlib import Path
import copy
import json
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_t121_penalty_forms as forms  # noqa: E402  (also sets the import paths)

d024, agg, sp = forms.d024, forms.agg, forms.sp
TASKS, ENZYMES = d024.TASKS, d024.ENZYMES

D029_CONFIGS = d024.REPO / "cyp/artifacts/d029_penalty"
CONFIGS = d024.REPO / "cyp/artifacts/d030_penalty"
LABEL = "updated_t121_label_rank_mirrored_penalty_from_screen_checkpoint"
# The D029 source version: the penalty module as packaged and run (uncommitted
# when D029 ran), inside t121_d029_package_20261126.tar.gz (SHA-256 ef84a802...).
D029_SOURCE = (d024.REPO / "t121_d029_build_20261126/t121_d029_package_20261126"
               / "cyp/scripts/t121_screening_penalty.py")
D029_SOURCE_SHA256 = "b6d72307fab389af6cbc9819b2b384998c588373553797380fd2fb9f6f9b0c34"
D029_CONFIG_SHA256 = {
    "penalty_rank_twosided.json": "347d48dac367db193152ffba83a32b8413ed358a681ccff88bb251818530232b",
    "penalty_rank_huber.json": "8f7dce50c7f980bdfd44414fe2e38c0119ac5606c081e62676481fc98b25a6af",
    "evaluation.json": "8ed9201250e3094bd1eed31fc791dd7cd7108f01e939c0c50609b83e57287553",
}
FAR = 1e6  # a bound this far away switches one side off in the reference module


def mirrored(**overrides):
    config = json.loads((CONFIGS / "penalty_label_mirrored.json").read_text())
    config.update(overrides)
    return sp.PenaltyArm(config, "test")


def d029_module():
    if not D029_SOURCE.is_file():
        if (d024.REPO / "screen_manifest_v2.json").is_file():  # repository checkout
            raise AssertionError(f"Pinned D029 source is missing: {D029_SOURCE}")
        raise unittest.SkipTest("pinned D029 build not present (deployed package)")
    source = D029_SOURCE.read_bytes()
    if sp.sha256_bytes(source) != D029_SOURCE_SHA256:
        raise AssertionError("D029 reference source does not have its pinned SHA-256.")
    module = types.ModuleType("penalty_d029")
    exec(compile(source, str(D029_SOURCE), "exec"), module.__dict__)
    return module


def independent_weights(train):
    """(upper, lower) tables by molecule and task, from pandas average ranks."""
    result = {}
    for task in TASKS:
        values = train[task].dropna()
        r = (values.rank(method="average") - 1.0) / (len(values) - 1.0)
        for name in values.index:
            result[(name, task)] = (2.0 * (1.0 - r[name]), 2.0 * r[name])
    return result


# --------------------------------------------------------------------------- config
class ConfigTests(unittest.TestCase):
    def test_frozen_arm_config(self):
        path = CONFIGS / "penalty_label_mirrored.json"
        config = json.loads(path.read_text())
        reference = json.loads((D029_CONFIGS / "penalty_rank_twosided.json").read_text())
        self.assertEqual(set(config), {
            "decision", "arm", "arm_label", "rule", "form", "lambda", "minimum_population", "tasks"})
        for key in ("form", "lambda", "minimum_population", "tasks"):
            self.assertEqual(config[key], reference[key], key)
        self.assertEqual((config["arm"], config["arm_label"], config["rule"]),
                         ("mirrored", LABEL, "label_rank_mirrored"))
        loaded = sp.PenaltyArm.load(path, expected_sha256=sp.sha256_file(path))
        self.assertFalse(loaded.uses_screening)
        self.assertTrue(forms.form_arm("two_sided_squared").uses_screening)

    def test_frozen_evaluation_config_changes_only_the_arms(self):
        own = {"decision", "arm_labels", "tie_preference"}
        old = json.loads((D029_CONFIGS / "evaluation.json").read_text())
        new = agg.load_config(CONFIGS / "evaluation.json")
        self.assertEqual({k: v for k, v in new.items() if k not in own},
                         {k: v for k, v in old.items() if k not in own})
        self.assertEqual(new["arm_labels"],
                         {"unchanged": old["arm_labels"]["unchanged"], "mirrored": LABEL})
        self.assertEqual(agg.arm_names(new), ("unchanged", "mirrored"))

    def test_guard_earlier_configs_are_byte_identical(self):
        for name, digest in D029_CONFIG_SHA256.items():
            self.assertEqual(sp.sha256_file(D029_CONFIGS / name), digest, name)
        for name, digest in forms.D024_CONFIG_SHA256.items():
            self.assertEqual(sp.sha256_file(forms.D024_CONFIGS / name), digest, name)

    def test_label_rule_refuses_other_forms_and_screening_keys(self):
        with self.assertRaisesRegex(ValueError, "two_sided_squared"):
            mirrored(form="upper_huber")
        with self.assertRaisesRegex(ValueError, "no screening inputs"):
            mirrored(screen={})


# --------------------------------------------------------------------------- weights
class WeightTests(unittest.TestCase):
    def test_hand_values_with_ties(self):
        r, a, b, summary = sp.label_rank_weights([1.0, 3.0, 2.0, 3.0], mirrored())
        np.testing.assert_allclose(r, [0, 2.5 / 3, 1 / 3, 2.5 / 3])
        np.testing.assert_allclose(a, [2, 1 / 3, 4 / 3, 1 / 3])
        np.testing.assert_allclose(b, [0, 5 / 3, 2 / 3, 5 / 3])
        self.assertTrue(summary["active"])

    def test_mean_one_and_sum_two_with_many_ties(self):
        values = np.round(np.random.default_rng(3).normal(5, 1, 501), 1)
        _, a, b, summary = sp.label_rank_weights(values, mirrored())
        self.assertAlmostEqual(np.mean(a), 1.0, places=12)
        self.assertAlmostEqual(np.mean(b), 1.0, places=12)
        np.testing.assert_allclose(np.add(a, b), 2.0, atol=1e-12)
        self.assertLessEqual(summary["max_abs_sum_minus_two"], sp.IDENTITY_TOLERANCE)

    def test_all_equal_labels_stay_active_with_unit_weights(self):
        r, a, b, summary = sp.label_rank_weights([4.2] * 6, mirrored())
        self.assertEqual((r, a, b), ([0.5] * 6, [1.0] * 6, [1.0] * 6))
        self.assertEqual((summary["active"], summary["reason"]), (True, None))

    def test_fewer_than_two_labels_switch_the_enzyme_off(self):
        r, a, b, summary = sp.label_rank_weights([4.2], mirrored())
        self.assertEqual((r, a, b), ([None], [0.0], [0.0]))
        self.assertEqual((summary["active"], summary["reason"]), (False, "fewer_than_two_eligible"))
        train, _ = d024.frames()
        train = train.copy()
        train[TASKS[1]] = np.nan
        rows, built = sp.build_label_population(train, mirrored())
        self.assertEqual(built["enzymes"]["CYP2C9"]["reason"], "no_labelled_training_molecules")
        self.assertFalse([x for x in rows if x["enzyme"] == "CYP2C9"])

    def test_ranked_set_is_exactly_the_observed_training_cells(self):
        train, _ = d024.frames()
        rows, built = sp.build_label_population(train, mirrored())
        observed = {(n, e) for t, e in zip(TASKS, ENZYMES) for n in train[t].dropna().index}
        self.assertEqual({(x["Molecule_Name"], x["enzyme"]) for x in rows}, observed)
        self.assertEqual(len(rows), len(observed))
        expected = independent_weights(train)
        task_of = dict(zip(ENZYMES, TASKS))
        for x in rows:
            a, b = expected[(x["Molecule_Name"], task_of[x["enzyme"]])]
            self.assertAlmostEqual(x["upper_weight"], a)
            self.assertAlmostEqual(x["lower_weight"], b)
        for task, enzyme in zip(TASKS, ENZYMES):
            entry = built["enzymes"][enzyme]
            self.assertEqual(entry["population"], int(train[task].notna().sum()))
            self.assertAlmostEqual(entry["mean_upper_weight"], 1.0)
            self.assertAlmostEqual(entry["mean_lower_weight"], 1.0)

    def test_weights_do_not_depend_on_screening_inputs(self):
        files = []
        for rows in (d024.default_screen_rows(), d024.default_screen_rows()[::3]):
            with tempfile.TemporaryDirectory() as tmp:
                state, train, _ = forms.prefit()
                report = forms.attach(state, train, tmp, mirrored(), screen_rows=rows)
                self.assertNotIn("screen_input_sha256", report)
                files.append((Path(tmp) / sp.WEIGHTS_FILE).read_bytes())
        self.assertEqual(files[0], files[1])


# --------------------------------------------------------------------------- loss
class LossTests(unittest.TestCase):
    def batch(self):
        import torch

        nan = float("nan")
        predictions = torch.tensor(
            [[2.0, -2.0, 0.0, 2.0], [-2.0, 7.0, 2.0, 0.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, nan, 0.0, 0.0]])
        upper = torch.full((2, 4), 0.5)
        lower = torch.full((2, 4), -0.5)
        upper[1, 1] = lower[1, 1] = nan
        a = torch.tensor([[2.0, 0.0, 1.0, 0.5], [0.0, 9.0, 2.0, 1.0]])
        b = torch.tensor([[0.0, 2.0, 1.0, 1.5], [2.0, 9.0, 0.0, 1.0]])
        return predictions, targets, a, b, upper, lower

    def test_each_side_has_its_own_weights_and_gradient_sign(self):
        import torch

        predictions, targets, a, b, upper, lower = self.batch()
        up, low = sp.side_cells(predictions, targets, a, b, upper, lower)
        torch.testing.assert_close(
            up.detach(), torch.tensor([[4.5, 0.0, 0.0, 1.125], [0.0, 0.0, 4.5, 0.0]]))
        torch.testing.assert_close(
            low.detach(), torch.tensor([[0.0, 4.5, 0.0, 0.0], [4.5, 0.0, 0.0, 0.0]]))
        (up + low).sum().backward()
        torch.testing.assert_close(
            predictions.grad,
            torch.tensor([[6.0, -6.0, 0.0, 1.5], [-6.0, 0.0, 6.0, 0.0]]))

    def test_denominator_side_sums_and_active_counts(self):
        import torch

        predictions, targets, a, b, upper, lower = self.batch()
        penalty = sp.ScreeningPenalty(
            strength=1.0, molecule_ids=["x", "y"], weights=a.numpy(), lower_weights=b.numpy(),
            upper=upper.numpy(), lower=lower.numpy(), enzymes=ENZYMES,
        )
        total = penalty.apply(
            torch.tensor(2.0), predictions, targets,
            torch.where(targets.isfinite(), 3.0, 0.0), torch.ones(2, 1), ["x", "y"],
        )
        self.assertAlmostEqual(float(total.detach()), 2.0 + 19.125 / 7, places=5)
        record = penalty.epoch_record(0)
        self.assertAlmostEqual(record["penalty_unscaled"], 19.125 / 7, places=5)
        entries = record["per_enzyme"]
        for entry in entries.values():
            self.assertAlmostEqual(
                entry["upper_penalty_sum"] + entry["lower_penalty_sum"], entry["penalty_sum"])
            self.assertEqual(
                entry["upper_active_cells"] + entry["lower_active_cells"], entry["active_cells"])
        self.assertEqual(
            (entries["CYP1A2"]["upper_active_cells"], entries["CYP1A2"]["lower_active_cells"]), (1, 1))
        self.assertEqual(entries["CYP2C9"]["observed_cells"], 1)

    def test_table_refuses_lower_weights_without_a_lower_bound(self):
        _, _, a, b, upper, lower = self.batch()
        base = dict(strength=1.0, molecule_ids=["x", "y"], weights=a.numpy(),
                    upper=upper.numpy(), enzymes=ENZYMES, lower_weights=b.numpy())
        with self.assertRaisesRegex(ValueError, "lower weights"):
            sp.ScreeningPenalty(**base)
        with self.assertRaisesRegex(ValueError, "lower weights"):
            sp.ScreeningPenalty(**base, delta=np.ones(4))

    def test_side_terms_equal_the_pinned_d029_terms_cell_by_cell(self):
        import torch

        old = d029_module()
        generator = torch.Generator().manual_seed(5)
        draw = lambda: torch.randn(64, 4, generator=generator)  # noqa: E731
        values, w = draw(), draw().abs()
        upper = draw()
        lower = upper - draw().abs() - 0.01
        targets = torch.zeros(64, 4)
        targets[draw() > 1.0] = float("nan")
        far = torch.full_like(upper, FAR)
        zero = torch.zeros_like(w)
        cases = {
            "upper": ((w, zero), lambda p: old.penalty_cells(p, targets, w, upper)),
            "lower": ((zero, w), lambda p: old.penalty_cells(p, targets, w, far, lower=lower)),
            "both": ((w, w), lambda p: old.penalty_cells(p, targets, w, upper, lower=lower)),
        }
        for name, ((a, b), reference) in cases.items():
            with self.subTest(term=name):
                mine, theirs = values.clone().requires_grad_(), values.clone().requires_grad_()
                up, low = sp.side_cells(mine, targets, a, b, upper, lower)
                expected = reference(theirs)
                self.assertGreater(float(expected.sum().detach()), 0)
                self.assertTrue(torch.equal(up + low, expected))
                if name != "both":
                    self.assertEqual(float((low if name == "upper" else up).abs().sum().detach()), 0.0)
                (up + low).sum().backward()
                expected.sum().backward()
                self.assertTrue(torch.equal(mine.grad, theirs.grad))


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

    def test_upper_lower_and_both_equal_d029_loss_and_gradients(self):
        import torch
        import t121_weighted_training as weighted

        old = d029_module()
        step = weighted.T121WeightedTrainingMPNN.training_step
        config = json.loads((D029_CONFIGS / "penalty_rank_twosided.json").read_text())

        def upper_only(penalty):
            penalty.lower = None

        def lower_only(penalty):
            penalty.upper = torch.full_like(penalty.upper, FAR)

        def both(penalty):
            pass

        for term, change, sides in (
            ("upper", upper_only, (1.0, 0.0)), ("lower", lower_only, (0.0, 1.0)),
            ("both", both, (1.0, 1.0)),
        ):
            with self.subTest(term=term), tempfile.TemporaryDirectory() as a, \
                    tempfile.TemporaryDirectory() as b:
                state, train, _ = forms.prefit()
                batch = next(iter(state["train_loader"]))
                base = d024.ModelTests.step(step, state["model"], batch)
                forms.attach(state, train, a, old.PenaltyArm(config, "test"), module=old)
                reference_table = state["model"].screening_penalty
                change(reference_table)
                reference = d024.ModelTests.step(step, state["model"], batch)
                self.assertFalse(torch.equal(reference[0], base[0]))  # the term is active

                forms.attach(state, train, b, mirrored())
                table = state["model"].screening_penalty
                self.assertEqual(table.index, reference_table.index)
                # D029 rank weights on the chosen sides, D030 code path
                source = reference_table.weights
                table.weights = source * sides[0]
                table.lower_weights = source * sides[1]
                self.assert_same(d024.ModelTests.step(step, state["model"], batch), reference)

    def test_zero_lambda_equals_the_pre_penalty_loss_and_gradients(self):
        import t121_weighted_training as weighted

        old = forms.committed_module(
            d024.PRE_EDIT_COMMIT, d024.PRE_EDIT_PATH, d024.PRE_EDIT_SHA256, "weighted_pre_edit"
        ).T121WeightedTrainingMPNN.training_step
        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = forms.prefit()
            batch = next(iter(state["train_loader"]))
            reference = d024.ModelTests.step(old, state["model"], batch)
            forms.attach(state, train, tmp, mirrored(**{"lambda": 0.0}))
            self.assertIsNotNone(state["model"].screening_penalty.lower_weights)
            self.assert_same(
                d024.ModelTests.step(weighted.T121WeightedTrainingMPNN.training_step,
                                     state["model"], batch), reference)

    def test_unit_lambda_adds_exactly_the_defined_penalty(self):
        import torch
        import t121_weighted_training as weighted

        step = weighted.T121WeightedTrainingMPNN.training_step
        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = forms.prefit()
            batch = next(iter(state["train_loader"]))
            base, base_grads = d024.ModelTests.step(step, state["model"], batch)
            forms.attach(state, train, tmp, mirrored())
            total, grads = d024.ModelTests.step(step, state["model"], batch)

            bmg, V_d, X_d = batch["chemprop_batch"][:3]
            model = copy.deepcopy(state["model"])
            model.train()
            torch.manual_seed(11)
            with torch.no_grad():
                predictions = model.predictor.train_step(model.fingerprint(bmg, V_d, X_d))
            weights, scaler = independent_weights(train), state["scaler"]
            upper_sum = lower_sum = 0.0
            observed = 0
            for i, molecule in enumerate(batch["molecule_ids"]):
                for j, task in enumerate(TASKS):
                    if not np.isfinite(train.loc[molecule, task]):
                        continue
                    observed += 1
                    a, b = weights[(molecule, task)]
                    value = float(predictions[i, j])
                    high = (train.loc[molecule, task + "_conf_high"] - scaler.mean_[j]) / scaler.scale_[j]
                    low = (train.loc[molecule, task + "_conf_low"] - scaler.mean_[j]) / scaler.scale_[j]
                    upper_sum += a * max(value - high, 0.0) ** 2
                    lower_sum += b * max(low - value, 0.0) ** 2
            self.assertGreater(upper_sum, 0)
            self.assertGreater(lower_sum, 0)
            self.assertAlmostEqual(float(total - base), (upper_sum + lower_sum) / observed, places=5)
            self.assertTrue(any(not torch.equal(grads[k], base_grads[k]) for k in grads))

    def test_one_fit_logs_both_sides_every_epoch(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = forms.prefit()
            forms.attach(state, train, tmp, mirrored())
            before = copy.deepcopy(state["model"].state_dict())
            d024.ModelTests.fit(None, state, tmp)
            after = state["model"].state_dict()
            self.assertEqual(list(after), list(before))
            self.assertTrue(any(not torch.equal(before[k], after[k].cpu()) for k in before))
            records = [json.loads(x) for x in (Path(tmp) / sp.HISTORY_FILE).read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records], [0, 1])
            for record in records:
                for entry in record["per_enzyme"].values():
                    self.assertAlmostEqual(
                        entry["upper_penalty_sum"] + entry["lower_penalty_sum"],
                        entry["penalty_sum"], places=6)


# --------------------------------------------------------------------------- attach and preview
class AttachTests(unittest.TestCase):
    def test_table_bounds_weights_and_model_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, train, _ = forms.prefit()
            before = list(state["model"].state_dict())
            report = forms.attach(state, train, tmp, mirrored())
            self.assertEqual(list(state["model"].state_dict()), before)
            self.assertEqual(report["bound_stop_reasons"], [])
            header = (Path(tmp) / sp.WEIGHTS_FILE).read_text().splitlines()[0]
            self.assertEqual(header, ",".join(sp.LABEL_WEIGHT_COLUMNS))
            penalty, scaler = state["model"].screening_penalty, state["scaler"]
            rows = [penalty.index[str(n)] for n in train["Molecule_Name"]]
            expected = independent_weights(train)
            for j, task in enumerate(TASKS):
                observed = train[task].notna().to_numpy()
                for attribute, suffix in (("upper", "_conf_high"), ("lower", "_conf_low")):
                    table = getattr(penalty, attribute)[rows, j].numpy()
                    native = train[task + suffix].to_numpy(dtype=np.float64)
                    np.testing.assert_allclose(
                        table[observed], (native[observed] - scaler.mean_[j]) / scaler.scale_[j])
                a = penalty.weights[rows, j].numpy()
                b = penalty.lower_weights[rows, j].numpy()
                np.testing.assert_allclose(a[observed] + b[observed], 2.0)
                np.testing.assert_array_equal(a[~observed] + b[~observed], 0.0)
                for k, name in enumerate(train.index):
                    if observed[k]:
                        self.assertAlmostEqual(a[k], expected[(name, task)][0])

    def test_missing_or_incompatible_bounds_stop_the_arm(self):
        task = TASKS[1]
        for change in ("missing", "inverted"):
            with self.subTest(case=change), tempfile.TemporaryDirectory() as tmp:
                state, train, _ = forms.prefit()
                broken = train.copy()
                name = forms.first_observed(broken, task)
                broken.loc[name, task + "_conf_low"] = (
                    np.nan if change == "missing" else broken.loc[name, task + "_conf_high"] + 0.1)
                with self.assertRaisesRegex(ValueError, "CYP2C9"):
                    forms.attach(state, broken, tmp, mirrored())
                self.assertTrue((Path(tmp) / sp.POPULATION_FILE).exists())
                self.assertIsNone(state["model"].screening_penalty)


class PreviewTests(unittest.TestCase):
    def preview(self, frame, directory):
        root = Path(directory)
        manifest = root / "rep_01_fold_00.json"
        manifest.write_text(json.dumps(
            {"repetition": 1, "fold": 0, "training_seeds": [d024.SEED]}))
        return sp.preview(
            arms=[mirrored()], screens={"mirrored": None}, dataframe=None, split_dataframe=None,
            manifests=[manifest], route_fold=lambda **kwargs: {"training_dataframe": frame},
            output_dir=root / "out",
        )

    def test_clean_and_all_equal_folds_do_not_stop(self):
        train, _ = d024.frames()
        flat = train.copy()
        observed = flat[TASKS[2]].notna()
        flat.loc[observed, TASKS[2]] = 5.0
        for frame in (train, flat):
            with tempfile.TemporaryDirectory() as tmp:
                report = self.preview(frame, tmp)
            self.assertEqual((report["stop"], report["disabled"]), (False, []))
            self.assertEqual(report["weight_identity_failures"], [])
            self.assertIsNone(report["screen_input_sha256"])
            self.assertIn("CYP2D6", sp.preview_table(report))

    def test_an_enzyme_with_one_label_is_reported(self):
        train, _ = d024.frames()
        broken = train.copy()
        keep = forms.first_observed(broken, TASKS[1])
        broken.loc[broken.index != keep, TASKS[1]] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            report = self.preview(broken, tmp)
        self.assertTrue(report["stop"])
        self.assertEqual([(d["enzyme"], d["reason"]) for d in report["disabled"]],
                         [("CYP2C9", "fewer_than_two_eligible")])


class RuleTests(unittest.TestCase):
    def test_single_arm_is_selected_only_if_all_four_clauses_pass(self):
        config = agg.load_config(CONFIGS / "evaluation.json")
        base = forms.d024_agg.metrics()
        passing = forms.d024_agg.metrics(0.69, 0.49, 0.49)
        failing = copy.deepcopy(passing)
        failing["tails"]["CYP3A4"]["over"]["mae"] = 0.5
        for arm, selected in ((passing, "mirrored"), (failing, "unchanged")):
            self.assertEqual(
                agg.decide({"unchanged": base, "mirrored": arm}, config)["selected"], selected)


if __name__ == "__main__":
    unittest.main()
