"""Numerical qualification against the installed Chemprop stock criterion."""
import ast
from pathlib import Path
import unittest

import numpy as np
import torch
from chemprop import nn

SOURCE = Path(__file__).resolve().parents[2] / "cyp/scripts/cyp002_model.py"
NAMES = {
    "t121_raw_observation_weights",
    "t121_weighted_training_criterion",
}

# Load the exact production function bodies without importing unrelated
# model construction, validation scoring, or repository reference modules.
tree = ast.parse(SOURCE.read_text())
nodes = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in NAMES
]
if {node.name for node in nodes} != NAMES:
    raise RuntimeError("Required production weighting functions are missing.")
namespace = {"torch": torch}
exec(
    compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"),
    namespace,
)
raw_weights = namespace["t121_raw_observation_weights"]
weighted_loss = namespace["t121_weighted_training_criterion"]


class TestT121TrainingWeights(unittest.TestCase):
    def test_curve_fit_formula_and_missing_labels(self):
        result = raw_weights(
            [[4.0, np.nan], [5.0, 2.0]],
            [[5.0, np.nan], [8.0, 4.0]],
            [[True, False], [True, True]],
            ["A", "B"], ["T1", "T2"],
            measurement="curve_fit",
        )
        np.testing.assert_allclose(
            result, [[1 / 1.25, 0], [1 / 3.25, 1 / 2.25]]
        )

    def test_screen_formula(self):
        result = raw_weights(
            [[0.5, 1.0]], None, [[True, True]],
            ["A"], ["T1", "T2"], measurement="screen",
        )
        np.testing.assert_allclose(result, [[4.0, 1.0]])

    def test_invalid_observed_uncertainty_reports_identity_and_task(self):
        for measurement, low, high in (
            ("curve_fit", [[2.0]], [[2.0]]),
            ("curve_fit", [[np.nan]], [[3.0]]),
            ("screen", [[0.0]], None),
            ("screen", [[np.inf]], None),
        ):
            with self.subTest(measurement=measurement, low=low):
                with self.assertRaisesRegex(ValueError, "BAD.*T1"):
                    raw_weights(
                        low, high, [[True]], ["BAD"], ["T1"],
                        measurement=measurement,
                    )

    def test_uniform_weights_recover_installed_stock_criterion(self):
        predictions = torch.tensor([[1., 3.], [2., 4.]])
        targets = torch.tensor([[0., 1.], [2., float("nan")]])
        mask = targets.isfinite()
        rows = torch.ones(2, 1)
        expected = nn.MSE()(
            predictions, targets.nan_to_num(), mask, rows, None, None
        )
        actual = weighted_loss(
            nn.MSE(), predictions, targets, mask,
            torch.ones_like(targets), rows,
        )
        torch.testing.assert_close(actual, expected)

    def test_nonuniform_weights_preserve_stock_reduction_and_gradients(self):
        predictions = torch.tensor(
            [[1., 3.], [2., 4.]], requires_grad=True
        )
        targets = torch.tensor([[0., 1.], [0., float("nan")]])
        mask = targets.isfinite()
        raw = torch.tensor([[1., 2.], [4., 0.]])
        rows = torch.ones(2, 1)
        normalized = torch.zeros_like(raw)
        normalized[mask] = raw[mask] / raw[mask].mean()

        # Independent reference: sum individual observed-cell contributions
        # evaluated through the installed stock criterion. Every call retains
        # the full mask, so its denominator and task weighting are unchanged.
        expected = predictions.new_zeros(())
        for i, j in torch.nonzero(mask).tolist():
            isolated_residual = torch.zeros_like(predictions)
            isolated_residual[i, j] = (
                predictions[i, j] - targets[i, j]
            )
            contribution = nn.MSE()(
                isolated_residual,
                torch.zeros_like(targets),
                mask,
                rows,
                None,
                None,
            )
            expected = expected + normalized[i, j] * contribution
        actual = weighted_loss(
            nn.MSE(), predictions, targets, mask, raw, rows
        )
        torch.testing.assert_close(actual, expected)
        unweighted = nn.MSE()(
            predictions, targets.nan_to_num(), mask, rows, None, None
        )
        self.assertFalse(torch.allclose(actual, unweighted))
        actual.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertGreater(float(predictions.grad.abs().sum()), 0)
        self.assertEqual(float(predictions.grad[1, 1]), 0)

    def test_metadata_is_not_trainable(self):
        predictions = torch.tensor([[1., 2.]], requires_grad=True)
        raw = torch.tensor([[1., 3.]], requires_grad=True)
        loss = weighted_loss(
            nn.MSE(), predictions, torch.zeros_like(predictions),
            torch.ones_like(predictions, dtype=torch.bool),
            raw, torch.ones(1, 1),
        )
        loss.backward()
        self.assertIsNone(raw.grad)
        self.assertIsNotNone(predictions.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
