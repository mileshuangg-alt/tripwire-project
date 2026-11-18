
from pathlib import Path
from types import SimpleNamespace
import ast
import io
import unittest

import torch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "cyp/scripts/actfound_production_train_v2.py"
)
NAMES = {
    "transfer_actfound_stage_initialization",
    "build_fresh_actfound_outer_state",
}
tree = ast.parse(SOURCE.read_text())
nodes = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in NAMES
]
if {node.name for node in nodes} != NAMES:
    raise RuntimeError("Required production functions missing.")
namespace = {"ACTFOUND_LR_PENALTY_PARAMETER": "layer_dict.linear.weights"}
exec(
    compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"),
    namespace,
)
transfer = namespace["transfer_actfound_stage_initialization"]
fresh = namespace["build_fresh_actfound_outer_state"]


class StageTransferTests(unittest.TestCase):
    def setup_models(self):
        source = torch.nn.Linear(3, 1)
        destination = torch.nn.Linear(3, 1)
        rates = torch.nn.Parameter(torch.zeros(3))
        checkpoint = {
            "regressor_state_dict": source.state_dict(),
            "inner_lrs": torch.tensor([0.1, 0.2, 0.3]),
            "epoch": 73,
            "optimizer_state_dict": {"must_not_import": True},
            "scheduler_state_dict": {"must_not_import": True},
        }
        return source, destination, rates, checkpoint

    def test_exact_transfer_and_parameter_identity(self):
        source, destination, rates, checkpoint = self.setup_models()
        identity = id(rates)
        report = transfer(checkpoint, destination, rates)
        self.assertEqual(id(rates), identity)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(destination.state_dict()[key], value)
        torch.testing.assert_close(rates, checkpoint["inner_lrs"])
        self.assertFalse(report["optimizer_state_transferred"])
        self.assertFalse(report["scheduler_state_transferred"])
        self.assertFalse(report["epoch_state_transferred"])
        self.assertFalse(report["task_normalization_transferred"])

    def test_incompatible_rates_fail_before_model_mutation(self):
        _, destination, rates, checkpoint = self.setup_models()
        before = {k: v.clone() for k, v in destination.state_dict().items()}
        checkpoint["inner_lrs"] = torch.ones(4)
        with self.assertRaisesRegex(ValueError, "Incompatible learned-rate"):
            transfer(checkpoint, destination, rates)
        for key, value in before.items():
            torch.testing.assert_close(destination.state_dict()[key], value)

    def test_incompatible_model_fails_before_mutation(self):
        _, destination, rates, checkpoint = self.setup_models()
        checkpoint["regressor_state_dict"]["weight"] = torch.ones(1, 4)
        before = rates.detach().clone()
        with self.assertRaisesRegex(ValueError, "Incompatible ActFound"):
            transfer(checkpoint, destination, rates)
        torch.testing.assert_close(rates, before)

    def test_fresh_outer_state_and_parameter_coverage(self):
        _, destination, rates, checkpoint = self.setup_models()
        transfer(checkpoint, destination, rates)
        args = SimpleNamespace(
            meta_lr=0.001, metatrain_iterations=80, min_learning_rate=0.00001
        )
        # Values above are test fixtures, not a production schedule.
        first = fresh(destination, rates, args)
        second = fresh(destination, rates, args)
        self.assertIsNot(first["optimizer"], second["optimizer"])
        self.assertIsNot(first["scheduler"], second["scheduler"])
        self.assertFalse(first["optimizer"].state)
        expected = {id(p) for p in destination.parameters()} | {id(rates)}
        actual = {
            id(p)
            for group in first["optimizer"].param_groups
            for p in group["params"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(
            first["scheduler"].state_dict(),
            second["scheduler"].state_dict(),
        )

    def test_state_checkpoint_round_trip(self):
        _, destination, rates, checkpoint = self.setup_models()
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu")
        transfer(loaded, destination, rates)
        torch.testing.assert_close(rates, checkpoint["inner_lrs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
