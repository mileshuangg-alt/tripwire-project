import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "cyp/scripts/actfound_production_train_v2.py"

sys.path.insert(0, str(RUNNER.parent))

from actfound_production_train_v2 import (
    ACTFOUND_LR_PENALTY_PARAMETER,
    add_inner_learning_rate_penalty,
)


@pytest.fixture
def args():
    return SimpleNamespace(
        begin_lrloss_epoch=50,
        lrloss_weight=35.0,
    )


def test_exact_upstream_parameter_group():
    assert ACTFOUND_LR_PENALTY_PARAMETER == "layer_dict.linear.weights"


def test_epoch_49_has_no_penalty(args):
    base = torch.tensor(2.0, requires_grad=True)
    rates = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))

    total, penalty = add_inner_learning_rate_penalty(
        base,
        {ACTFOUND_LR_PENALTY_PARAMETER: rates},
        49,
        args,
        is_training_phase=True,
    )

    torch.testing.assert_close(penalty, torch.tensor(0.0))
    torch.testing.assert_close(total, base)


def test_epoch_50_has_exact_penalty(args):
    base = torch.tensor(2.0, requires_grad=True)
    rates = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))

    total, penalty = add_inner_learning_rate_penalty(
        base,
        {ACTFOUND_LR_PENALTY_PARAMETER: rates},
        50,
        args,
        is_training_phase=True,
    )

    torch.testing.assert_close(penalty, torch.tensor(10.5))
    torch.testing.assert_close(total, torch.tensor(12.5))


def test_penalty_gradient_is_exact(args):
    base = torch.tensor(2.0, requires_grad=True)
    rates = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))

    total, _ = add_inner_learning_rate_penalty(
        base,
        {ACTFOUND_LR_PENALTY_PARAMETER: rates},
        50,
        args,
        is_training_phase=True,
    )
    total.backward()

    torch.testing.assert_close(rates.grad, 70.0 * rates.detach())
    torch.testing.assert_close(base.grad, torch.tensor(1.0))


def test_evaluation_phase_has_no_penalty_or_lr_gradient(args):
    base = torch.tensor(2.0, requires_grad=True)
    rates = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))

    total, penalty = add_inner_learning_rate_penalty(
        base,
        {ACTFOUND_LR_PENALTY_PARAMETER: rates},
        50,
        args,
        is_training_phase=False,
    )

    torch.testing.assert_close(penalty, torch.tensor(0.0))
    torch.testing.assert_close(total, base)
    total.backward()

    assert rates.grad is None
    torch.testing.assert_close(base.grad, torch.tensor(1.0))


def test_runner_applies_penalty_once_in_training_only():
    tree = ast.parse(RUNNER.read_text())
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    def calls(function):
        return [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "add_inner_learning_rate_penalty"
        ]

    training_calls = calls(functions["run_training"])
    validation_calls = calls(functions["evaluate_validation"])

    assert len(training_calls) == 1
    assert len(validation_calls) == 0

    call = training_calls[0]
    keywords = {item.arg: item.value for item in call.keywords}

    assert isinstance(keywords["epoch"], ast.Name)
    assert keywords["epoch"].id == "epoch"
    assert isinstance(keywords["is_training_phase"], ast.Constant)
    assert keywords["is_training_phase"].value is True

    mapping = ast.literal_eval(call.args[1]) if isinstance(
        call.args[1], ast.Dict
    ) else None

    assert mapping is None  # Runtime mapping uses inner_lrs as its tensor value.
    assert "layer_dict.linear.weights" in RUNNER.read_text()
