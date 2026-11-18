import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cyp" / "scripts"))

from actfound_production_train_v2 import (
    ACTFOUND_LR_PENALTY_PARAMETER,
    add_inner_learning_rate_penalty,
    validate_actfound_lr_parameter_mapping,
)


@pytest.fixture
def args():
    return SimpleNamespace(
        begin_lrloss_epoch=50,
        lrloss_weight=35.0,
    )


def test_upstream_parameter_group():
    assert ACTFOUND_LR_PENALTY_PARAMETER == "layer_dict.linear.weights"


def test_mapping_accepts_only_the_designated_group():
    validate_actfound_lr_parameter_mapping(
        ["layer_dict.linear.weights"]
    )


def test_mapping_rejects_other_or_additional_groups():
    with pytest.raises(RuntimeError, match="differs from upstream"):
        validate_actfound_lr_parameter_mapping(
            ["layer_dict.linear.biases"]
        )

    with pytest.raises(RuntimeError, match="differs from upstream"):
        validate_actfound_lr_parameter_mapping(
            ["layer_dict.linear.weights", "other.group"]
        )


def test_epoch_49_penalty_is_zero(args):
    base = torch.tensor(2.0, requires_grad=True)
    rates = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

    total, penalty = add_inner_learning_rate_penalty(
        base, rates, 49, args, is_training_phase=True
    )

    torch.testing.assert_close(penalty, torch.tensor(0.0))
    torch.testing.assert_close(total, base)


def test_epoch_50_exact_penalty_and_gradient(args):
    base = torch.tensor(2.0, requires_grad=True)
    rates = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

    total, penalty = add_inner_learning_rate_penalty(
        base, rates, 50, args, is_training_phase=True
    )

    expected = 35.0 * (0.1**2 + 0.2**2 + 0.3**2)

    torch.testing.assert_close(penalty, torch.tensor(expected))
    torch.testing.assert_close(total, base + expected)

    total.backward()

    torch.testing.assert_close(
        rates.grad,
        70.0 * rates.detach(),
    )


def test_evaluation_phase_has_no_penalty_or_lr_gradient(args):
    base = torch.tensor(2.0, requires_grad=True)
    rates = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

    total, penalty = add_inner_learning_rate_penalty(
        base, rates, 50, args, is_training_phase=False
    )

    torch.testing.assert_close(penalty, torch.tensor(0.0))
    torch.testing.assert_close(total, base)

    total.backward()
    assert rates.grad is None
