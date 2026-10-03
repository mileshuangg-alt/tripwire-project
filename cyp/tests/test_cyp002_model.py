from __future__ import annotations

import torch

from chemprop import nn

from cyp002_config import (
    CYP002_C0,
    CYP002_SEED,
    CYP002_VARIANT_STOCK,
    CYP002_VARIANT_TASK_BALANCED,
)
from cyp002_model import build_cyp002_model
from multitask_losses import TaskBalancedMSE


def test_variants_use_identical_frozen_architecture():
    stock = build_cyp002_model(
        CYP002_C0,
        CYP002_VARIANT_STOCK,
    )
    balanced = build_cyp002_model(
        CYP002_C0,
        CYP002_VARIANT_TASK_BALANCED,
    )

    assert isinstance(stock.message_passing, nn.BondMessagePassing)
    assert isinstance(balanced.message_passing, nn.BondMessagePassing)

    assert isinstance(stock.agg, nn.NormAggregation)
    assert isinstance(balanced.agg, nn.NormAggregation)

    assert stock.agg.norm == 100
    assert balanced.agg.norm == 100

    assert stock.message_passing.output_dim == 300
    assert balanced.message_passing.output_dim == 300

    assert stock.predictor.n_tasks == 4
    assert balanced.predictor.n_tasks == 4

    assert isinstance(stock.predictor.criterion, nn.MSE)
    assert isinstance(balanced.predictor.criterion, TaskBalancedMSE)

    assert stock.warmup_epochs == CYP002_C0.warmup_epochs
    assert balanced.warmup_epochs == CYP002_C0.warmup_epochs

    assert stock.init_lr == CYP002_C0.initial_lr
    assert balanced.init_lr == CYP002_C0.initial_lr

    assert stock.max_lr == CYP002_C0.max_lr
    assert balanced.max_lr == CYP002_C0.max_lr

    assert stock.final_lr == CYP002_C0.final_lr
    assert balanced.final_lr == CYP002_C0.final_lr


def test_variants_have_identical_parameter_initialization_under_shared_seed():
    torch.manual_seed(CYP002_SEED)
    stock = build_cyp002_model(
        CYP002_C0,
        CYP002_VARIANT_STOCK,
    )

    torch.manual_seed(CYP002_SEED)
    balanced = build_cyp002_model(
        CYP002_C0,
        CYP002_VARIANT_TASK_BALANCED,
    )

    stock_state = stock.state_dict()
    balanced_state = balanced.state_dict()

    assert stock_state.keys() == balanced_state.keys()

    for key in stock_state:
        torch.testing.assert_close(
            stock_state[key],
            balanced_state[key],
            atol=0.0,
            rtol=0.0,
        )


def run_tests() -> None:
    tests = [
        test_variants_use_identical_frozen_architecture,
        test_variants_have_identical_parameter_initialization_under_shared_seed,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} CYP-002 model tests")


if __name__ == "__main__":
    run_tests()
