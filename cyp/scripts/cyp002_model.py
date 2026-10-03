from __future__ import annotations

import torch

from chemprop import models, nn

from cyp002_config import (
    CYP002_VARIANT_STOCK,
    CYP002_VARIANT_TASK_BALANCED,
    CYP002ModelConfig,
)
from multitask_losses import TaskBalancedMSE


def build_cyp002_model(
    config: CYP002ModelConfig,
    variant: str,
) -> models.MPNN:
    """
    Build the frozen CYP-002 Chemprop architecture.

    Variant is the only scientifically consequential difference:
    stock MSE versus the D001 task-balanced masked MSE criterion.
    """
    if config.num_tasks != 4:
        raise ValueError(
            f"CYP-002 requires exactly 4 tasks, got {config.num_tasks}"
        )

    if config.aggregation != "norm":
        raise ValueError(
            "CYP-002 aggregation is frozen to Chemprop norm aggregation."
        )

    if config.aggregation_norm != 100:
        raise ValueError(
            "CYP-002 aggregation norm factor is frozen to 100."
        )

    if variant == CYP002_VARIANT_STOCK:
        criterion = nn.MSE()
    elif variant == CYP002_VARIANT_TASK_BALANCED:
        criterion = TaskBalancedMSE()
    else:
        raise ValueError(f"Unknown CYP-002 variant: {variant}")

    message_passing = nn.BondMessagePassing(
        d_h=config.message_hidden_dim,
        depth=config.message_passing_depth,
        dropout=config.dropout,
        activation=config.activation,
    )

    aggregation = nn.NormAggregation(
        norm=config.aggregation_norm,
    )

    predictor = nn.RegressionFFN(
        n_tasks=config.num_tasks,
        input_dim=config.message_hidden_dim,
        hidden_dim=config.ffn_hidden_dim,
        n_layers=config.ffn_num_layers,
        dropout=config.dropout,
        activation=config.activation,
        criterion=criterion,
    )

    model = models.MPNN(
        message_passing=message_passing,
        agg=aggregation,
        predictor=predictor,
        batch_norm=config.batch_norm,
        warmup_epochs=config.warmup_epochs,
        init_lr=config.initial_lr,
        max_lr=config.max_lr,
        final_lr=config.final_lr,
    )

    result = model
    return result
