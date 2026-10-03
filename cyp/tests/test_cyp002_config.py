from __future__ import annotations

from cyp002_config import (
    CYP002_C0,
    CYP002_GRID,
    CYP002_N_TASKS,
    CYP002_SEED,
    CYP002_VARIANT_STOCK,
    CYP002_VARIANT_TASK_BALANCED,
)


def test_headline_configuration_is_c0():
    assert CYP002_C0.name == "C0"
    assert CYP002_GRID[0] == CYP002_C0


def test_c0_matches_frozen_headline_contract():
    assert CYP002_C0.message_passing_depth == 3
    assert CYP002_C0.message_hidden_dim == 300
    assert CYP002_C0.ffn_hidden_dim == 300
    assert CYP002_C0.ffn_num_layers == 1
    assert CYP002_C0.dropout == 0.0
    assert CYP002_C0.aggregation == "norm"
    assert CYP002_C0.aggregation_norm == 100
    assert CYP002_C0.batch_size == 64
    assert CYP002_C0.max_epochs == 50
    assert CYP002_C0.early_stopping_patience == 20
    assert CYP002_C0.early_stopping_min_delta == 0.0
    assert CYP002_C0.initial_lr == 1e-4
    assert CYP002_C0.max_lr == 1e-3
    assert CYP002_C0.final_lr == 1e-4
    assert CYP002_C0.warmup_epochs == 2
    assert CYP002_C0.activation == "relu"
    assert CYP002_C0.batch_norm is False
    assert CYP002_C0.num_tasks == 4
    assert CYP002_C0.seed == CYP002_SEED


def test_grid_has_exactly_eight_configurations():
    assert len(CYP002_GRID) == 8


def test_all_grid_configs_share_frozen_controls():
    for config in CYP002_GRID:
        assert config.num_tasks == CYP002_N_TASKS
        assert config.seed == CYP002_SEED
        assert config.aggregation == "norm"
        assert config.aggregation_norm == 100
        assert config.batch_size == 64
        assert config.max_epochs == 50
        assert config.early_stopping_patience == 20
        assert config.early_stopping_min_delta == 0.0
        assert config.initial_lr == 1e-4
        assert config.final_lr == 1e-4
        assert config.warmup_epochs == 2
        assert config.activation == "relu"
        assert config.batch_norm is False


def test_variant_names_are_fixed():
    assert CYP002_VARIANT_STOCK == "stock"
    assert CYP002_VARIANT_TASK_BALANCED == "task_balanced"


def test_grid_names_are_unique_and_ordered():
    assert [config.name for config in CYP002_GRID] == [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
    ]
