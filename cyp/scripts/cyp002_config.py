from __future__ import annotations

from dataclasses import dataclass


CYP002_SEED = 20261001
CYP002_N_TASKS = 4

CYP002_TASK_NAMES = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)

CYP002_VARIANT_STOCK = "stock"
CYP002_VARIANT_TASK_BALANCED = "task_balanced"

CYP002_AGGREGATION = "norm"
CYP002_AGGREGATION_NORM = 100

CYP002_BATCH_SIZE = 64
CYP002_MAX_EPOCHS = 50
CYP002_EARLY_STOPPING_PATIENCE = 20
CYP002_EARLY_STOPPING_MIN_DELTA = 0.0

CYP002_INITIAL_LR = 1e-4
CYP002_MAX_LR = 1e-3
CYP002_FINAL_LR = 1e-4
CYP002_WARMUP_EPOCHS = 2

CYP002_DATA_LOADER_WORKERS = 0
CYP002_GRADIENT_CLIPPING = None


@dataclass(frozen=True)
class CYP002ModelConfig:
    name: str
    message_passing_depth: int
    message_hidden_dim: int
    ffn_hidden_dim: int
    ffn_num_layers: int
    dropout: float
    max_lr: float = CYP002_MAX_LR

    aggregation: str = CYP002_AGGREGATION
    aggregation_norm: int = CYP002_AGGREGATION_NORM

    batch_size: int = CYP002_BATCH_SIZE
    max_epochs: int = CYP002_MAX_EPOCHS
    early_stopping_patience: int = CYP002_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = CYP002_EARLY_STOPPING_MIN_DELTA

    initial_lr: float = CYP002_INITIAL_LR
    final_lr: float = CYP002_FINAL_LR
    warmup_epochs: int = CYP002_WARMUP_EPOCHS

    activation: str = "relu"
    batch_norm: bool = False
    num_tasks: int = CYP002_N_TASKS
    seed: int = CYP002_SEED


CYP002_C0 = CYP002ModelConfig(
    name="C0",
    message_passing_depth=3,
    message_hidden_dim=300,
    ffn_hidden_dim=300,
    ffn_num_layers=1,
    dropout=0.0,
)

CYP002_GRID = (
    CYP002_C0,
    CYP002ModelConfig(
        name="C1",
        message_passing_depth=2,
        message_hidden_dim=300,
        ffn_hidden_dim=300,
        ffn_num_layers=1,
        dropout=0.0,
    ),
    CYP002ModelConfig(
        name="C2",
        message_passing_depth=4,
        message_hidden_dim=300,
        ffn_hidden_dim=300,
        ffn_num_layers=1,
        dropout=0.0,
    ),
    CYP002ModelConfig(
        name="C3",
        message_passing_depth=3,
        message_hidden_dim=500,
        ffn_hidden_dim=300,
        ffn_num_layers=1,
        dropout=0.0,
    ),
    CYP002ModelConfig(
        name="C4",
        message_passing_depth=3,
        message_hidden_dim=300,
        ffn_hidden_dim=500,
        ffn_num_layers=1,
        dropout=0.0,
    ),
    CYP002ModelConfig(
        name="C5",
        message_passing_depth=3,
        message_hidden_dim=300,
        ffn_hidden_dim=300,
        ffn_num_layers=2,
        dropout=0.10,
    ),
    CYP002ModelConfig(
        name="C6",
        message_passing_depth=3,
        message_hidden_dim=500,
        ffn_hidden_dim=500,
        ffn_num_layers=2,
        dropout=0.10,
    ),
    CYP002ModelConfig(
        name="C7",
        message_passing_depth=3,
        message_hidden_dim=300,
        ffn_hidden_dim=300,
        ffn_num_layers=1,
        dropout=0.0,
        max_lr=5e-4,
    ),
)

assert len(CYP002_GRID) == 8
assert CYP002_GRID[0] == CYP002_C0
assert all(config.seed == CYP002_SEED for config in CYP002_GRID)
assert all(config.num_tasks == CYP002_N_TASKS for config in CYP002_GRID)
assert all(config.aggregation == "norm" for config in CYP002_GRID)
assert all(config.aggregation_norm == 100 for config in CYP002_GRID)
