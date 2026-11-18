from __future__ import annotations

import math
from dataclasses import asdict, dataclass

D007_REPETITIONS = 5
D007_FOLDS = 10
D007_TRAINING_SEEDS = (20261101, 20261102, 20261103)

D007_REDUCED_SPLITS = (
    (1, 1),
    (1, 9),
    (2, 2),
    (2, 10),
    (3, 2),
    (3, 7),
    (4, 1),
    (4, 3),
    (5, 2),
    (5, 10),
)

D007_TOTAL_CELLS = D007_REPETITIONS * D007_FOLDS * len(D007_TRAINING_SEEDS)
D007_REDUCED_TOTAL_CELLS = len(D007_REDUCED_SPLITS) * len(D007_TRAINING_SEEDS)
D007_CONTROLLER_RNG_SEED = 20261107

D007_SEARCH_SPACE = {
    "message_passing_depth": {"kind": "int", "low": 2, "high": 6},
    "message_hidden_dim": {"kind": "int_step", "low": 300, "high": 2400, "step": 100},
    "ffn_hidden_dim": {"kind": "int_step", "low": 300, "high": 2400, "step": 100},
    "ffn_num_layers": {"kind": "int", "low": 1, "high": 2},
    "dropout": {"kind": "categorical", "values": [0.0] * 8 + [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]},
    "batch_size": {"kind": "categorical", "values": [16, 32, 64, 128, 256]},
    "max_lr": {"kind": "loguniform", "low": 1e-4, "high": 1e-2},
    "init_lr_ratio": {"kind": "loguniform", "low": 1e-2, "high": 1.0},
    "final_lr_ratio": {"kind": "loguniform", "low": 1e-2, "high": 1.0},
    "warmup_epochs": {"kind": "int", "low": 1, "high": 25},
}

@dataclass(frozen=True)
class D007ControllerConfig:
    total_trials: int = 128
    random_startup_trials: int = 32
    tpe_trials: int = 96
    controller_rng_seed: int = D007_CONTROLLER_RNG_SEED
    pruning: str = "none"
    imported_observations: int = 0
    objective: str = "mean_validation_macro_st_rae_50_folds_x_3_seeds"

D007_CONTROLLER_CONFIG = D007ControllerConfig()

def cell_keys() -> list[tuple[int, int, int]]:
    return [(r, f, s) for r in range(1, D007_REPETITIONS + 1) for f in range(1, D007_FOLDS + 1) for s in D007_TRAINING_SEEDS]

def reduced_cell_keys() -> list[tuple[int, int, int]]:
    result = [
        (repetition, fold, seed)
        for repetition, fold in D007_REDUCED_SPLITS
        for seed in D007_TRAINING_SEEDS
    ]

    if len(result) != D007_REDUCED_TOTAL_CELLS:
        raise RuntimeError(
            "D007 reduced panel has unexpected cell count."
        )

    if len(set(result)) != D007_REDUCED_TOTAL_CELLS:
        raise RuntimeError(
            "D007 reduced panel contains duplicate cell identities."
        )

    return result


def reduced_cell_indices() -> list[int]:
    """
    Map the frozen reduced-search panel onto canonical D007 cell indices.

    The returned indices remain positions in the original 150-cell
    repetition x fold x seed universe. Reduced-search cells therefore
    retain the same scientific identity when reused for full finalist
    evaluation.
    """
    full_keys = cell_keys()
    index_by_key = {
        key: index
        for index, key in enumerate(full_keys)
    }

    reduced_keys = reduced_cell_keys()

    missing = [
        key
        for key in reduced_keys
        if key not in index_by_key
    ]

    if missing:
        raise RuntimeError(
            "D007 reduced cells are absent from the canonical "
            f"150-cell universe: {missing[:5]}"
        )

    result = [
        index_by_key[key]
        for key in reduced_keys
    ]

    if len(result) != D007_REDUCED_TOTAL_CELLS:
        raise RuntimeError(
            "D007 reduced index mapping has unexpected cell count."
        )

    if len(set(result)) != D007_REDUCED_TOTAL_CELLS:
        raise RuntimeError(
            "D007 reduced index mapping contains duplicate indices."
        )

    return result


def aggregate_d007_objective(results) -> float:
    expected = set(cell_keys())
    seen = {}
    for item in results:
        key = (item.get("repetition"), item.get("fold"), item.get("seed"))
        if key not in expected:
            raise ValueError(f"unexpected D007 cell: {key!r}")
        if key in seen:
            raise ValueError(f"duplicate D007 cell: {key!r}")
        value = float(item.get("macro_st_rae"))
        if not math.isfinite(value):
            raise ValueError(f"nonfinite D007 macro ST-RAE: {key!r}")
        seen[key] = value
    missing = expected - set(seen)
    if missing:
        raise ValueError(f"missing D007 cells: {sorted(missing)[:5]}")
    if len(seen) != D007_TOTAL_CELLS:
        raise ValueError(f"expected {D007_TOTAL_CELLS} cells, got {len(seen)}")
    return sum(seen.values()) / D007_TOTAL_CELLS

def controller_config() -> dict:
    return asdict(D007_CONTROLLER_CONFIG)
