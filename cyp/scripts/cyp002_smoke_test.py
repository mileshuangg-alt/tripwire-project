from __future__ import annotations

import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch import seed_everything

import chemprop
from chemprop import data, models, nn

from cyp002_config import CYP002_N_TASKS, CYP002_SEED
from multitask_losses import TaskBalancedMSE


DATA_PATH = Path("cyp/data/cyp-challenge-TRAIN_inhibition.csv")
ARTIFACT_DIR = Path("cyp/artifacts/cyp_002/smoke_test")
BATCH_SIZE = 8


def deterministic_worker_init(worker_id: int) -> None:
    """Seed Python and NumPy in each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_production_seed() -> None:
    """Apply the frozen CYP-002 seed to all relevant global RNGs."""
    seed_everything(CYP002_SEED, workers=True)

    random.seed(CYP002_SEED)
    np.random.seed(CYP002_SEED)
    torch.manual_seed(CYP002_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CYP002_SEED)

    torch.use_deterministic_algorithms(True)


def collect_environment() -> dict:
    """Collect environment-scoped reproducibility metadata."""
    cuda_version = getattr(torch.version, "cuda", None)

    cudnn_version = None
    if torch.backends.cudnn.is_available():
        cudnn_version = torch.backends.cudnn.version()

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": CYP002_SEED,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "chemprop_version": getattr(chemprop, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "lightning_version": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "mps_available": bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
        "deterministic_algorithms": True,
    }

    if torch.cuda.is_available():
        result["gpu_count"] = torch.cuda.device_count()
        result["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    else:
        result["gpu_count"] = 0
        result["gpu_names"] = []

    return result


def load_real_training_subset() -> data.MoleculeDataset:
    """Build a small real CYP dataset from the frozen training CSV only."""
    dataframe = pd.read_csv(DATA_PATH)

    target_columns = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    datapoints = []

    for row_index, row in dataframe.head(32).iterrows():
        y = row[target_columns].to_numpy(dtype=np.float64)

        datapoints.append(
            data.MoleculeDatapoint.from_smi(
                row["SMILES"],
                y,
                name=str(row["Molecule_Name"]),
            )
        )

    dataset = data.MoleculeDataset(datapoints)

    return dataset


def build_loader(
    dataset: data.MoleculeDataset,
) -> torch.utils.data.DataLoader:
    """Build the deterministic smoke-test DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(CYP002_SEED)

    loader = data.build_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=CYP002_SEED,
        shuffle=True,
        generator=generator,
        worker_init_fn=deterministic_worker_init,
    )

    return loader


def build_model(variant: str) -> models.MPNN:
    """Construct one of the two frozen CYP-002 loss variants."""
    if variant == "stock":
        criterion = nn.MSE()
    elif variant == "task_balanced":
        criterion = TaskBalancedMSE()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    predictor = nn.RegressionFFN(
        n_tasks=CYP002_N_TASKS,
        criterion=criterion,
    )

    model = models.MPNN(
        nn.BondMessagePassing(),
        nn.MeanAggregation(),
        predictor,
        batch_norm=False,
    )

    return model


def model_parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    """Copy model parameters for deterministic initialization comparison."""
    return [
        parameter.detach().cpu().clone()
        for parameter in model.parameters()
    ]


def assert_same_initialization(
    first: list[torch.Tensor],
    second: list[torch.Tensor],
) -> None:
    assert len(first) == len(second)

    for first_parameter, second_parameter in zip(first, second):
        torch.testing.assert_close(
            first_parameter,
            second_parameter,
            atol=0.0,
            rtol=0.0,
        )


def run_variant(
    variant: str,
    loader: torch.utils.data.DataLoader,
    environment: dict,
) -> dict:
    """Run exactly one CPU smoke-training batch for one loss variant."""
    set_production_seed()

    model = build_model(variant)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model,
        train_dataloaders=loader,
    )

    result = {
        "variant": variant,
        "seed": CYP002_SEED,
        "status": "pass",
        "environment": environment,
    }

    return result


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    environment = collect_environment()

    assert environment["chemprop_version"] == "2.3.1", (
        "D001 requires Chemprop 2.3.1, got "
        f"{environment['chemprop_version']}"
    )

    set_production_seed()

    dataset = load_real_training_subset()
    loader = build_loader(dataset)

    # Verify the loader actually yields a four-task target tensor before
    # constructing the models.
    batch = next(iter(loader))
    assert batch.Y.shape[1] == CYP002_N_TASKS

    # Rebuild the loader so the actual training runs start from the same
    # deterministic shuffle state rather than consuming the iterator above.
    set_production_seed()
    loader = build_loader(dataset)

    set_production_seed()
    stock_model = build_model("stock")
    stock_initial = model_parameter_snapshot(stock_model)

    set_production_seed()
    balanced_model = build_model("task_balanced")
    balanced_initial = model_parameter_snapshot(balanced_model)

    assert_same_initialization(stock_initial, balanced_initial)

    # Rebuild loaders independently so each smoke run receives the same
    # deterministic batch ordering.
    set_production_seed()
    stock_loader = build_loader(dataset)

    stock_result = run_variant(
        "stock",
        stock_loader,
        environment,
    )

    set_production_seed()
    balanced_loader = build_loader(dataset)

    balanced_result = run_variant(
        "task_balanced",
        balanced_loader,
        environment,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "pass",
        "seed": CYP002_SEED,
        "chemprop_version": environment["chemprop_version"],
        "environment": environment,
        "dataset_rows": len(dataset),
        "batch_size": BATCH_SIZE,
        "variants": [
            stock_result,
            balanced_result,
        ],
        "determinism_controls": {
            "lightning_seed_everything_workers": True,
            "chemprop_dataloader_seed": CYP002_SEED,
            "dataloader_generator_seed": CYP002_SEED,
            "worker_init_fn": "deterministic_worker_init",
            "torch_deterministic_algorithms": True,
            "lightning_trainer_deterministic": True,
            "primary_seed_shared_by_variants": True,
        },
    }

    output_path = ARTIFACT_DIR / "smoke_test_results.json"
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print()
    print(f"PASS: CYP-002 Chemprop smoke test")
    print(f"Artifact: {output_path}")


if __name__ == "__main__":
    main()
