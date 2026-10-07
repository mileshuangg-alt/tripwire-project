from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop import data

from cyp002_config import (
    CYP002_GRID,
    CYP002_SEED,
    CYP002_TASK_NAMES,
    CYP002_VARIANT_STOCK,
    CYP002_VARIANT_TASK_BALANCED,
)
from cyp002_manifest import (
    canonical_experiment_config,
    experiment_config_sha256,
    finalize_manifest,
    utc_now,
    write_manifest,
)
from cyp002_model import build_cyp002_model
from cyp002_validation import build_validation_callbacks
from multitask_targets import build_cyp_target_matrix


EXPECTED_TRAIN_ROWS = 3433
EXPECTED_VALIDATION_ROWS = 736
EXPECTED_TEST_ROWS = 736

CYP002_CONFIG_BY_NAME = {config.name: config for config in CYP002_GRID}


def _route_frozen_split(
    dataframe: pd.DataFrame,
    split_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"Molecule_Name", "cluster_id", "split"}
    missing = required - set(split_dataframe.columns)
    if missing:
        raise ValueError(
            f"Frozen split is missing required columns: {sorted(missing)}"
        )

    if split_dataframe["Molecule_Name"].duplicated().any():
        raise ValueError("Frozen split contains duplicate Molecule_Name values.")

    if dataframe["Molecule_Name"].duplicated().any():
        raise ValueError("Dataset contains duplicate Molecule_Name values.")

    merged = dataframe.merge(
        split_dataframe[["Molecule_Name", "cluster_id", "split"]],
        on="Molecule_Name",
        how="left",
        validate="one_to_one",
    )

    if merged["split"].isna().any():
        missing_names = merged.loc[
            merged["split"].isna(), "Molecule_Name"
        ].tolist()
        raise ValueError(
            "Dataset contains molecules absent from the frozen split: "
            f"{missing_names[:5]}"
        )

    if len(merged) != len(split_dataframe):
        raise ValueError(
            "Dataset and frozen split do not contain the same molecular universe."
        )

    values = set(merged["split"].unique())
    expected = {"train", "validation", "test"}
    if values != expected:
        raise ValueError(
            f"Unexpected split values: expected {expected}, got {values}"
        )

    train_df = merged.loc[merged["split"] == "train"].copy()
    validation_df = merged.loc[
        merged["split"] == "validation"
    ].copy()
    test_df = merged.loc[merged["split"] == "test"].copy()

    return train_df, validation_df, test_df


def build_cyp002_dataframes(
    dataframe: pd.DataFrame,
    split_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Route the authoritative dataset through the already-frozen split.

    The test partition is intentionally not returned to the training caller.
    """
    train_df, validation_df, test_df = _route_frozen_split(
        dataframe,
        split_dataframe,
    )

    if len(train_df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_ROWS} train rows, got {len(train_df)}"
        )

    if len(validation_df) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_VALIDATION_ROWS} validation rows, "
            f"got {len(validation_df)}"
        )

    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS} test rows, got {len(test_df)}"
        )

    return train_df, validation_df


def _frame_to_molecule_dataset(
    dataframe: pd.DataFrame,
) -> data.MoleculeDataset:
    target_matrix, _mask = build_cyp_target_matrix(
        dataframe,
        CYP002_TASK_NAMES,
    )

    datapoints = [
        data.MoleculeDatapoint.from_smi(
            smiles,
            target_row,
        )
        for smiles, target_row in zip(
            dataframe["SMILES"].tolist(),
            target_matrix,
            strict=True,
        )
    ]

    return data.MoleculeDataset(datapoints)


def build_cyp002_datasets(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
):
    """
    Build Chemprop datasets and fit target scaling on training data only.
    """
    train_dataset = _frame_to_molecule_dataset(train_dataframe)
    validation_dataset = _frame_to_molecule_dataset(validation_dataframe)

    scaler = train_dataset.normalize_targets()
    validation_dataset.normalize_targets(scaler)

    return train_dataset, validation_dataset, scaler


def canonical_manifest_matches(path: str | Path) -> bool:
    manifest = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    recorded = manifest["experiment_config_sha256"]
    recomputed = experiment_config_sha256(
        manifest["canonical_experiment_config"]
    )

    return recorded == recomputed


def _configure_determinism() -> None:
    pl.seed_everything(CYP002_SEED, workers=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _assert_manifest_contract(
    manifest_path: Path,
    variant: str,
    config_name: str,
) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Planned manifest does not exist: {manifest_path}"
        )

    if not canonical_manifest_matches(manifest_path):
        raise RuntimeError(
            "Planned manifest canonical configuration hash does not match."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest["configuration"] != config_name:
        raise RuntimeError(
            "Manifest configuration does not match requested config: "
            f"{manifest['configuration']!r} != {config_name!r}"
        )

    if manifest["variant"] != variant:
        raise RuntimeError(
            "Manifest variant does not match requested variant: "
            f"{manifest['variant']!r} != {variant!r}"
        )

    if manifest["seed"] != CYP002_SEED:
        raise RuntimeError(
            "Manifest seed does not match CYP002_SEED."
        )

    if manifest["execution"]["status"] != "running":
        raise RuntimeError(
            "Manifest must be in planned/running pre-execution state."
        )

    if manifest["outputs"]["best_checkpoint_sha256"] is not None:
        raise RuntimeError(
            "Checkpoint hash is already populated; refusing to overwrite."
        )

    if manifest["outputs"]["metrics_sha256"] is not None:
        raise RuntimeError(
            "Metrics hash is already populated; refusing to overwrite."
        )

    if manifest["environment"]["hostname"] != socket.gethostname():
        raise RuntimeError(
            "Manifest hostname does not match the actual execution host."
        )

    return manifest


def run_training(
    *,
    variant: str,
    config_name: str,
    manifest_path: Path,
    dataset_path: Path,
    split_path: Path,
) -> None:
    try:
        config = CYP002_CONFIG_BY_NAME[config_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CYP002 configuration {config_name!r}"
        ) from exc

    if variant not in {
        CYP002_VARIANT_STOCK,
        CYP002_VARIANT_TASK_BALANCED,
    }:
        raise ValueError(f"Unknown CYP-002 variant: {variant!r}")

    _configure_determinism()

    manifest = _assert_manifest_contract(
        manifest_path,
        variant,
        config.name,
    )

    dataset = pd.read_csv(dataset_path)
    split_dataframe = pd.read_csv(split_path)

    train_df, validation_df = build_cyp002_dataframes(
        dataset,
        split_dataframe,
    )

    train_dataset, validation_dataset, scaler = build_cyp002_datasets(
        train_df,
        validation_df,
    )

    train_loader = data.build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=CYP002_SEED,
        shuffle=True,
    )

    validation_loader = data.build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=CYP002_SEED,
        shuffle=False,
    )

    confidence_low = {
        task: validation_df[f"{task}_conf_low"].to_numpy(
            dtype=np.float64
        )
        for task in CYP002_TASK_NAMES
    }

    confidence_high = {
        task: validation_df[f"{task}_conf_high"].to_numpy(
            dtype=np.float64
        )
        for task in CYP002_TASK_NAMES
    }

    model = build_cyp002_model(
        config,
        variant,
        output_scaler=scaler,
        validation_conf_low=confidence_low,
        validation_conf_high=confidence_high,
    )

    checkpoint_dir = manifest_path.parent

    callbacks = build_validation_callbacks(
        checkpoint_dir,
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    manifest["execution"]["start_utc"] = utc_now()
    manifest["execution"]["status"] = "running"
    write_manifest(manifest, manifest_path)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        deterministic=True,
        max_epochs=config.max_epochs,
        logger=False,
        enable_progress_bar=True,
        callbacks=callbacks,
        gradient_clip_val=None,
    )

    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )

        checkpoint_path = checkpoint_dir / "best.ckpt"
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Expected best checkpoint was not created: "
                f"{checkpoint_path}"
            )

        if not getattr(model, "validation_history", None):
            raise RuntimeError(
                "No validation ST-RAE history was recorded."
            )

        best_validation = min(
            model.validation_history,
            key=lambda item: item["macro_st_rae"],
        )

        metrics = {
            "experiment_id": manifest["experiment_id"],
            "configuration": config.name,
            "variant": variant,
            "seed": CYP002_SEED,
            "best_validation": best_validation,
            "validation_history": model.validation_history,
            "test_evaluation_performed": False,
            "best_checkpoint_path": str(checkpoint_path),
        }

        metrics_path = checkpoint_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        finalize_manifest(
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            status="completed",
        )

    except Exception:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["execution"]["end_utc"] = utc_now()
        manifest["execution"]["status"] = "failed"
        write_manifest(manifest, manifest_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            CYP002_VARIANT_STOCK,
            CYP002_VARIANT_TASK_BALANCED,
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        choices=tuple(config.name for config in CYP002_GRID),
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        default="cyp/data/cyp-challenge-TRAIN_inhibition.csv",
    )
    parser.add_argument(
        "--split",
        default=(
            "cyp/artifacts/cyp_baseline/"
            "molecule_split_assignments.csv"
        ),
    )

    args = parser.parse_args()

    run_training(
        variant=args.variant,
        config_name=args.config,
        manifest_path=Path(args.manifest),
        dataset_path=Path(args.dataset),
        split_path=Path(args.split),
    )


if __name__ == "__main__":
    main()
