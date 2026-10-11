from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


STEP3_UNIVERSE_SIZE = 4169

VALID_STEP3_ARMS = {
    "repeated_clustered_10fold",
    "monte_carlo_cluster_split_refit",
}


def sha256_file(path: str | Path) -> str:
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    result = digest.hexdigest()
    return result


def _read_frozen_sha256s(
    frozen_root: str | Path,
) -> dict[str, str]:
    frozen_root = Path(frozen_root)
    sums_path = frozen_root / "SHA256SUMS"

    if not sums_path.is_file():
        raise FileNotFoundError(
            f"Frozen Step-3 SHA256SUMS not found: {sums_path}"
        )

    result: dict[str, str] = {}

    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue

        try:
            digest, relative_path = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed SHA256SUMS line: {line!r}"
            ) from exc

        if relative_path in result:
            raise ValueError(
                "Duplicate path in frozen SHA256SUMS: "
                f"{relative_path!r}"
            )

        result[relative_path] = digest

    return result


def load_frozen_step3_manifest(
    manifest_path: str | Path,
    frozen_root: str | Path,
) -> dict[str, Any]:
    """
    Load one Step-3 split manifest only after proving that it is an
    unchanged member of the frozen production split package.

    This function validates frozen-package provenance and internal
    partition structure. It does not yet establish equality with the
    authoritative molecular dataset; dataset identity is checked by the
    subsequent routing function.
    """
    manifest_path = Path(manifest_path)
    frozen_root = Path(frozen_root)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Step-3 split manifest not found: {manifest_path}"
        )

    if not frozen_root.is_dir():
        raise FileNotFoundError(
            f"Frozen Step-3 package not found: {frozen_root}"
        )

    resolved_manifest = manifest_path.resolve()
    resolved_root = frozen_root.resolve()

    try:
        relative_path = resolved_manifest.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "Step-3 manifest is outside the frozen production package: "
            f"{manifest_path}"
        ) from exc

    frozen_sha256s = _read_frozen_sha256s(frozen_root)
    relative_text = relative_path.as_posix()

    if relative_text not in frozen_sha256s:
        raise ValueError(
            "Step-3 manifest is not recorded in the frozen SHA256SUMS: "
            f"{relative_text}"
        )

    expected_sha256 = frozen_sha256s[relative_text]
    actual_sha256 = sha256_file(manifest_path)

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Frozen Step-3 manifest SHA-256 mismatch: "
            f"{relative_text}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    arm = manifest.get("arm")
    if arm not in VALID_STEP3_ARMS:
        raise ValueError(
            f"Unexpected Step-3 arm: {arm!r}"
        )

    training_seed = manifest.get("training_seed")
    if not isinstance(training_seed, int) or isinstance(
        training_seed,
        bool,
    ):
        raise ValueError(
            "Step-3 training_seed must be an integer, got "
            f"{training_seed!r}"
        )

    partition_keys = (
        "inner_training_names",
        "inner_validation_names",
        "outer_holdout_names",
    )

    partitions: dict[str, set[str]] = {}

    for key in partition_keys:
        values = manifest.get(key)

        if not isinstance(values, list):
            raise ValueError(
                f"Step-3 manifest {key} must be a list."
            )

        if not all(
            isinstance(value, str) and value
            for value in values
        ):
            raise ValueError(
                f"Step-3 manifest {key} contains an invalid molecule name."
            )

        if len(values) != len(set(values)):
            raise ValueError(
                f"Step-3 manifest {key} contains duplicate molecule names."
            )

        partitions[key] = set(values)

    inner_train = partitions["inner_training_names"]
    inner_validation = partitions["inner_validation_names"]
    outer_holdout = partitions["outer_holdout_names"]

    if inner_train & inner_validation:
        raise ValueError(
            "Step-3 inner-training and inner-validation partitions overlap."
        )

    if inner_train & outer_holdout:
        raise ValueError(
            "Step-3 inner-training and outer-held-out partitions overlap."
        )

    if inner_validation & outer_holdout:
        raise ValueError(
            "Step-3 inner-validation and outer-held-out partitions overlap."
        )

    molecular_universe = (
        inner_train
        | inner_validation
        | outer_holdout
    )

    if len(molecular_universe) != STEP3_UNIVERSE_SIZE:
        raise ValueError(
            "Step-3 manifest does not contain the frozen 4,169-molecule "
            f"universe: got {len(molecular_universe)} unique molecules."
        )

    result = manifest
    return result


def route_step3_dataframes(
    dataframe: Any,
    original_split_dataframe: Any,
    step3_manifest: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """
    Route the authoritative CYP dataset into the frozen Step-3
    inner-training, inner-validation, and outer-held-out partitions.

    The authorized Step-3 molecular universe is exactly the original
    frozen train + validation population. The original internal-test
    population is protected and must not enter any Step-3 partition.
    """
    required_split_columns = {
        "Molecule_Name",
        "cluster_id",
        "split",
    }
    missing_split_columns = (
        required_split_columns
        - set(original_split_dataframe.columns)
    )

    if missing_split_columns:
        raise ValueError(
            "Original frozen split is missing required columns: "
            f"{sorted(missing_split_columns)}"
        )

    if "Molecule_Name" not in dataframe.columns:
        raise ValueError(
            "Authoritative CYP dataset is missing Molecule_Name."
        )

    if dataframe["Molecule_Name"].duplicated().any():
        raise ValueError(
            "Authoritative CYP dataset contains duplicate Molecule_Name "
            "values."
        )

    if original_split_dataframe["Molecule_Name"].duplicated().any():
        raise ValueError(
            "Original frozen split contains duplicate Molecule_Name values."
        )

    dataset_names = set(
        dataframe["Molecule_Name"].tolist()
    )
    original_split_names = set(
        original_split_dataframe["Molecule_Name"].tolist()
    )

    if dataset_names != original_split_names:
        raise ValueError(
            "Authoritative CYP dataset and original frozen split do not "
            "contain the same molecular universe."
        )

    split_values = set(
        original_split_dataframe["split"].tolist()
    )
    expected_split_values = {
        "train",
        "validation",
        "test",
    }

    if split_values != expected_split_values:
        raise ValueError(
            "Unexpected original frozen split values: "
            f"expected {expected_split_values}, got {split_values}"
        )

    original_train_names = set(
        original_split_dataframe.loc[
            original_split_dataframe["split"] == "train",
            "Molecule_Name",
        ].tolist()
    )

    original_validation_names = set(
        original_split_dataframe.loc[
            original_split_dataframe["split"] == "validation",
            "Molecule_Name",
        ].tolist()
    )

    protected_test_names = set(
        original_split_dataframe.loc[
            original_split_dataframe["split"] == "test",
            "Molecule_Name",
        ].tolist()
    )

    if len(original_train_names) != 3433:
        raise ValueError(
            "Expected 3,433 original training molecules, got "
            f"{len(original_train_names)}."
        )

    if len(original_validation_names) != 736:
        raise ValueError(
            "Expected 736 original validation molecules, got "
            f"{len(original_validation_names)}."
        )

    if len(protected_test_names) != 736:
        raise ValueError(
            "Expected 736 protected internal-test molecules, got "
            f"{len(protected_test_names)}."
        )

    authorized_step3_names = (
        original_train_names
        | original_validation_names
    )

    if len(authorized_step3_names) != STEP3_UNIVERSE_SIZE:
        raise ValueError(
            "Original train + validation population does not equal the "
            "frozen 4,169-molecule Step-3 universe."
        )

    inner_training_names = set(
        step3_manifest["inner_training_names"]
    )
    inner_validation_names = set(
        step3_manifest["inner_validation_names"]
    )
    outer_holdout_names = set(
        step3_manifest["outer_holdout_names"]
    )

    manifest_names = (
        inner_training_names
        | inner_validation_names
        | outer_holdout_names
    )

    if manifest_names != authorized_step3_names:
        missing = sorted(
            authorized_step3_names - manifest_names
        )
        unauthorized = sorted(
            manifest_names - authorized_step3_names
        )

        raise ValueError(
            "Frozen Step-3 manifest does not exactly equal the authorized "
            "original train + validation universe. "
            f"Missing examples: {missing[:5]}; "
            f"unauthorized examples: {unauthorized[:5]}"
        )

    if manifest_names & protected_test_names:
        raise RuntimeError(
            "Protected internal-test molecule entered the Step-3 universe."
        )

    cluster_id_by_name = original_split_dataframe.set_index(
        "Molecule_Name"
    )["cluster_id"]

    routed_dataframe = dataframe.copy()
    routed_dataframe["cluster_id"] = routed_dataframe[
        "Molecule_Name"
    ].map(cluster_id_by_name)

    if routed_dataframe["cluster_id"].isna().any():
        raise RuntimeError(
            "Authoritative cluster_id mapping produced missing values."
        )

    indexed = routed_dataframe.set_index(
        "Molecule_Name",
        drop=False,
        verify_integrity=True,
    )

    # Preserve the frozen manifest's molecule ordering within each
    # partition rather than inheriting source-dataset row order.
    inner_training_df = indexed.loc[
        step3_manifest["inner_training_names"]
    ].copy()

    inner_validation_df = indexed.loc[
        step3_manifest["inner_validation_names"]
    ].copy()

    outer_holdout_df = indexed.loc[
        step3_manifest["outer_holdout_names"]
    ].copy()

    if len(inner_training_df) != len(inner_training_names):
        raise RuntimeError(
            "Inner-training routing changed molecule cardinality."
        )

    if len(inner_validation_df) != len(inner_validation_names):
        raise RuntimeError(
            "Inner-validation routing changed molecule cardinality."
        )

    if len(outer_holdout_df) != len(outer_holdout_names):
        raise RuntimeError(
            "Outer-held-out routing changed molecule cardinality."
        )

    result = (
        inner_training_df,
        inner_validation_df,
        outer_holdout_df,
    )
    return result


def build_step3_datasets(
    inner_training_dataframe: Any,
    inner_validation_dataframe: Any,
    outer_holdout_dataframe: Any,
):
    """
    Build the three Step-3 Chemprop datasets.

    Target scaling is fit exactly once on inner-training targets.
    The resulting frozen scaler is then applied unchanged to inner
    validation and outer holdout.
    """
    from cyp002_train import _frame_to_molecule_dataset

    inner_training_dataset = _frame_to_molecule_dataset(
        inner_training_dataframe
    )
    inner_validation_dataset = _frame_to_molecule_dataset(
        inner_validation_dataframe
    )
    outer_holdout_dataset = _frame_to_molecule_dataset(
        outer_holdout_dataframe
    )

    scaler = inner_training_dataset.normalize_targets()

    inner_validation_dataset.normalize_targets(scaler)
    outer_holdout_dataset.normalize_targets(scaler)

    result = (
        inner_training_dataset,
        inner_validation_dataset,
        outer_holdout_dataset,
        scaler,
    )
    return result


def build_step3_prefit_state(
    *,
    inner_training_dataframe: Any,
    inner_validation_dataframe: Any,
    outer_holdout_dataframe: Any,
    training_seed: int,
    variant: str,
):
    """
    Build the frozen C0 Step-3 pre-fit state.

    Inner training supplies gradient data.
    Inner validation supplies checkpoint/early-stopping data and
    validation confidence bounds.
    Outer holdout is prepared for post-checkpoint evaluation only.
    """
    import numpy as np
    from chemprop import data

    from cyp002_config import (
        CYP002_GRID,
        CYP002_TASK_NAMES,
        CYP002_VARIANT_STOCK,
        CYP002_VARIANT_TASK_BALANCED,
    )
    from cyp002_model import build_cyp002_model
    from cyp002_train import _configure_determinism

    valid_variants = {
        CYP002_VARIANT_STOCK,
        CYP002_VARIANT_TASK_BALANCED,
    }

    if variant not in valid_variants:
        raise ValueError(
            f"Unknown CYP-002 variant: {variant!r}"
        )

    if not isinstance(training_seed, int) or isinstance(
        training_seed,
        bool,
    ):
        raise ValueError(
            f"training_seed must be an integer, got {training_seed!r}"
        )

    config_by_name = {
        config.name: config
        for config in CYP002_GRID
    }
    config = config_by_name["C0"]

    _configure_determinism(training_seed)

    (
        inner_training_dataset,
        inner_validation_dataset,
        outer_holdout_dataset,
        scaler,
    ) = build_step3_datasets(
        inner_training_dataframe,
        inner_validation_dataframe,
        outer_holdout_dataframe,
    )

    train_loader = data.build_dataloader(
        inner_training_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=training_seed,
        shuffle=True,
    )

    validation_loader = data.build_dataloader(
        inner_validation_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=training_seed,
        shuffle=False,
    )

    outer_loader = data.build_dataloader(
        outer_holdout_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=training_seed,
        shuffle=False,
    )

    validation_conf_low = {
        task: inner_validation_dataframe[
            f"{task}_conf_low"
        ].to_numpy(dtype=np.float64)
        for task in CYP002_TASK_NAMES
    }

    validation_conf_high = {
        task: inner_validation_dataframe[
            f"{task}_conf_high"
        ].to_numpy(dtype=np.float64)
        for task in CYP002_TASK_NAMES
    }

    model = build_cyp002_model(
        config,
        variant,
        output_scaler=scaler,
        validation_conf_low=validation_conf_low,
        validation_conf_high=validation_conf_high,
    )

    result = {
        "config": config,
        "model": model,
        "scaler": scaler,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "outer_loader": outer_loader,
        "validation_conf_low": validation_conf_low,
        "validation_conf_high": validation_conf_high,
        "training_seed": training_seed,
        "variant": variant,
    }
    return result


def fit_step3_inner_model(
    *,
    prefit_state: dict[str, Any],
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """
    Fit one frozen C0 Step-3 refit using inner training and inner
    validation only.

    This function MUST NOT access or evaluate the outer holdout.
    Its scientific endpoint is creation of the validation-selected
    best.ckpt.
    """
    from lightning import pytorch as pl

    from cyp002_validation import build_validation_callbacks

    checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_dir.exists():
        if any(checkpoint_dir.iterdir()):
            raise RuntimeError(
                "Step-3 checkpoint directory is not empty; refusing "
                f"to overwrite prior artifacts: {checkpoint_dir}"
            )
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)

    required_keys = {
        "config",
        "model",
        "train_loader",
        "validation_loader",
        "outer_loader",
        "training_seed",
        "variant",
    }
    missing = required_keys - set(prefit_state)

    if missing:
        raise ValueError(
            f"Step-3 prefit state is missing keys: {sorted(missing)}"
        )

    config = prefit_state["config"]
    model = prefit_state["model"]

    callbacks = build_validation_callbacks(
        checkpoint_dir,
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

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

    # SCIENTIFIC BOUNDARY:
    # outer_loader is intentionally NOT supplied to trainer.fit().
    trainer.fit(
        model,
        train_dataloaders=prefit_state["train_loader"],
        val_dataloaders=prefit_state["validation_loader"],
    )

    checkpoint_path = checkpoint_dir / "best.ckpt"

    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Validation-selected Step-3 checkpoint was not created: "
            f"{checkpoint_path}"
        )

    validation_history = getattr(
        model,
        "validation_history",
        None,
    )

    if not validation_history:
        raise RuntimeError(
            "No Step-3 inner-validation ST-RAE history was recorded."
        )

    best_validation = min(
        validation_history,
        key=lambda item: item["macro_st_rae"],
    )

    result = {
        "checkpoint_path": checkpoint_path,
        "best_validation": best_validation,
        "validation_history": list(validation_history),
        "epochs_completed": len(validation_history),
        "training_seed": prefit_state["training_seed"],
        "variant": prefit_state["variant"],
    }
    return result


def load_step3_selected_checkpoint(
    *,
    prefit_state: dict[str, Any],
    checkpoint_path: str | Path,
):
    """
    Rebuild the frozen C0 model and load exactly the validation-selected
    checkpoint weights.

    The rebuilt model receives the same training-only output scaler but
    no validation confidence state, because checkpoint selection is
    already complete.
    """
    import torch

    from cyp002_model import build_cyp002_model

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Selected Step-3 checkpoint not found: {checkpoint_path}"
        )

    model = build_cyp002_model(
        prefit_state["config"],
        prefit_state["variant"],
        output_scaler=prefit_state["scaler"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" not in checkpoint:
        raise RuntimeError(
            "Selected Step-3 checkpoint does not contain state_dict."
        )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()

    result = model
    return result


def evaluate_step3_outer_holdout(
    *,
    selected_model: Any,
    outer_loader: Any,
    outer_dataframe: Any,
    scaler: Any,
) -> dict[str, Any]:
    """
    Evaluate the frozen outer holdout only after checkpoint selection.

    Predictions are scored using the already-qualified CYP ST-RAE path.
    """
    import numpy as np
    import torch

    from cyp002_config import CYP002_TASK_NAMES
    from cyp002_validation import score_collected_validation_epoch

    prediction_batches = []
    target_batches = []
    mask_batches = []

    selected_model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    selected_model.to(device)

    with torch.no_grad():
        for batch in outer_loader:
            bmg = batch.bmg
            bmg.to(device)

            V_d = (
                None
                if batch.V_d is None
                else batch.V_d.to(device)
            )
            X_d = (
                None
                if batch.X_d is None
                else batch.X_d.to(device)
            )

            targets = batch.Y.to(device)

            mask = targets.isfinite()
            scoring_targets = targets.nan_to_num(nan=0.0)

            predictions = selected_model(
                bmg,
                V_d,
                X_d,
            )

            if selected_model.predictor.n_targets > 1:
                predictions = predictions[..., 0]

            prediction_batches.append(
                predictions.detach().cpu()
            )
            target_batches.append(
                scoring_targets.detach().cpu()
            )
            mask_batches.append(
                mask.detach().cpu()
            )

    conf_low = {
        task: outer_dataframe[
            f"{task}_conf_low"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        for task in CYP002_TASK_NAMES
    }

    conf_high = {
        task: outer_dataframe[
            f"{task}_conf_high"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        for task in CYP002_TASK_NAMES
    }

    score = score_collected_validation_epoch(
        prediction_batches=prediction_batches,
        target_batches=target_batches,
        mask_batches=mask_batches,
        conf_low=conf_low,
        conf_high=conf_high,
        task_names=CYP002_TASK_NAMES,
        target_mean=np.asarray(
            scaler.mean_,
            dtype=np.float64,
        ),
        target_scale=np.asarray(
            scaler.scale_,
            dtype=np.float64,
        ),
    )

    predictions = torch.cat(
        prediction_batches,
        dim=0,
    ).numpy()

    masks = torch.cat(
        mask_batches,
        dim=0,
    ).numpy()

    if predictions.shape != (
        len(outer_dataframe),
        len(CYP002_TASK_NAMES),
    ):
        raise RuntimeError(
            "Outer prediction matrix has unexpected shape: "
            f"{predictions.shape}"
        )

    result = {
        "score": score,
        "predictions": predictions,
        "masks": masks,
        "molecule_names": outer_dataframe[
            "Molecule_Name"
        ].tolist(),
    }
    return result


def _write_step3_prediction_matrix(
    *,
    outer_dataframe: Any,
    evaluation: dict[str, Any],
    output_path: str | Path,
) -> None:
    import pandas as pd

    from cyp002_config import CYP002_TASK_NAMES

    output_path = Path(output_path)

    predictions = evaluation["predictions"]
    masks = evaluation["masks"]

    if len(outer_dataframe) != predictions.shape[0]:
        raise RuntimeError("Outer prediction row count mismatch.")

    rows: dict[str, Any] = {
        "Molecule_Name": outer_dataframe["Molecule_Name"].tolist(),
        "cluster_id": outer_dataframe["cluster_id"].tolist(),
    }

    for task_index, task in enumerate(CYP002_TASK_NAMES):
        rows[f"{task}_y_true"] = outer_dataframe[task].to_numpy()
        rows[f"{task}_y_pred"] = predictions[:, task_index]
        rows[f"{task}_observed"] = masks[:, task_index]
        rows[f"{task}_conf_low"] = outer_dataframe[
            f"{task}_conf_low"
        ].to_numpy()
        rows[f"{task}_conf_high"] = outer_dataframe[
            f"{task}_conf_high"
        ].to_numpy()

    result = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def run_step3_refit(
    *,
    variant: str,
    split_manifest_path: str | Path,
    frozen_root: str | Path,
    dataset_path: str | Path,
    original_split_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    import socket

    import pandas as pd

    from cyp002_manifest import environment_identity, utc_now

    split_manifest_path = Path(split_manifest_path)
    frozen_root = Path(frozen_root)
    dataset_path = Path(dataset_path)
    original_split_path = Path(original_split_path)
    output_dir = Path(output_dir)

    if output_dir.exists():
        raise RuntimeError(
            f"Production output already exists; refusing overwrite: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    run_manifest_path = output_dir / "run_manifest.json"

    try:
        split_manifest = load_frozen_step3_manifest(
            split_manifest_path,
            frozen_root,
        )

        dataframe = pd.read_csv(dataset_path)
        original_split = pd.read_csv(original_split_path)

        train_df, val_df, outer_df = route_step3_dataframes(
            dataframe,
            original_split,
            split_manifest,
        )

        prefit = build_step3_prefit_state(
            inner_training_dataframe=train_df,
            inner_validation_dataframe=val_df,
            outer_holdout_dataframe=outer_df,
            training_seed=split_manifest["training_seed"],
            variant=variant,
        )

        run_manifest = {
            "status": "running",
            "start_utc": utc_now(),
            "end_utc": None,
            "hostname": socket.gethostname(),
            "variant": variant,
            "configuration": "C0",
            "training_seed": split_manifest["training_seed"],
            "split_manifest_path": str(split_manifest_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "frozen_freeze_record_sha256": sha256_file(
                frozen_root / "FREEZE_RECORD.json"
            ),
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "original_split_path": str(original_split_path),
            "original_split_sha256": sha256_file(original_split_path),
            "environment": environment_identity(),
            "partition_counts": {
                "inner_training": len(train_df),
                "inner_validation": len(val_df),
                "outer_holdout": len(outer_df),
            },
            "outputs": {},
        }

        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        fit_result = fit_step3_inner_model(
            prefit_state=prefit,
            checkpoint_dir=output_dir / "fit",
        )

        selected_model = load_step3_selected_checkpoint(
            prefit_state=prefit,
            checkpoint_path=fit_result["checkpoint_path"],
        )

        evaluation = evaluate_step3_outer_holdout(
            selected_model=selected_model,
            outer_loader=prefit["outer_loader"],
            outer_dataframe=outer_df,
            scaler=prefit["scaler"],
        )

        predictions_path = output_dir / "outer_predictions.csv"
        _write_step3_prediction_matrix(
            outer_dataframe=outer_df,
            evaluation=evaluation,
            output_path=predictions_path,
        )

        metrics = {
            "variant": variant,
            "configuration": "C0",
            "training_seed": split_manifest["training_seed"],
            "best_inner_validation": fit_result["best_validation"],
            "validation_epochs": fit_result["epochs_completed"],
            "outer_evaluation": evaluation["score"],
            "outer_rows": len(outer_df),
            "test_evaluation_performed": False,
        }

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        run_manifest["outputs"] = {
            "best_checkpoint": str(fit_result["checkpoint_path"]),
            "best_checkpoint_sha256": sha256_file(
                fit_result["checkpoint_path"]
            ),
            "outer_predictions": str(predictions_path),
            "outer_predictions_sha256": sha256_file(predictions_path),
            "metrics": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        }
        run_manifest["status"] = "completed"
        run_manifest["end_utc"] = utc_now()

        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        result = {
            "output_dir": output_dir,
            "manifest_path": run_manifest_path,
            "metrics_path": metrics_path,
            "predictions_path": predictions_path,
            "checkpoint_path": fit_result["checkpoint_path"],
            "outer_score": evaluation["score"],
        }
        return result

    except Exception:
        if run_manifest_path.exists():
            failed = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            failed["status"] = "failed"
            failed["end_utc"] = utc_now()
            run_manifest_path.write_text(
                json.dumps(failed, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        required=True,
        choices=("stock", "task_balanced"),
    )
    parser.add_argument(
        "--split-manifest",
        required=True,
    )
    parser.add_argument(
        "--frozen-root",
        default="cyp/artifacts/cyp_003/step3_split_frozen",
    )
    parser.add_argument(
        "--dataset",
        default="cyp/data/cyp-challenge-TRAIN_inhibition.csv",
    )
    parser.add_argument(
        "--original-split",
        default=(
            "cyp/artifacts/cyp_baseline/"
            "molecule_split_assignments.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    result = run_step3_refit(
        variant=args.variant,
        split_manifest_path=args.split_manifest,
        frozen_root=args.frozen_root,
        dataset_path=args.dataset,
        original_split_path=args.original_split,
        output_dir=args.output_dir,
    )

    print("STEP-3 REFIT COMPLETE")
    print("output:", result["output_dir"])
    print(
        "outer macro ST-RAE:",
        result["outer_score"]["macro_st_rae"],
    )


if __name__ == "__main__":
    main()
