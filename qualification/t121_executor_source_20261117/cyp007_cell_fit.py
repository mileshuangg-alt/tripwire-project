from __future__ import annotations

import argparse
from pathlib import Path



def route_d007_fold(
    *,
    dataframe,
    original_split_dataframe,
    split_manifest: dict,
    training_seed: int,
):
    """
    Route the authoritative CYP dataset into one frozen D007
    development-training / development-validation fold.
    """
    import pandas as pd

    from cyp007_infrastructure import D007_TRAINING_SEEDS

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if not isinstance(original_split_dataframe, pd.DataFrame):
        raise TypeError(
            "original_split_dataframe must be a pandas DataFrame."
        )

    if training_seed not in D007_TRAINING_SEEDS:
        raise ValueError(
            f"Unexpected D007 training seed: {training_seed}"
        )

    if tuple(split_manifest["training_seeds"]) != D007_TRAINING_SEEDS:
        raise ValueError(
            "D007 split manifest has unexpected training seeds."
        )

    required_split_columns = {
        "Molecule_Name",
        "cluster_id",
        "split",
    }

    if set(original_split_dataframe.columns) != required_split_columns:
        raise ValueError(
            "Original split must contain exactly "
            "Molecule_Name, cluster_id, split."
        )

    if dataframe["Molecule_Name"].duplicated().any():
        raise ValueError(
            "Authoritative CYP dataset contains duplicate Molecule_Name."
        )

    if original_split_dataframe["Molecule_Name"].duplicated().any():
        raise ValueError(
            "Original split contains duplicate Molecule_Name."
        )

    dataset_names = set(dataframe["Molecule_Name"])
    split_names = set(original_split_dataframe["Molecule_Name"])

    if dataset_names != split_names:
        raise ValueError(
            "Dataset and original split do not contain the same "
            "molecular universe."
        )

    protected_test_names = set(
        original_split_dataframe.loc[
            original_split_dataframe["split"] == "test",
            "Molecule_Name",
        ]
    )

    training_names = list(split_manifest["training_names"])
    validation_names = list(split_manifest["validation_names"])

    training_set = set(training_names)
    validation_set = set(validation_names)

    if training_set & validation_set:
        raise ValueError(
            "D007 training and validation partitions overlap."
        )

    if (training_set | validation_set) & protected_test_names:
        raise RuntimeError(
            "Protected internal-test molecule entered D007 fold."
        )

    development_names = set(
        original_split_dataframe.loc[
            original_split_dataframe["split"].isin(
                ["train", "validation"]
            ),
            "Molecule_Name",
        ]
    )

    if training_set | validation_set != development_names:
        raise ValueError(
            "D007 fold does not exactly equal the frozen "
            "development universe."
        )

    if len(training_names) != split_manifest["training_count"]:
        raise ValueError(
            "D007 training count does not match split manifest."
        )

    if len(validation_names) != split_manifest["validation_count"]:
        raise ValueError(
            "D007 validation count does not match split manifest."
        )

    cluster_by_name = original_split_dataframe.set_index(
        "Molecule_Name"
    )["cluster_id"]

    training_clusters = {
        cluster_by_name[name]
        for name in training_names
    }

    validation_clusters = {
        cluster_by_name[name]
        for name in validation_names
    }

    if training_clusters & validation_clusters:
        raise ValueError(
            "A frozen molecular cluster crosses D007 "
            "training/validation."
        )

    indexed = dataframe.set_index(
        "Molecule_Name",
        drop=False,
        verify_integrity=True,
    )

    training_df = indexed.loc[training_names].copy()
    validation_df = indexed.loc[validation_names].copy()

    result = {
        "training_dataframe": training_df,
        "validation_dataframe": validation_df,
        "training_seed": training_seed,
        "repetition": int(split_manifest["repetition"]),
        "fold": int(split_manifest["fold"]),
    }

    return result


def build_d007_prefit_state(
    *,
    training_dataframe,
    validation_dataframe,
    sampled_config: dict,
    config_name: str,
    training_seed: int,
):
    """
    Build one frozen D007 train/validation pre-fit state.

    Target scaling is fit on development-training targets only and
    applied unchanged to development-validation targets.
    """
    import numpy as np
    from chemprop import data

    from cyp002_config import (
        CYP002_TASK_NAMES,
        CYP002_VARIANT_STOCK,
    )
    from cyp002_model import build_cyp002_model
    from cyp002_train import (
        _configure_determinism,
        _frame_to_molecule_dataset,
    )
    from cyp007_infrastructure import (
        D007_TRAINING_SEEDS,
        config_from_trial,
    )

    if training_seed not in D007_TRAINING_SEEDS:
        raise ValueError(
            f"Unexpected D007 training seed: {training_seed}"
        )

    config = config_from_trial(
        name=config_name,
        sampled=sampled_config,
        seed=training_seed,
    )

    _configure_determinism(training_seed)

    training_dataset = _frame_to_molecule_dataset(
        training_dataframe
    )
    validation_dataset = _frame_to_molecule_dataset(
        validation_dataframe
    )

    scaler = training_dataset.normalize_targets()
    validation_dataset.normalize_targets(scaler)

    train_loader = data.build_dataloader(
        training_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=training_seed,
        shuffle=True,
    )

    validation_loader = data.build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        seed=training_seed,
        shuffle=False,
        drop_last=False,
    )

    validation_conf_low = {
        task: validation_dataframe[
            f"{task}_conf_low"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        for task in CYP002_TASK_NAMES
    }

    validation_conf_high = {
        task: validation_dataframe[
            f"{task}_conf_high"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        for task in CYP002_TASK_NAMES
    }

    model = build_cyp002_model(
        config,
        CYP002_VARIANT_STOCK,
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
        "validation_conf_low": validation_conf_low,
        "validation_conf_high": validation_conf_high,
        "training_seed": training_seed,
        "variant": CYP002_VARIANT_STOCK,
    }

    return result


def fit_d007_cell(
    *,
    prefit_state: dict,
    checkpoint_dir,
):
    """
    Fit one D007 split x seed cell.

    The scientific endpoint is the validation-selected best.ckpt and
    its best development-validation macro ST-RAE.
    """
    from pathlib import Path

    from lightning import pytorch as pl

    from cyp002_validation import build_validation_callbacks

    checkpoint_dir = Path(checkpoint_dir)

    required_keys = {
        "config",
        "model",
        "train_loader",
        "validation_loader",
        "training_seed",
        "variant",
    }

    missing = required_keys - set(prefit_state)

    if missing:
        raise ValueError(
            "D007 prefit state is missing keys: "
            f"{sorted(missing)}"
        )

    if checkpoint_dir.exists():
        if any(checkpoint_dir.iterdir()):
            raise RuntimeError(
                "D007 checkpoint directory is not empty; refusing "
                f"to overwrite prior artifacts: {checkpoint_dir}"
            )
    else:
        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=False,
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

    trainer.fit(
        model,
        train_dataloaders=prefit_state["train_loader"],
        val_dataloaders=prefit_state["validation_loader"],
    )

    checkpoint_path = checkpoint_dir / "best.ckpt"

    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Validation-selected D007 checkpoint was not created: "
            f"{checkpoint_path}"
        )

    validation_history = getattr(
        model,
        "validation_history",
        None,
    )

    if not validation_history:
        raise RuntimeError(
            "No D007 validation ST-RAE history was recorded."
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

def _read_json(path):
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text())


def _write_json_atomic(path, payload):
    import json
    import os
    from pathlib import Path

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _write_json_exclusive(path, payload):
    import json
    import os
    from pathlib import Path

    path = Path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256_file(path):
    import hashlib
    from pathlib import Path

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_paths(run_root, trial_index, cell_index):
    from pathlib import Path

    trial_dir = Path(run_root) / "trials" / f"trial_{trial_index:03d}"
    cell_dir = trial_dir / "cells" / f"cell_{cell_index:03d}"
    return trial_dir, cell_dir


def run_production_cell(
    *,
    run_root,
    trial_index,
    cell_index,
    dataset_path,
    original_split_path,
    split_manifest_root,
):
    import socket
    import time
    from pathlib import Path

    import pandas as pd

    run_root = Path(run_root)
    trial_dir, cell_dir = _cell_paths(run_root, trial_index, cell_index)
    plan_path = cell_dir / "cell_plan.json"
    proposal_path = trial_dir / "proposal.json"
    result_path = cell_dir / "result.json"
    complete_path = cell_dir / "COMPLETED.json"
    started_path = cell_dir / "STARTED.json"
    incomplete_path = cell_dir / "INCOMPLETE.json"

    if complete_path.exists() and result_path.exists():
        return _read_json(result_path)
    if started_path.exists() and not complete_path.exists():
        raise RuntimeError(
            "D007 cell has STARTED without COMPLETED; human adjudication required."
        )
    if result_path.exists() or complete_path.exists():
        raise RuntimeError("D007 cell has partial result evidence; refusing overwrite.")

    plan = _read_json(plan_path)
    proposal = _read_json(proposal_path)
    if plan["trial_index"] != trial_index or plan["cell_index"] != cell_index:
        raise RuntimeError("D007 cell identity does not match its frozen plan.")
    sampled_config = proposal.get("sampled_config")
    if not isinstance(sampled_config, dict):
        raise RuntimeError("D007 proposal is missing frozen sampled_config.")
    config_name = proposal.get("config_name", f"D007-T{trial_index:03d}")

    split_manifest_path = (
        Path(split_manifest_root)
        / f"rep_{plan['repetition']:02d}_fold_{plan['fold']:02d}.json"
    )
    split_manifest = _read_json(split_manifest_path)
    if tuple(split_manifest["training_seeds"]) and plan["seed"] not in split_manifest["training_seeds"]:
        raise RuntimeError("D007 cell seed is not in the frozen split manifest.")
    if split_manifest["repetition"] != plan["repetition"] or split_manifest["fold"] != plan["fold"]:
        raise RuntimeError("D007 split manifest does not match cell plan.")

    _write_json_exclusive(started_path, {
        "status": "started",
        "trial_index": trial_index,
        "cell_index": cell_index,
        "repetition": plan["repetition"],
        "fold": plan["fold"],
        "seed": plan["seed"],
        "hostname": socket.gethostname(),
        "started_time": time.time(),
        "split_manifest_path": str(split_manifest_path),
        "proposal_path": str(proposal_path),
    })

    try:
        dataframe = pd.read_csv(dataset_path)
        original_split = pd.read_csv(original_split_path)
        routed = route_d007_fold(
            dataframe=dataframe,
            original_split_dataframe=original_split,
            split_manifest=split_manifest,
            training_seed=plan["seed"],
        )
        prefit = build_d007_prefit_state(
            training_dataframe=routed["training_dataframe"],
            validation_dataframe=routed["validation_dataframe"],
            sampled_config=sampled_config,
            config_name=config_name,
            training_seed=plan["seed"],
        )
        fit = fit_d007_cell(
            prefit_state=prefit,
            checkpoint_dir=cell_dir / "fit",
        )
        checkpoint_path = Path(fit["checkpoint_path"])
        best_validation = fit["best_validation"]
        result = {
            "trial_index": trial_index,
            "cell_index": cell_index,
            "repetition": plan["repetition"],
            "fold": plan["fold"],
            "seed": plan["seed"],
            "status": "completed",
            "macro_st_rae": best_validation["macro_st_rae"],
            "best_validation": best_validation,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "split_manifest_path": str(split_manifest_path),
            "split_manifest_sha256": _sha256_file(split_manifest_path),
            "proposal_path": str(proposal_path),
            "config_name": config_name,
        }
        provenance = {
            "result": result,
            "sampled_config": sampled_config,
            "epochs_completed": fit["epochs_completed"],
            "validation_history": fit["validation_history"],
            "dataset_path": str(dataset_path),
            "dataset_sha256": _sha256_file(dataset_path),
            "original_split_path": str(original_split_path),
            "original_split_sha256": _sha256_file(original_split_path),
        }
        _write_json_atomic(cell_dir / "provenance.json", provenance)
        _write_json_exclusive(result_path, result)
        _write_json_exclusive(complete_path, {
            "status": "completed",
            "result_sha256": _sha256_file(result_path),
            "provenance_sha256": _sha256_file(cell_dir / "provenance.json"),
        })
        return result
    except BaseException as exc:
        if not incomplete_path.exists():
            _write_json_atomic(incomplete_path, {
                "status": "incomplete",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "requires_human_adjudication": True,
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--cell-index", required=True, type=int)
    parser.add_argument(
        "--dataset",
        default="cyp/data/cyp-challenge-TRAIN_inhibition.csv",
    )
    parser.add_argument(
        "--original-split",
        default="cyp/artifacts/cyp_baseline/molecule_split_assignments.csv",
    )
    parser.add_argument(
        "--split-manifest-root",
        default="cyp/artifacts/cyp_007/split_manifests",
    )
    args = parser.parse_args()
    result = run_production_cell(
        run_root=args.run_root,
        trial_index=args.trial_index,
        cell_index=args.cell_index,
        dataset_path=args.dataset,
        original_split_path=args.original_split,
        split_manifest_root=args.split_manifest_root,
    )
    print(result)


if __name__ == "__main__":
    main()
