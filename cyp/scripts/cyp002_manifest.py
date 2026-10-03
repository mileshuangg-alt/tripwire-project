from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cyp002_config import (
    CYP002_C0,
    CYP002_N_TASKS,
    CYP002_SEED,
    CYP002_TASK_NAMES,
)


MANIFEST_SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    """
    Serialize a JSON-compatible object using the frozen CYP-002
    canonicalization rule.
    """
    result = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return result


def sha256_bytes(value: bytes) -> str:
    result = hashlib.sha256(value).hexdigest()
    return result


def sha256_file(path: str | Path) -> str:
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    result = digest.hexdigest()
    return result


def git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_commit_sha(repo_root: Path) -> str:
    result = git_value(repo_root, "rev-parse", "HEAD")
    return result


def git_repo_dirty(repo_root: Path) -> bool:
    status = git_value(repo_root, "status", "--porcelain")
    return bool(status)


def package_version(module_name: str) -> str:
    module = __import__(module_name)

    version = getattr(module, "__version__", None)

    if version is None:
        try:
            from importlib.metadata import version as package_version_lookup

            version = package_version_lookup(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not determine version for {module_name}"
            ) from exc

    return str(version)


def environment_identity() -> dict[str, Any]:
    import torch

    import lightning.pytorch as pl

    result: dict[str, Any] = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lightning_version": pl.__version__,
        "chemprop_version": package_version("chemprop"),
        "rdkit_version": package_version("rdkit"),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_model": None,
        "cpu_model": platform.processor(),
        "os": platform.platform(),
        "architecture": platform.machine(),
    }

    if torch.cuda.is_available():
        result["gpu_model"] = torch.cuda.get_device_name(0)

    return result


def determinism_identity() -> dict[str, Any]:
    import torch

    result = {
        "torch_use_deterministic_algorithms":
            torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "lightning_deterministic": True,
        "lightning_seed_everything_workers": True,
        "chemprop_dataloader_seed": CYP002_SEED,
        "dataloader_generator_seed": CYP002_SEED,
        "dataloader_workers": 0,
    }

    return result


def canonical_experiment_config(
    config: Any,
    variant: str,
) -> dict[str, Any]:
    if config.name != "C0":
        raise ValueError(
            f"D001 headline manifest requires C0, got {config.name}"
        )

    if config.seed != CYP002_SEED:
        raise ValueError(
            f"C0 seed must be {CYP002_SEED}, got {config.seed}"
        )

    if config.num_tasks != CYP002_N_TASKS:
        raise ValueError(
            f"CYP-002 requires {CYP002_N_TASKS} tasks, "
            f"got {config.num_tasks}"
        )

    if variant not in {"stock", "task_balanced"}:
        raise ValueError(f"Unknown variant: {variant}")

    if variant == "stock":
        loss = {
            "type_per_task": {
                name: "MSE"
                for name in CYP002_TASK_NAMES
            },
            "mask_missing_targets": True,
            "weighting_strategy": "observed_label_count",
            "task_weights": None,
            "zero_label_task_behavior":
                "not_applicable_stock_objective",
            "renormalize_over_present_tasks": False,
            "criterion_implementation": "stock Chemprop MSE",
        }
    else:
        loss = {
            "type_per_task": {
                name: "MSE"
                for name in CYP002_TASK_NAMES
            },
            "mask_missing_targets": True,
            "weighting_strategy": "uniform",
            "task_weights": {
                name: 0.25
                for name in CYP002_TASK_NAMES
            },
            "zero_label_task_behavior": {
                "loss": 0.0,
                "gradient": 0.0,
                "denominator": 4,
                "renormalize_over_present_tasks": False,
            },
            "renormalize_over_present_tasks": False,
            "criterion_implementation":
                "custom D001 TaskBalancedMSE",
        }

    result = {
        "configuration": "C0",
        "variant": variant,
        "seed": CYP002_SEED,
        "architecture": {
            "model": "Chemprop-DMPNN",
            "message_passing": "BondMessagePassing",
            "depth": config.message_passing_depth,
            "message_hidden_dim": config.message_hidden_dim,
            "aggregation": config.aggregation,
            "aggregation_norm": config.aggregation_norm,
            "ffn_hidden_dim": config.ffn_hidden_dim,
            "ffn_num_layers": config.ffn_num_layers,
            "activation": config.activation,
            "dropout": config.dropout,
            "batch_norm": config.batch_norm,
        },
        "loss": loss,
        "training": {
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "optimizer": "Adam",
            "lr_schedule": "NoamLR",
            "warmup_epochs": config.warmup_epochs,
            "initial_lr": config.initial_lr,
            "max_lr": config.max_lr,
            "final_lr": config.final_lr,
            "checkpoint_monitor": "val_macro_st_rae",
            "checkpoint_mode": "min",
            "early_stopping_monitor": "val_macro_st_rae",
            "early_stopping_mode": "min",
            "early_stopping_patience":
                config.early_stopping_patience,
            "early_stopping_min_delta":
                config.early_stopping_min_delta,
            "gradient_clipping": None,
        },
        "target_processing": {
            "per_task_standard_scaler": True,
            "scaler_fit_scope": "training_only",
            "prediction_unscaling": True,
            "missing_target_handling": "masked",
        },
        "data_loading": {
            "workers": 0,
            "shuffle_seed": CYP002_SEED,
        },
        "determinism": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "lightning_deterministic": True,
            "lightning_seed_everything_workers": True,
        },
    }

    return result


def experiment_config_sha256(
    config_object: dict[str, Any],
) -> str:
    result = sha256_bytes(
        canonical_json_bytes(config_object)
    )
    return result


def build_planned_manifest(
    *,
    repo_root: str | Path,
    git_commit_sha_value: str,
    repo_dirty_value: bool,
    dataset_path: str | Path,
    split_path: str | Path,
    dataset_source: str,
    dataset_version_or_export_date: str,
    split_method: str,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    variant: str,
    experiment_id: str,
    invocation_command: str,
    output_checkpoint_path: str | Path,
    output_metrics_path: str | Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)

    dataset_path = Path(dataset_path)
    split_path = Path(split_path)

    config = canonical_experiment_config(
        CYP002_C0,
        variant,
    )

    config_hash = experiment_config_sha256(config)

    environment = environment_identity()

    if environment["hostname"] != "gpu-dev1":
        raise RuntimeError(
            "CYP-002 scientific runs are frozen to gpu-dev1; "
            f"found hostname {environment['hostname']!r}"
        )

    if environment["chemprop_version"] != "2.3.1":
        raise RuntimeError(
            "CYP-002 requires Chemprop 2.3.1; "
            f"found {environment['chemprop_version']!r}"
        )

    dataset_row_count = sum(1 for _ in dataset_path.open("rb")) - 1
    if dataset_row_count != 4905:
        raise ValueError(
            f"Expected 4905 dataset rows, got {dataset_row_count}"
        )

    result = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "configuration": "C0",
        "variant": variant,
        "seed": CYP002_SEED,
        "experiment_config_sha256": config_hash,
        "code_identity": {
            "git_commit_sha": git_commit_sha_value,
            "repo_dirty": repo_dirty_value,
            "chemprop_version": "2.3.1",
        },
        "canonicalization": {
            "json_encoding": "UTF-8",
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
            "trailing_newline": False,
            "hash": "SHA-256",
        },
        "input_artifacts": {
            "dataset": {
                "path": str(dataset_path),
                "source": dataset_source,
                "version_or_export_date":
                    dataset_version_or_export_date,
                "row_count": dataset_row_count,
                "sha256": sha256_file(dataset_path),
            },
            "split": {
                "path": str(split_path),
                "method": split_method,
                "fractions": {
                    "train": train_fraction,
                    "validation": validation_fraction,
                    "test": test_fraction,
                },
                "sha256": sha256_file(split_path),
            },
            "model_code_sha256":
                sha256_file(repo_root / "cyp/scripts/cyp002_model.py"),
            "loss_code_sha256":
                sha256_file(repo_root / "cyp/scripts/multitask_losses.py"),
            "validation_code_sha256":
                sha256_file(
                    repo_root / "cyp/scripts/cyp002_validation.py"
                ),
        },
        "environment": {
            "execution_target": "gpu-dev1",
            **environment,
        },
        "determinism": determinism_identity(),
        "execution": {
            "invocation_command": invocation_command,
            "start_utc": utc_now(),
            "end_utc": None,
            "status": "running",
        },
        "outputs": {
            "best_checkpoint_path": str(output_checkpoint_path),
            "best_checkpoint_sha256": None,
            "metrics_path": str(output_metrics_path),
            "metrics_sha256": None,
        },
    }

    return result


def write_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def finalize_manifest(
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    end_utc: str | None = None,
    status: str = "completed",
) -> None:
    manifest_path = Path(manifest_path)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    manifest["outputs"]["best_checkpoint_sha256"] = (
        sha256_file(checkpoint_path)
    )
    manifest["outputs"]["metrics_sha256"] = (
        sha256_file(metrics_path)
    )

    manifest["execution"]["end_utc"] = (
        utc_now() if end_utc is None else end_utc
    )
    manifest["execution"]["status"] = status

    write_manifest(manifest, manifest_path)


def current_invocation() -> str:
    import sys

    return shlex.join(sys.argv)
