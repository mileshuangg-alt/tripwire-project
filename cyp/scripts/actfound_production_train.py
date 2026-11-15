from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence


PINNED_UPSTREAM_COMMIT = "c96f0559ea7a2504e797b9d65b9f1af1aa899ddd"
PINNED_INPUT = Path(
    "/scratch/user/mhuang8/actfound/chembl32/graph_production_20261113/"
    "actfound_chembl32_graph_qualified.tsv"
)
PINNED_INPUT_SHA256 = "24d9fd297b735843284d82e7d31b85ccd46038f279931db8d5997fb9a096149e"
PINNED_ENV = Path("/scratch/user/mhuang8/actfound/envs/actfound_c96f055")
PINNED_PYTHON = "3.9.18"
PINNED_TORCH = "1.12.1"
PINNED_RDKIT = "2023.03.1"


UPSTREAM_DEFAULTS = {
    "datasource": "chembl",
    "model_name": "actfound",
    "dim_w": 2048,
    "hid_dim": 2048,
    "num_stages": 2,
    "per_step_bn_statistics": True,
    "learnable_bn_gamma": True,
    "learnable_bn_beta": True,
    "enable_inner_loop_optimizable_bn_params": False,
    "learnable_per_layer_per_step_inner_loop_learning_rate": True,
    "use_multi_step_loss_optimization": True,
    "second_order": 1,
    "first_order_to_second_order_epoch": 10,
    "transfer_lr": 0.004,
    "test_sup_num": 16,
    "test_repeat_num": 2,
    "expert_test": "",
    "similarity_cut": 1.0,
    "train_seed": 1111,
    "val_seed": 1111,
    "test_seed": 1111,
    "metatrain_iterations": 60,
    "meta_batch_size": 16,
    "min_learning_rate": 0.0001,
    "update_lr": 0.001,
    "meta_lr": 0.00015,
    "num_updates": 5,
    "test_num_updates": 5,
    "multi_step_loss_num_epochs": 5,
    "norm_layer": "batch_norm",
    "weight_decay": 0.0,
    "alpha": 0.5,
    "resume": 0,
    "train": 1,
    "test_epoch": -1,
    "new_ddg": False,
    "input_celline": False,
    "cross_test": False,
    "gdsc_pretrain": "none",
    "use_byhand_lr": False,
    "inverse_ylabel": False,
    "knn_maml": False,
    "knn_dist_thres": 0.3,
    "begin_lrloss_epoch": 50,
    "lrloss_weight": 35.0,
    "no_fep_lig": False,
    "act_cliff_test": False,
    "global_python_random_seed": 1,
    "global_numpy_random_seed": 2,
    "torch_seed_source_seed": 0,
}

REQUIRED_CONFIG_KEYS = {
    "artifact_type",
    "upstream_commit",
    "frozen_input_tsv",
    "frozen_input_sha256",
    "output_dir",
    "hyperparameters",
    "split_reconciliation_summary_path",
    "split_reconciliation_summary_sha256",
    "split_reconciliation_per_id_path",
    "split_reconciliation_per_id_sha256",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("configuration JSON must be an object")
    return data


def read_assay_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"assay list is missing: {path}")
    values = []
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "\t" in sample or "," in sample:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
            reader = csv.DictReader(handle, dialect=dialect)
            if "assay_id" not in (reader.fieldnames or ()):
                raise ValueError(f"assay list {path} must contain assay_id column")
            values = [row["assay_id"] for row in reader]
        else:
            values = [line.strip() for line in handle if line.strip()]
    if not values:
        raise ValueError(f"assay list is empty: {path}")
    if len(set(values)) != len(values):
        raise ValueError(f"assay list contains duplicates: {path}")
    return values


def validate_split_reconciliation(config: Mapping[str, Any], config_dir: Path) -> dict[str, Any]:
    path = Path(config["split_reconciliation_summary_path"])
    if not path.is_absolute():
        path = config_dir / path
    if not path.exists():
        raise FileNotFoundError(f"split reconciliation summary is missing: {path}")
    expected_sha = config["split_reconciliation_summary_sha256"]
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError("split reconciliation summary SHA256 mismatch")
    summary = read_json(path)
    if summary.get("artifact_type") != "ActFoundSplitReconciliationV1":
        raise ValueError("split reconciliation artifact_type mismatch")
    if summary.get("status") != "complete":
        raise RuntimeError("split reconciliation is not complete")
    if summary.get("frozen_dataset_sha256") != PINNED_INPUT_SHA256:
        raise RuntimeError("split reconciliation frozen dataset SHA256 mismatch")
    if summary.get("unresolved_total") != 0:
        raise RuntimeError("split reconciliation has unresolved IDs")
    if summary.get("ambiguous_total") != 0:
        raise RuntimeError("split reconciliation has ambiguous IDs")
    if summary.get("train_validation_overlap_after_mapping") != 0:
        raise RuntimeError("split reconciliation has train/validation overlap")
    if summary.get("mapped_total") != summary.get("train_split_ids", 0) + summary.get("validation_split_ids", 0):
        raise RuntimeError("split reconciliation mapped_total does not cover train+validation")
    return {"path": str(path), "sha256": actual_sha, "summary": summary}


def validate_split_reconciliation_per_id(
    config: Mapping[str, Any],
    config_dir: Path,
    train_split_ids: Sequence[str],
    val_split_ids: Sequence[str],
) -> dict[str, Any]:
    path = Path(config["split_reconciliation_per_id_path"])
    if not path.is_absolute():
        path = config_dir / path
    if not path.exists():
        raise FileNotFoundError(f"split reconciliation per-ID TSV is missing: {path}")

    expected_sha = config["split_reconciliation_per_id_sha256"]
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError("split reconciliation per-ID SHA256 mismatch")

    train_expected = set(train_split_ids)
    val_expected = set(val_split_ids)
    expected_partition = {split_id: "train" for split_id in train_split_ids}
    expected_partition.update({split_id: "validation" for split_id in val_split_ids})

    required_columns = {"upstream_split_id", "partition", "classification", "mapped_assay_key"}
    train_map: dict[str, str] = {}
    val_map: dict[str, str] = {}
    reverse: dict[str, str] = {}
    unresolved: list[str] = []
    ambiguous: list[str] = []
    duplicated: list[str] = []
    inconsistent: list[str] = []
    unexpected: list[str] = []
    partition_mismatch: list[str] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="	")
        fieldnames = set(reader.fieldnames or ())
        missing_columns = required_columns - fieldnames
        if missing_columns:
            raise ValueError(
                "split reconciliation per-ID TSV missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            split_id = (row.get("upstream_split_id") or "").strip()
            partition = (row.get("partition") or "").strip()
            classification = (row.get("classification") or "").strip().lower()
            mapped = (row.get("mapped_assay_key") or "").strip()

            if not split_id or split_id not in expected_partition:
                unexpected.append(split_id or "<blank>")
                continue
            expected = expected_partition[split_id]
            if partition == "val":
                partition = "validation"
            if partition != expected:
                partition_mismatch.append(split_id)
            if not mapped or "unresolved" in classification or classification == "absent":
                unresolved.append(split_id)
                continue
            if "ambiguous" in classification:
                ambiguous.append(split_id)
                continue

            target = train_map if expected == "train" else val_map
            previous = target.get(split_id)
            if previous is not None:
                duplicated.append(split_id)
                if previous != mapped:
                    inconsistent.append(split_id)
                continue
            owner = reverse.get(mapped)
            if owner is not None and owner != split_id:
                inconsistent.append(split_id)
                continue
            target[split_id] = mapped
            reverse[mapped] = split_id

    missing_train = sorted(train_expected - set(train_map))
    missing_val = sorted(val_expected - set(val_map))
    overlap = set(train_map.values()) & set(val_map.values())
    summary = config["split_reconciliation"]["summary"]

    if missing_train or missing_val or unresolved or ambiguous or duplicated or inconsistent or unexpected or partition_mismatch or overlap:
        raise RuntimeError(
            "split reconciliation per-ID TSV failed closed validation: "
            f"missing_train={len(missing_train)} missing_val={len(missing_val)} "
            f"unresolved={len(unresolved)} ambiguous={len(ambiguous)} "
            f"duplicated={len(duplicated)} inconsistent={len(inconsistent)} "
            f"unexpected={len(unexpected)} partition_mismatch={len(partition_mismatch)} "
            f"train_validation_overlap={len(overlap)}; examples="
            f"{(missing_train or missing_val or unresolved or ambiguous or duplicated or inconsistent or unexpected or partition_mismatch or sorted(overlap))[:10]}"
        )

    if len(train_map) != summary.get("train_mapped"):
        raise RuntimeError("split reconciliation per-ID train count disagrees with summary")
    if len(val_map) != summary.get("validation_mapped"):
        raise RuntimeError("split reconciliation per-ID validation count disagrees with summary")
    if len(train_map) + len(val_map) != summary.get("mapped_total"):
        raise RuntimeError("split reconciliation per-ID mapped count disagrees with summary")

    return {
        "path": str(path),
        "sha256": actual_sha,
        "train_assays": [train_map[split_id] for split_id in train_split_ids],
        "val_assays": [val_map[split_id] for split_id in val_split_ids],
        "counts": {
            "train_resolved": len(train_map),
            "validation_resolved": len(val_map),
            "unresolved": len(unresolved),
            "ambiguous": len(ambiguous),
            "train_validation_overlap": len(overlap),
        },
    }


def resolve_assays(config: Mapping[str, Any], key: str, config_dir: Path) -> list[str]:
    value = config.get(key)
    path_value = config.get(f"{key}_path")
    if value is not None and path_value is not None:
        raise ValueError(f"provide only one of {key} or {key}_path")
    if path_value is not None:
        path = Path(path_value)
        if not path.is_absolute():
            path = config_dir / path
        return read_assay_list(path)
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ValueError(f"{key} must be a non-empty list of assay IDs or {key}_path must be provided")
    if len(set(value)) != len(value):
        raise ValueError(f"{key} contains duplicates")
    return list(value)


def validate_and_resolve_config(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"configuration is missing required keys: {sorted(missing)}")
    if config["artifact_type"] != "ActFoundProductionConfigV1":
        raise ValueError("artifact_type must be ActFoundProductionConfigV1")
    if config["upstream_commit"] != PINNED_UPSTREAM_COMMIT:
        raise ValueError("upstream_commit does not match pinned ActFound commit")

    frozen_input = Path(config["frozen_input_tsv"])
    if frozen_input != PINNED_INPUT:
        raise ValueError("frozen_input_tsv does not match pinned graph-qualified TSV")
    if config["frozen_input_sha256"] != PINNED_INPUT_SHA256:
        raise ValueError("frozen_input_sha256 does not match pinned digest")

    hyper = config["hyperparameters"]
    if not isinstance(hyper, dict):
        raise ValueError("hyperparameters must be an object")
    missing_hyper = set(UPSTREAM_DEFAULTS) - set(hyper)
    extra_hyper = set(hyper) - set(UPSTREAM_DEFAULTS)
    if missing_hyper or extra_hyper:
        raise ValueError(
            "hyperparameters must exactly match the pinned upstream/default key set; "
            f"missing={sorted(missing_hyper)} extra={sorted(extra_hyper)}"
        )
    for key, expected in UPSTREAM_DEFAULTS.items():
        if hyper[key] != expected:
            raise ValueError(f"hyperparameter {key!r} differs from pinned production value")

    config_dir = config_path.parent
    train_assays = resolve_assays(config, "train_assays", config_dir)
    val_assays = resolve_assays(config, "val_assays", config_dir)
    overlap = set(train_assays) & set(val_assays)
    if overlap:
        raise ValueError(f"train/validation assay overlap: {sorted(overlap)[:5]}")

    resolved = dict(config)
    resolved["config_path"] = str(config_path)
    resolved["train_assays"] = train_assays
    resolved["val_assays"] = val_assays
    split_reconciliation = validate_split_reconciliation(resolved, config_dir)
    resolved["split_reconciliation"] = split_reconciliation
    split_reconciliation_per_id = validate_split_reconciliation_per_id(
        resolved,
        config_dir,
        train_assays,
        val_assays,
    )
    resolved["split_reconciliation_per_id"] = split_reconciliation_per_id
    resolved["mapped_train_assays"] = split_reconciliation_per_id["train_assays"]
    resolved["mapped_val_assays"] = split_reconciliation_per_id["val_assays"]
    resolved["output_dir"] = str(Path(config["output_dir"]))
    return resolved


def assert_runtime(upstream_dir: Path, skip_hash: bool = False) -> dict[str, Any]:
    if platform.python_version() != PINNED_PYTHON:
        raise RuntimeError(f"Python runtime must be {PINNED_PYTHON}, got {platform.python_version()}")
    if not PINNED_INPUT.exists():
        raise FileNotFoundError(f"frozen dataset is missing: {PINNED_INPUT}")
    if not skip_hash and file_sha256(PINNED_INPUT) != PINNED_INPUT_SHA256:
        raise RuntimeError("frozen input SHA256 mismatch")
    if not upstream_dir.exists():
        raise FileNotFoundError(f"upstream ActFound directory is missing: {upstream_dir}")
    required_upstream = upstream_dir / "models" / "meta_neural_network_architectures.py"
    if not required_upstream.exists():
        raise FileNotFoundError(f"upstream model source is missing: {required_upstream}")

    import torch
    from rdkit import rdBase

    torch_version = torch.__version__.split("+")[0]
    if torch_version != PINNED_TORCH:
        raise RuntimeError(f"PyTorch runtime must be {PINNED_TORCH}, got {torch.__version__}")
    if rdBase.rdkitVersion != PINNED_RDKIT:
        raise RuntimeError(f"RDKit runtime must be {PINNED_RDKIT}, got {rdBase.rdkitVersion}")
    if not torch.cuda.is_available():
        raise RuntimeError("ActFound production training requires CUDA")

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rdkit": rdBase.rdkitVersion,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
    }


def source_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"required source file missing: {path}")
        result[str(path)] = file_sha256(path)
    return result


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def make_args(hyperparameters: Mapping[str, Any]) -> SimpleNamespace:
    args = SimpleNamespace(**hyperparameters)
    rng_seed = args.torch_seed_source_seed
    import numpy as np
    rng = np.random.RandomState(seed=rng_seed)
    args.rng = rng
    args.logdir = ""
    args.test_write_file = ""
    args.train_assay_feat_all = ""
    args.train_assay_idxes = ""
    args.cell_line_feat = "./datas/gdsc/cellline_to_feat.pkl"
    return args


def seed_like_upstream(args: SimpleNamespace) -> None:
    import numpy as np
    import torch

    random.seed(args.global_python_random_seed)
    np.random.seed(args.global_numpy_random_seed)
    torch_seed = args.rng.randint(0, 999999)
    torch.manual_seed(seed=int(torch_seed))
    torch.backends.cudnn.benchmark = True


def load_dataset(config: Mapping[str, Any]) -> Any:
    from actfound_model_side import ActFoundFrozenMetaDataset

    train_assays = set(config["mapped_train_assays"])
    val_assays = set(config["mapped_val_assays"])

    if len(train_assays) != len(config["train_assays"]):
        raise RuntimeError("mapped train assay count does not match upstream split count")
    if len(val_assays) != len(config["val_assays"]):
        raise RuntimeError("mapped validation assay count does not match upstream split count")

    overlap = train_assays & val_assays
    if overlap:
        raise RuntimeError(
            "train/validation overlap after per-ID reconciliation: "
            f"{sorted(overlap)[:10]}"
        )

    dataset = ActFoundFrozenMetaDataset.from_tsv(
        PINNED_INPUT,
        train_assays=train_assays,
        val_assays=val_assays,
        featurize=True,
    )

    if dataset.data_length["train"] == 0:
        raise RuntimeError("no train assays were loaded from frozen dataset")

    if dataset.data_length["val"] == 0:
        raise RuntimeError("no validation assays were loaded from frozen dataset")

    loaded_train = {
        dataset.assaes[i]
        for i in dataset.train_indices
    }
    loaded_val = {
        dataset.assaes[i]
        for i in dataset.val_indices
    }

    # Train indices contain one original assay plus optional repeated
    # copies named "<assay_key>_<repeat_index>". Test membership against
    # the authoritative mapped assay keys rather than parsing descriptors.
    loaded_train_base = set()
    for assay_key in train_assays:
        if assay_key in loaded_train:
            loaded_train_base.add(assay_key)
            continue
        prefix = assay_key + "_"
        if any(name.startswith(prefix) for name in loaded_train):
            loaded_train_base.add(assay_key)

    missing_train = train_assays - loaded_train_base
    missing_val = val_assays - loaded_val

    if missing_train:
        raise RuntimeError(
            "configured train assays missing from frozen dataset after per-ID reconciliation: "
            f"{sorted(missing_train)[:10]}"
        )

    if missing_val:
        raise RuntimeError(
            "configured validation assays missing from frozen dataset after per-ID reconciliation: "
            f"{sorted(missing_val)[:10]}"
        )

    return dataset


def train_batches(dataset: Any, args: SimpleNamespace, data_epoch: int) -> Iterable[list[int]]:
    import numpy as np

    indices = list(dataset.train_indices)
    rng = np.random.RandomState(seed=args.train_seed + data_epoch)
    rng.shuffle(indices)
    n_batches = len(indices) // args.meta_batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * args.meta_batch_size
        yield indices[start:start + args.meta_batch_size]


def task_arrays(dataset: Any, index: int, args: SimpleNamespace, set_name: str, data_epoch: int) -> tuple[Any, Any, Any]:
    seed_base = {
        "train": args.train_seed,
        "val": args.val_seed,
    }[set_name]
    is_test = set_name != "train"
    return dataset.get_task_split(
        index,
        test_sup_num=args.test_sup_num,
        is_test=is_test,
        rand_seed=seed_base + index + data_epoch,
    )


def per_step_loss_importance(args: SimpleNamespace, current_epoch: int):
    import numpy as np
    import torch

    loss_weights = np.ones(shape=(args.num_updates)) * (1.0 / args.num_updates)
    decay_rate = 1.0 / args.num_updates / args.multi_step_loss_num_epochs
    min_value_for_non_final_losses = 0.03 / args.num_updates
    for i in range(len(loss_weights) - 1):
        loss_weights[i] = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
    loss_weights[-1] = np.minimum(
        loss_weights[-1] + (current_epoch * (args.num_updates - 1) * decay_rate),
        1.0 - ((args.num_updates - 1) * min_value_for_non_final_losses),
    )
    return torch.Tensor(loss_weights).cuda()


def run_task(regressor: Any, inner_lrs: Any, arrays: tuple[Any, Any, Any], args: SimpleNamespace, epoch: int, is_training_phase: bool):
    import torch
    from actfound_model_side import actfound_forward_with_regressor, select_actfound_inner_loop_params

    x_np, y_np, split_np = arrays
    x_task = torch.tensor(x_np).float().cuda()
    y_task = torch.tensor(y_np).float().cuda()
    split = torch.tensor(split_np).cuda()
    if args.inverse_ylabel:
        y_task = -y_task

    names_weights = select_actfound_inner_loop_params(list(regressor.named_parameters()))
    support_losses = []
    task_losses = []
    use_second_order = bool(args.second_order and epoch > args.first_order_to_second_order_epoch)
    loss_weights = per_step_loss_importance(args, epoch)
    sup_num = torch.sum(split)
    num_steps = args.num_updates if is_training_phase else args.test_num_updates

    for num_step in range(num_steps):
        support_loss, _ = actfound_forward_with_regressor(
            regressor,
            x_task,
            y_task,
            split,
            weights=names_weights,
            is_support=True,
            is_training_phase=is_training_phase,
            training=True,
            backup_running_statistics=True if num_step == 0 else False,
            num_step=num_step,
        )
        support_losses.append(float(support_loss.detach().cpu().item()))
        if support_loss >= support_losses[0] * 10 or sup_num <= 5:
            pass
        else:
            grads = torch.autograd.grad(
                support_loss,
                tuple(names_weights.values()),
                create_graph=use_second_order if is_training_phase else False,
            )
            if args.use_byhand_lr:
                lr = [0.012, 0.010, 0.008, 0.004, 0.002, 0.001][num_step]
            else:
                lr = inner_lrs[num_step]
            names_weights = {
                name: value - lr * grad
                for (name, value), grad in zip(names_weights.items(), grads)
            }

        if is_training_phase:
            is_multi_step = args.use_multi_step_loss_optimization and epoch < args.multi_step_loss_num_epochs
            is_last_step = num_step == (args.num_updates - 1)
            if is_multi_step or is_last_step:
                target_loss, _ = actfound_forward_with_regressor(
                    regressor,
                    x_task,
                    y_task,
                    split,
                    weights=names_weights,
                    is_support=False,
                    is_training_phase=True,
                    training=True,
                    backup_running_statistics=False,
                    num_step=num_step,
                )
                task_losses.append(loss_weights[num_step] * target_loss if is_multi_step else target_loss)

    target_loss, target_preds = actfound_forward_with_regressor(
        regressor,
        x_task,
        y_task,
        split,
        weights=names_weights,
        is_support=False,
        is_training_phase=is_training_phase,
        training=True,
        backup_running_statistics=False,
        num_step=num_steps - 1,
    )
    if not is_training_phase:
        task_losses.append(target_loss)
    if hasattr(regressor, "restore_backup_stats") and not is_training_phase:
        regressor.restore_backup_stats()
    return torch.sum(torch.stack(task_losses)), target_loss, target_preds, split, y_task, support_losses, names_weights


def metric_from_task(y_task: Any, target_preds: Any, split: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    sup_idx = torch.nonzero(split)[:, 0]
    tgt_idx = torch.nonzero(1.0 - split)[:, 0]
    y_true_t = y_task[tgt_idx]
    y_train_mean = torch.mean(y_task[sup_idx]).cpu().item()
    y_true = y_true_t.cpu().numpy()
    y_pred = target_preds.detach().cpu().numpy()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = np.corrcoef(y_true, y_pred)[0, 1]
    if math.isnan(r2) or r2 < 0.0:
        r2 = 0.0
    else:
        r2 = float(r2 ** 2)
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - y_train_mean) ** 2).sum(axis=0, dtype=np.float64)
    r2os = 0.0 if denominator == 0 else float(1.0 - (numerator / denominator))
    return {"r2": float(r2), "rmse": rmse, "R2os": r2os, "y_train_mean": y_train_mean}


def evaluate_validation(regressor: Any, inner_lrs: Any, dataset: Any, args: SimpleNamespace, repeat_cnt: int) -> dict[str, Any]:
    import numpy as np
    import torch

    regressor.eval()
    r2_values = []
    r2os_values = []
    rmse_values = []
    with torch.enable_grad():
        for index in dataset.val_indices:
            arrays = task_arrays(dataset, index, args, "val", repeat_cnt)
            _, target_loss, preds, split, y_task, support_losses, _ = run_task(
                regressor,
                inner_lrs,
                arrays,
                args,
                epoch=0,
                is_training_phase=False,
            )
            metric = metric_from_task(y_task, preds, split)
            r2_values.append(metric["r2"])
            r2os_values.append(metric["R2os"])
            rmse_values.append(metric["rmse"])
    mean_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")
    mean_r2 = float(np.mean(r2_values)) if r2_values else float("nan")
    mean_r2os = float(np.mean(r2os_values)) if r2os_values else float("nan")
    return {
        "mean_rmse": mean_rmse,
        "mean_r2": mean_r2,
        "mean_R2os": mean_r2os,
        "score": mean_r2 - mean_rmse + 1.0,
        "assay_count": len(rmse_values),
    }


def save_checkpoint(path: Path, regressor: Any, inner_lrs: Any, optimizer: Any, scheduler: Any, epoch: int, config: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "upstream_commit": PINNED_UPSTREAM_COMMIT,
            "config_path": config["config_path"],
            "regressor_state_dict": regressor.state_dict(),
            "inner_lrs": inner_lrs.detach().cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        str(path),
    )


def run_training(config: Mapping[str, Any], repo_root: Path, upstream_dir: Path, skip_hash: bool = False) -> None:
    runtime = assert_runtime(upstream_dir, skip_hash=skip_hash)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    completion_path = output_dir / "completion.json"

    source_paths = [
        repo_root / "cyp" / "scripts" / "actfound_model_side.py",
        repo_root / "cyp" / "scripts" / "actfound_production_train.py",
        upstream_dir / "models" / "meta_neural_network_architectures.py",
        upstream_dir / "learning_system" / "system_actfound.py",
        upstream_dir / "learning_system" / "system_base.py",
        upstream_dir / "inner_loop_optimizers.py",
    ]
    startup = {
        "artifact_type": "ActFoundProductionStartupV1",
        "config": config,
        "runtime": runtime,
        "source_hashes": source_hashes(source_paths),
        "frozen_input_sha256": file_sha256(PINNED_INPUT) if not skip_hash else PINNED_INPUT_SHA256,
        "started_at_unix": time.time(),
    }
    (output_dir / "startup_config.json").write_text(json.dumps(startup, indent=2, sort_keys=True) + "\n")

    import numpy as np
    import torch

    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR

    args = make_args(config["hyperparameters"])
    seed_like_upstream(args)
    dataset = load_dataset(config)

    regressor = FCNReLUNormNetworkQSAR(input_shape=(2, args.dim_w), args=args, meta=True).cuda()
    inner_lrs = torch.nn.Parameter(
        torch.ones(args.num_updates + 1).cuda() * args.update_lr,
        requires_grad=bool(args.learnable_per_layer_per_step_inner_loop_learning_rate),
    )
    optimizer = torch.optim.Adam(list(regressor.parameters()) + [inner_lrs], lr=args.meta_lr, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.metatrain_iterations,
        eta_min=args.min_learning_rate,
    )
    best_score = float("-inf")
    best_checkpoint = None

    append_jsonl(progress_path, {"event": "dataset_loaded", "train_indices": len(dataset.train_indices), "val_indices": len(dataset.val_indices)})
    initial_val = evaluate_validation(regressor, inner_lrs, dataset, args, repeat_cnt=0)
    append_jsonl(progress_path, {"event": "validation", "epoch": 0, "phase": "initial", **initial_val})

    for epoch in range(args.metatrain_iterations):
        regressor.train()
        scheduler.step(epoch=epoch)
        data_epoch = epoch * args.meta_batch_size
        batch_losses = []
        for batch_idx, batch_indices in enumerate(train_batches(dataset, args, data_epoch)):
            total_losses = []
            for index in batch_indices:
                arrays = task_arrays(dataset, index, args, "train", data_epoch)
                task_loss, _, _, _, _, _, _ = run_task(regressor, inner_lrs, arrays, args, epoch=epoch, is_training_phase=True)
                total_losses.append(task_loss)
            loss = torch.mean(torch.stack(total_losses))
            optimizer.zero_grad()
            regressor.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            regressor.zero_grad()
            batch_losses.append(float(loss.detach().cpu().item()))
            append_jsonl(progress_path, {"event": "train_batch", "epoch": epoch, "batch": batch_idx, "loss": batch_losses[-1]})

        val = evaluate_validation(regressor, inner_lrs, dataset, args, repeat_cnt=0)
        epoch_checkpoint = output_dir / "checkpoints" / f"model_{epoch}.pt"
        save_checkpoint(epoch_checkpoint, regressor, inner_lrs, optimizer, scheduler, epoch, config)
        if val["score"] > best_score:
            best_score = val["score"]
            best_checkpoint = output_dir / "checkpoints" / "model_best.pt"
            save_checkpoint(best_checkpoint, regressor, inner_lrs, optimizer, scheduler, epoch, config)
        append_jsonl(
            progress_path,
            {
                "event": "epoch_complete",
                "epoch": epoch,
                "mean_train_loss": float(np.mean(batch_losses)) if batch_losses else None,
                "checkpoint": str(epoch_checkpoint),
                "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
                **val,
            },
        )

    completion = {
        "artifact_type": "ActFoundProductionCompletionV1",
        "status": "completed",
        "completed_at_unix": time.time(),
        "best_score": best_score,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "progress_path": str(progress_path),
    }
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--upstream-dir", type=Path, default=Path(__file__).resolve().parents[2] / "external" / "ActFound")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-input-hash", action="store_true", help="Only for bounded smoke tests; never use for production.")
    args = parser.parse_args(argv)

    config = validate_and_resolve_config(args.config)
    if args.validate_only:
        print(json.dumps({"status": "validated", "config_path": str(args.config)}, sort_keys=True))
        return 0
    run_training(config, args.repo_root, args.upstream_dir, skip_hash=args.skip_input_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
