from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import socket
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from chemprop import data
import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call

from cyp002_validation import compute_validation_macro_st_rae
from d012_arms import (
    D012_FINAL_LR,
    D012_INITIAL_LR,
    D012_INNER_LR,
    D012_MAX_LR,
    ROOT,
    TARGET_DATA,
    TARGET_TASKS,
    configure_determinism,
    load_manifest,
    scheduler_budget_for_cell,
    split_semicolon,
    stable_seed,
)
from d012_complete_model_runtime_benchmark import (
    D012ChempropEncoder,
    D012ScalarHead,
    LinearAnchorScheduler,
    adapt_head_exact,
    d012_config,
    molecule_dataset,
    mse,
)


QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
D012_POP = QUAL / "d012_auxiliary_population/d012_task_identity_instances.tsv"
D012_PLAN = QUAL / "d012_implementation_qualification"
MANIFEST_ID = "rep_01_fold_01"
TRAINING_SEED = 20261101
TARGET_BATCH_SIZE = 16
TARGET_MAX_EPOCHS = 50
TARGET_PATIENCE = 20


class D012BTargetHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(600, 2000)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(2000, len(TARGET_TASKS))

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.output(self.dropout(self.activation(self.hidden(embeddings))))


class ArmBProductionModel(nn.Module):
    def __init__(self, task_unit_ids: list[str], *, seed: int) -> None:
        super().__init__()
        self.encoder = D012ChempropEncoder(seed=seed)
        self.auxiliary_heads = nn.ModuleDict({task: D012ScalarHead() for task in sorted(task_unit_ids)})


class ArmCProductionModel(nn.Module):
    def __init__(self, *, seed: int) -> None:
        super().__init__()
        self.encoder = D012ChempropEncoder(seed=seed)
        self.scalar_head_initialization = D012ScalarHead()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value > 10_000_000:
        return value / (1024 * 1024)
    return value / 1024


def memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_cuda_allocated_mb": None, "peak_cuda_reserved_mb": None, "max_rss_mb": max_rss_mb()}
    return {
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "max_rss_mb": max_rss_mb(),
    }


def environment(device: torch.device) -> dict[str, object]:
    result = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "chemprop": __import__("chemprop").__version__,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = 0 if device.index is None else device.index
        result["gpu_model"] = torch.cuda.get_device_name(index)
        result["cuda_capability"] = torch.cuda.get_device_capability(index)
    return result


def first_smiles(row: dict[str, str]) -> str:
    values = json.loads(row["contributing_source_smiles"])
    if not values:
        raise ValueError("D012 instance lacks contributing source SMILES")
    return str(values[0])


def load_instances() -> list[dict[str, str]]:
    rows = read_tsv(D012_POP)
    for row in rows:
        row["instance_id"] = f"{row['task_unit_id']}|{row['full_standard_inchikey']}"
    if len(rows) != 4664:
        raise RuntimeError("D012 auxiliary population changed")
    return rows


def instance_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["instance_id"]: row for row in rows}


def plan_rows(path: Path, *, manifest_id: str = MANIFEST_ID, training_seed: int = TRAINING_SEED) -> list[dict[str, str]]:
    return [
        row for row in read_tsv(path)
        if row["manifest_id"] == manifest_id and int(row["training_seed"]) == int(training_seed)
    ]


def retained_instance_ids_from_b_plan(
    manifest_id: str = MANIFEST_ID,
    training_seed: int = TRAINING_SEED,
) -> set[str]:
    ids = set()
    for row in plan_rows(
        D012_PLAN / "d012_arm_b_epoch_plan.tsv",
        manifest_id=manifest_id,
        training_seed=training_seed,
    ):
        ids.update(split_semicolon(row["instance_ids"]))
    return ids


def cell_workload_assertions(manifest_id: str, training_seed: int) -> dict[str, object]:
    instances = load_instances()
    retained_ids = retained_instance_ids_from_b_plan(manifest_id, training_seed)
    retained = [row for row in instances if row["instance_id"] in retained_ids]
    task_units = {row["task_unit_id"] for row in retained}
    b_rows = plan_rows(D012_PLAN / "d012_arm_b_epoch_plan.tsv", manifest_id=manifest_id, training_seed=training_seed)
    c_rows = plan_rows(D012_PLAN / "d012_arm_c_epoch_plan.tsv", manifest_id=manifest_id, training_seed=training_seed)
    b_budget = scheduler_budget_for_cell(manifest_id, training_seed)
    c_query_ids = {
        item for row in c_rows for item in split_semicolon(row["query_instance_ids"])
    }
    return {
        "manifest_id": manifest_id,
        "training_seed": training_seed,
        "retained_auxiliary_instances": len(retained_ids),
        "surviving_task_units": len(task_units),
        "b_updates_per_aux_epoch": len(b_rows),
        "b_total_aux_updates_50_epochs": 50 * len(b_rows),
        "c_updates_per_aux_epoch": len(c_rows),
        "c_total_aux_updates_50_epochs": 50 * len(c_rows),
        "b_c_same_masked_auxiliary_population": retained_ids == c_query_ids,
        "qualified_budget": b_budget,
        "minimum_task_size_filter_applied": False,
        "validation_labels_enter_auxiliary_training": False,
        "protected_test_or_challenge_labels_accessed": False,
    }


def first_cell_workload_assertions() -> dict[str, object]:
    return cell_workload_assertions(MANIFEST_ID, TRAINING_SEED)


class ArtifactWriter:
    def __init__(self, output_dir: Path, *, arm: str) -> None:
        self.output_dir = output_dir
        self.arm = arm
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = output_dir / "progress.jsonl"
        self.timings: dict[str, float] = {}

    def write_json(self, name: str, value: object) -> None:
        (self.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def event(self, event: str, **payload: object) -> None:
        record = {"event": event, "time": time.time(), **payload}
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def time_phase(self, name: str):
        writer = self

        class Timer:
            def __enter__(self):
                self.start = time.perf_counter()
                writer.event("phase_start", phase=name)
                return self

            def __exit__(self, exc_type, exc, tb):
                elapsed = time.perf_counter() - self.start
                writer.timings[name] = writer.timings.get(name, 0.0) + elapsed
                writer.event("phase_end", phase=name, seconds=elapsed)

        return Timer()

    def checkpoint(self, name: str, state: dict[str, object]) -> str:
        path = self.output_dir / name
        torch.save(state, path)
        digest = sha256_file(path)
        self.event("checkpoint_written", path=name, sha256=digest)
        return digest

    def finalize_hashes(self) -> None:
        paths = sorted(path for path in self.output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
        with (self.output_dir / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
            for path in paths:
                handle.write(f"{sha256_file(path)}  {path.name}\n")


def build_startup_config(
    arm: str,
    device: torch.device,
    *,
    manifest_id: str,
    training_seed: int,
) -> dict[str, object]:
    return {
        "artifact_type": "D012_first_cell_production_startup_v1",
        "arm": arm,
        "manifest_id": manifest_id,
        "training_seed": training_seed,
        "environment": environment(device),
        "inputs": {
            "d012_population": {"path": str(D012_POP.relative_to(ROOT)), "sha256": sha256_file(D012_POP)},
            "b_epoch_plan": {"path": str((D012_PLAN / "d012_arm_b_epoch_plan.tsv").relative_to(ROOT)), "sha256": sha256_file(D012_PLAN / "d012_arm_b_epoch_plan.tsv")},
            "c_epoch_plan": {"path": str((D012_PLAN / "d012_arm_c_epoch_plan.tsv").relative_to(ROOT)), "sha256": sha256_file(D012_PLAN / "d012_arm_c_epoch_plan.tsv")},
            "scheduler_budgets": {"path": str((D012_PLAN / "d012_cell_scheduler_budgets.tsv").relative_to(ROOT)), "sha256": sha256_file(D012_PLAN / "d012_cell_scheduler_budgets.tsv")},
            "target_data": {"path": str(TARGET_DATA.relative_to(ROOT)), "sha256": sha256_file(TARGET_DATA)},
        },
        "workload_assertions": cell_workload_assertions(manifest_id, training_seed),
        "method_boundaries": {
            "d011_modified": False,
            "t121_retrained_or_retuned": False,
            "arrays_submitted": False,
            "methodology_changed": False,
        },
    }


def dataset_for_instances(instance_ids: list[str], lookup: dict[str, dict[str, str]]) -> tuple[data.MoleculeDataset, Tensor, str]:
    rows = [lookup[item] for item in instance_ids]
    task_ids = {row["task_unit_id"] for row in rows}
    if len(task_ids) != 1:
        raise ValueError("auxiliary update must be task-local")
    dataset = molecule_dataset([first_smiles(row) for row in rows], names=instance_ids)
    labels = torch.tensor([float(row["median_derived_pic50"]) for row in rows], dtype=torch.float32)
    return dataset, labels, next(iter(task_ids))


def target_matrices(rows: list[dict[str, str]], names: list[str]) -> tuple[data.MoleculeDataset, Tensor, Tensor, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_name = {row["Molecule_Name"]: row for row in rows}
    dataset = molecule_dataset([by_name[name]["SMILES"] for name in names], names=names)
    values = np.full((len(names), len(TARGET_TASKS)), np.nan, dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    lows: dict[str, np.ndarray] = {}
    highs: dict[str, np.ndarray] = {}
    y_true: dict[str, np.ndarray] = {}
    for j, task in enumerate(TARGET_TASKS):
        for i, name in enumerate(names):
            value = by_name[name].get(task, "")
            if value != "":
                values[i, j] = float(value)
                mask[i, j] = True
        lows[task] = np.asarray([float(by_name[name][f"{task}_conf_low"]) for name in names if by_name[name].get(task, "") != ""], dtype=np.float64)
        highs[task] = np.asarray([float(by_name[name][f"{task}_conf_high"]) for name in names if by_name[name].get(task, "") != ""], dtype=np.float64)
        y_true[task] = np.asarray([float(by_name[name][task]) for name in names if by_name[name].get(task, "") != ""], dtype=np.float64)
    return dataset, torch.tensor(values), torch.tensor(mask), y_true, lows, highs


def fit_target_scaler(train_values: Tensor, train_mask: Tensor) -> tuple[Tensor, Tensor]:
    means = []
    scales = []
    for j in range(train_values.shape[1]):
        observed = train_values[train_mask[:, j], j]
        mean = observed.mean()
        scale = observed.std(unbiased=False)
        if float(scale) <= 0:
            raise ValueError("nonpositive target scaler")
        means.append(mean)
        scales.append(scale)
    return torch.stack(means), torch.stack(scales)


def masked_mse(pred: Tensor, target_z: Tensor, mask: Tensor) -> Tensor:
    keep = mask & torch.isfinite(target_z)
    if not bool(keep.any()):
        raise ValueError("empty target batch")
    return (pred[keep] - target_z[keep]).square().mean()


def predict_target(encoder: D012ChempropEncoder, head: nn.Module, dataset: data.MoleculeDataset, *, device: torch.device) -> Tensor:
    return head(encoder.forward_dataset(dataset, device=device, batch_size=TARGET_BATCH_SIZE))


def score_validation(
    encoder: D012ChempropEncoder,
    head: D012BTargetHead,
    dataset: data.MoleculeDataset,
    values: Tensor,
    mask: Tensor,
    means: Tensor,
    scales: Tensor,
    y_true: dict[str, np.ndarray],
    lows: dict[str, np.ndarray],
    highs: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, object]:
    encoder.eval()
    head.eval()
    with torch.no_grad():
        pred_z = predict_target(encoder, head, dataset, device=device).cpu()
    pred = pred_z * scales.cpu() + means.cpu()
    predictions = {}
    for j, task in enumerate(TARGET_TASKS):
        keep = mask[:, j].detach().cpu().numpy().astype(bool)
        predictions[task] = pred[keep, j].numpy().astype(np.float64)
    return compute_validation_macro_st_rae(y_true, predictions, lows, highs, TARGET_TASKS)


def run_arm_b(
    writer: ArtifactWriter,
    device: torch.device,
    *,
    manifest_id: str,
    training_seed: int,
    max_aux_epochs: int,
    max_target_epochs: int,
) -> None:
    instances = load_instances()
    lookup = instance_lookup(instances)
    task_ids = sorted({row["task_unit_id"] for row in instances})
    b_rows = plan_rows(D012_PLAN / "d012_arm_b_epoch_plan.tsv", manifest_id=manifest_id, training_seed=training_seed)
    budget = scheduler_budget_for_cell(manifest_id, training_seed)

    with writer.time_phase("model_construction"):
        model = ArmBProductionModel(task_ids, seed=training_seed).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=D012_INITIAL_LR)
        scheduler = LinearAnchorScheduler(optimizer, int(budget["b_warmup_steps"]), int(budget["b_total_steps"]))

    with writer.time_phase("auxiliary_training"):
        update = 0
        for epoch in range(1, max_aux_epochs + 1):
            for row in b_rows:
                dataset, labels, task = dataset_for_instances(split_semicolon(row["instance_ids"]), lookup)
                labels = labels.to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                pred = model.auxiliary_heads[task](model.encoder.forward_dataset(dataset, device=device, batch_size=16))
                loss = (pred - labels).square().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                update += 1
            writer.event("arm_b_aux_epoch_complete", epoch=epoch, updates=update)
        writer.checkpoint("arm_b_auxiliary_final.pt", {"encoder": model.encoder.state_dict(), "auxiliary_heads": model.auxiliary_heads.state_dict(), "updates": update})

    with writer.time_phase("target_preparation"):
        manifest = load_manifest(manifest_id)
        target_rows = read_csv(TARGET_DATA)
        train_dataset, train_values, train_mask, _train_true, _train_low, _train_high = target_matrices(target_rows, list(manifest["training_names"]))
        val_dataset, val_values, val_mask, val_true, val_low, val_high = target_matrices(target_rows, list(manifest["validation_names"]))
        means, scales = fit_target_scaler(train_values, train_mask)
        train_z = (train_values - means) / scales
        train_z[~train_mask] = float("nan")
        val_values = val_values.to(device)
        val_mask = val_mask.to(device)
        means = means.to(device)
        scales = scales.to(device)

    with writer.time_phase("target_training_checkpoint_selection"):
        target_head = D012BTargetHead().to(device)
        target_optimizer = torch.optim.Adam([*model.encoder.parameters(), *target_head.parameters()], lr=D012_INITIAL_LR)
        batches = list(range(0, train_values.shape[0], TARGET_BATCH_SIZE))
        target_scheduler = LinearAnchorScheduler(target_optimizer, 10 * len(batches), TARGET_MAX_EPOCHS * len(batches))
        best_score = math.inf
        best_epoch = None
        bad = 0
        history = []
        for epoch in range(1, max_target_epochs + 1):
            for start in batches:
                stop = min(start + TARGET_BATCH_SIZE, train_values.shape[0])
                names = list(manifest["training_names"])[start:stop]
                batch_dataset = data.MoleculeDataset([train_dataset.data[i] for i in range(start, stop)])
                batch_y = train_z[start:stop].to(device)
                batch_mask = train_mask[start:stop].to(device)
                target_optimizer.zero_grad(set_to_none=True)
                pred = predict_target(model.encoder, target_head, batch_dataset, device=device)
                loss = masked_mse(pred, batch_y, batch_mask)
                loss.backward()
                target_optimizer.step()
                target_scheduler.step()
            score = score_validation(model.encoder, target_head, val_dataset, val_values, val_mask, means, scales, val_true, val_low, val_high, device=device)
            macro = float(score["macro_st_rae"])
            history.append({"epoch": epoch, "macro_st_rae": macro})
            writer.event("arm_b_validation", epoch=epoch, macro_st_rae=macro)
            if macro < best_score:
                best_score = macro
                best_epoch = epoch
                bad = 0
                writer.checkpoint("arm_b_best_target_checkpoint.pt", {"encoder": model.encoder.state_dict(), "target_head": target_head.state_dict(), "epoch": epoch, "macro_st_rae": macro})
            else:
                bad += 1
                if bad >= TARGET_PATIENCE:
                    break
        writer.write_json("arm_b_target_metrics.json", {"best_epoch": best_epoch, "best_macro_st_rae": best_score, "history": history, "validation_checkpoint_selection": True})


def run_arm_c(
    writer: ArtifactWriter,
    device: torch.device,
    *,
    manifest_id: str,
    training_seed: int,
    max_aux_epochs: int,
) -> None:
    instances = load_instances()
    lookup = instance_lookup(instances)
    task_ids = sorted({row["task_unit_id"] for row in instances})
    task_to_ids = defaultdict(list)
    for instance_id in retained_instance_ids_from_b_plan(manifest_id, training_seed):
        task_to_ids[lookup[instance_id]["task_unit_id"]].append(instance_id)
    c_rows = plan_rows(D012_PLAN / "d012_arm_c_epoch_plan.tsv", manifest_id=manifest_id, training_seed=training_seed)
    budget = scheduler_budget_for_cell(manifest_id, training_seed)

    with writer.time_phase("model_construction"):
        model = ArmCProductionModel(seed=training_seed).to(device)
        optimizer = torch.optim.Adam([*model.encoder.parameters(), *model.scalar_head_initialization.parameters()], lr=D012_INITIAL_LR)
        scheduler = LinearAnchorScheduler(optimizer, int(budget["c_warmup_steps"]), int(budget["c_total_steps"]))

    with writer.time_phase("auxiliary_training"):
        update = 0
        for epoch in range(1, max_aux_epochs + 1):
            for row in c_rows:
                query_ids = split_semicolon(row["query_instance_ids"])
                query_set = set(query_ids)
                support_ids = [item for item in task_to_ids[row["task_unit_id"]] if item not in query_set]
                support_dataset, support_y, _task = dataset_for_instances(support_ids, lookup)
                query_dataset, query_y, _task = dataset_for_instances(query_ids, lookup)
                support_y = support_y.to(device)
                query_y = query_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                support_embeddings = model.encoder.forward_dataset(support_dataset, device=device, batch_size=16)
                adapted, _support_loss = adapt_head_exact(model.scalar_head_initialization, support_embeddings, support_y)
                query_embeddings = model.encoder.forward_dataset(query_dataset, device=device, batch_size=16)
                query_loss = mse(functional_call(model.scalar_head_initialization, adapted, (query_embeddings,)), query_y)
                query_loss.backward()
                optimizer.step()
                scheduler.step()
                update += 1
            writer.event("arm_c_aux_epoch_complete", epoch=epoch, updates=update)
        writer.checkpoint("arm_c_auxiliary_final.pt", {"encoder": model.encoder.state_dict(), "scalar_head_initialization": model.scalar_head_initialization.state_dict(), "updates": update})

    with writer.time_phase("target_preparation"):
        manifest = load_manifest(manifest_id)
        target_rows = read_csv(TARGET_DATA)
        train_dataset, train_values, train_mask, _train_true, _train_low, _train_high = target_matrices(target_rows, list(manifest["training_names"]))
        val_dataset, val_values, val_mask, val_true, val_low, val_high = target_matrices(target_rows, list(manifest["validation_names"]))
        means, scales = fit_target_scaler(train_values, train_mask)
        train_z = (train_values - means) / scales
        train_z[~train_mask] = float("nan")
        val_predictions = {}
        model.encoder.eval()

    with writer.time_phase("target_one_step_adaptation_and_validation"):
        for j, task in enumerate(TARGET_TASKS):
            keep = train_mask[:, j]
            support_dataset = data.MoleculeDataset([train_dataset.data[i] for i, flag in enumerate(keep.tolist()) if flag])
            support_y = train_z[keep, j].to(device)
            head = D012ScalarHead().to(device)
            head.load_state_dict(model.scalar_head_initialization.state_dict())
            support_embeddings = model.encoder.forward_dataset(support_dataset, device=device, batch_size=TARGET_BATCH_SIZE).detach()
            adapted, _support_loss = adapt_head_exact(head, support_embeddings, support_y)
            val_embeddings = model.encoder.forward_dataset(val_dataset, device=device, batch_size=TARGET_BATCH_SIZE).detach()
            pred_z = functional_call(head, adapted, (val_embeddings,)).detach().cpu()
            pred = pred_z * scales[j].cpu() + means[j].cpu()
            val_predictions[task] = pred[val_mask[:, j]].numpy().astype(np.float64)
        score = compute_validation_macro_st_rae(val_true, val_predictions, val_low, val_high, TARGET_TASKS)
        writer.write_json("arm_c_validation_metrics.json", {"score": score, "validation_checkpoint_selection": False, "target_update": "one_head_only_step_lr_0.01_k1_encoder_frozen"})
        writer.checkpoint("arm_c_final_transfer_state.pt", {"encoder": model.encoder.state_dict(), "scalar_head_initialization": model.scalar_head_initialization.state_dict(), "validation_score": score})


def run(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest_id)
    if int(args.training_seed) not in set(manifest["training_seeds"]):
        raise ValueError("training seed is not present in the frozen D007 manifest")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cpu")
    configure_determinism(stable_seed("D012-production-cell", args.arm, args.manifest_id, args.training_seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    writer = ArtifactWriter(args.output_dir, arm=args.arm)
    start = time.perf_counter()
    writer.write_json(
        "startup_config.json",
        build_startup_config(
            args.arm,
            device,
            manifest_id=args.manifest_id,
            training_seed=args.training_seed,
        ),
    )
    writer.event("startup", arm=args.arm)
    try:
        if args.fail_after_startup:
            raise RuntimeError("intentional failure after startup for artifact-preservation test")
        if args.preflight_only:
            writer.write_json(
                "preflight_complete.json",
                cell_workload_assertions(args.manifest_id, args.training_seed),
            )
        elif args.arm == "B":
            run_arm_b(
                writer,
                device,
                manifest_id=args.manifest_id,
                training_seed=args.training_seed,
                max_aux_epochs=args.max_aux_epochs,
                max_target_epochs=args.max_target_epochs,
            )
        else:
            run_arm_c(
                writer,
                device,
                manifest_id=args.manifest_id,
                training_seed=args.training_seed,
                max_aux_epochs=args.max_aux_epochs,
            )
        writer.write_json("completion.json", {"status": "completed", "arm": args.arm, "elapsed_seconds": time.perf_counter() - start, "phase_timings_seconds": writer.timings, "memory": memory_snapshot(device)})
        writer.event("completed")
    except Exception as exc:
        writer.write_json("failure.json", {"status": "failed", "arm": args.arm, "error": repr(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - start, "phase_timings_seconds": writer.timings, "memory": memory_snapshot(device)})
        writer.event("failed", error=repr(exc))
        raise
    finally:
        writer.finalize_hashes()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run D012 first production cell for Arm B or C.")
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    parser.add_argument("--manifest-id", default=MANIFEST_ID)
    parser.add_argument("--training-seed", default=TRAINING_SEED, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fail-after-startup", action="store_true")
    parser.add_argument("--max-aux-epochs", default=50, type=int)
    parser.add_argument("--max-target-epochs", default=50, type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
