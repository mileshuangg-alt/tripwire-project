from __future__ import annotations

import argparse
import math
import json
import platform
import resource
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from d012_arms import (
    D012_AUX_EPOCHS,
    D012_INITIAL_LR,
    D012_FINAL_LR,
    D012_MAX_LR,
    D012_WARMUP_EPOCHS,
    TARGET_DATA,
    TARGET_TASKS,
    ArmBMultitaskModel,
    ArmCAnilModel,
    auxiliary_batch_from_instance_ids,
    arm_b_auxiliary_update,
    arm_b_target_update,
    arm_c_exact_outer_update,
    arm_c_target_adapt_and_score,
    configure_determinism,
    fit_training_only_target_scaling,
    fresh_target_head,
    load_d012_instances,
    load_manifest,
    read_csv,
    rows_by_instance_id,
    scheduler_budget_for_cell,
    select_b_minibatch,
    select_c_episode,
    split_semicolon,
    stable_seed,
    target_xy,
)


WARMUP_COMPLETE_UPDATES = 10
MEASURED_COMPLETE_UPDATES = 100
TARGET_BATCH_SIZE = 16
TARGET_AUXILIARY_CELLS = 150
TARGET_MAX_EPOCHS = 50
TARGET_VALIDATION_CHECKS = 50


@dataclass
class LinearAnchorScheduler:
    optimizer: torch.optim.Optimizer
    warmup_steps: int
    total_steps: int
    step_count: int = 0

    def step(self) -> None:
        self.step_count += 1
        if self.total_steps <= 1:
            lr = D012_FINAL_LR
        elif self.step_count <= self.warmup_steps:
            frac = self.step_count / max(1, self.warmup_steps)
            lr = D012_INITIAL_LR + frac * (D012_MAX_LR - D012_INITIAL_LR)
        else:
            cooldown = max(1, self.total_steps - self.warmup_steps)
            frac = min(1.0, (self.step_count - self.warmup_steps) / cooldown)
            lr = D012_MAX_LR + frac * (D012_FINAL_LR - D012_MAX_LR)
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value > 10_000_000:
        return value / (1024 * 1024)
    return value / 1024


def environment(device: torch.device) -> dict[str, object]:
    result = {
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "visible_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = 0 if device.index is None else device.index
        result.update({
            "device_index": index,
            "gpu_model": torch.cuda.get_device_name(index),
            "cuda_capability": torch.cuda.get_device_capability(index),
        })
    return result


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "peak_cuda_allocated_mb": None,
            "peak_cuda_reserved_mb": None,
            "max_rss_mb": max_rss_mb(),
        }
    return {
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "max_rss_mb": max_rss_mb(),
    }


def summarize_seconds(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "total_measured_seconds": sum(values),
    }


def measured_loop(device: torch.device, update_fn) -> dict[str, object]:
    warmup_losses = []
    measured_seconds = []
    measured_losses = []
    for _ in range(WARMUP_COMPLETE_UPDATES):
        warmup_losses.append(float(update_fn()))
    for _ in range(MEASURED_COMPLETE_UPDATES):
        sync(device)
        start = time.perf_counter()
        loss = float(update_fn())
        sync(device)
        measured_seconds.append(time.perf_counter() - start)
        measured_losses.append(loss)
    return {
        "warmup_complete_updates": WARMUP_COMPLETE_UPDATES,
        "measured_complete_updates": MEASURED_COMPLETE_UPDATES,
        "warmup_loss_first_last": [warmup_losses[0], warmup_losses[-1]],
        "measured_loss_first_last": [measured_losses[0], measured_losses[-1]],
        "timing": summarize_seconds(measured_seconds),
        "protocol": "10 continuous warmup complete updates followed by 100 measured complete updates; optimizer/scheduler not reset",
    }


def benchmark_context(manifest_id: str, training_seed: int, device: torch.device):
    configure_determinism(stable_seed("D012-runtime-benchmark", manifest_id, training_seed))
    instances = load_d012_instances()
    lookup = rows_by_instance_id(instances)
    task_ids = sorted({row["task_unit_id"] for row in instances})
    budget = scheduler_budget_for_cell(manifest_id, training_seed)

    b_plan = select_b_minibatch(manifest_id, training_seed)
    aux_x, aux_y, aux_task = auxiliary_batch_from_instance_ids(split_semicolon(b_plan["instance_ids"]), lookup)
    aux_x = aux_x.to(device)
    aux_y = aux_y.to(device)

    c_plan = select_c_episode(manifest_id, training_seed)
    query_ids = split_semicolon(c_plan["query_instance_ids"])
    query_set = set(query_ids)
    c_task_rows = [row for row in lookup.values() if row["task_unit_id"] == c_plan["task_unit_id"]]
    support_ids = [
        f"{row['task_unit_id']}|{row['full_standard_inchikey']}"
        for row in c_task_rows
        if f"{row['task_unit_id']}|{row['full_standard_inchikey']}" not in query_set
    ]
    support_x, support_y, c_task = auxiliary_batch_from_instance_ids(support_ids, lookup)
    query_x, query_y, _ = auxiliary_batch_from_instance_ids(query_ids, lookup)
    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)

    manifest = load_manifest(manifest_id)
    target_rows = read_csv(TARGET_DATA)
    scaling = fit_training_only_target_scaling(target_rows, set(manifest["training_names"]), TARGET_TASKS[0])
    train_names, train_x, train_y = target_xy(target_rows, list(manifest["training_names"]), TARGET_TASKS[0], scaling, limit=TARGET_BATCH_SIZE)
    validation_names, val_x, val_y = target_xy(target_rows, list(manifest["validation_names"]), TARGET_TASKS[0], scaling, limit=TARGET_BATCH_SIZE)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    return {
        "task_ids": task_ids,
        "budget": budget,
        "b_auxiliary": (aux_x, aux_y, aux_task),
        "c_auxiliary": (support_x, support_y, query_x, query_y, c_task),
        "target": (train_x, train_y, val_x, val_y),
        "target_counts": {
            "train_batch_examples": len(train_names),
            "validation_examples": len(validation_names),
            "manifest_training_count": int(manifest["training_count"]),
            "manifest_validation_count": int(manifest["validation_count"]),
        },
    }


def run_benchmark(*, manifest_id: str, training_seed: int, device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ctx = benchmark_context(manifest_id, training_seed, device)
    budget = ctx["budget"]
    b_u = int(budget["b_updates_per_epoch"])
    c_u = int(budget["c_updates_per_epoch"])
    task_ids = ctx["task_ids"]

    b_model = ArmBMultitaskModel(task_ids).to(device)
    b_optimizer = torch.optim.Adam(b_model.parameters(), lr=D012_INITIAL_LR)
    b_scheduler = LinearAnchorScheduler(b_optimizer, int(budget["b_warmup_steps"]), int(budget["b_total_steps"]))
    aux_x, aux_y, aux_task = ctx["b_auxiliary"]

    def b_aux_update():
        loss = arm_b_auxiliary_update(b_model, b_optimizer, aux_x, aux_y, aux_task)
        b_scheduler.step()
        return loss

    b_aux = measured_loop(device, b_aux_update)

    train_x, train_y, val_x, val_y = ctx["target"]
    b_target_head = fresh_target_head().to(device)
    b_target_optimizer = torch.optim.Adam([*b_model.encoder.parameters(), *b_target_head.parameters()], lr=D012_INITIAL_LR)
    b_target_scheduler = LinearAnchorScheduler(b_target_optimizer, 10 * math.ceil(len(train_y) / TARGET_BATCH_SIZE), TARGET_MAX_EPOCHS * math.ceil(len(train_y) / TARGET_BATCH_SIZE))

    def b_target_update():
        loss = arm_b_target_update(b_model.encoder, b_target_head, b_target_optimizer, train_x, train_y)
        b_target_scheduler.step()
        return loss

    b_target = measured_loop(device, b_target_update)

    c_model = ArmCAnilModel(task_ids).to(device)
    c_optimizer = torch.optim.Adam(c_model.parameters(), lr=D012_INITIAL_LR)
    c_scheduler = LinearAnchorScheduler(c_optimizer, int(budget["c_warmup_steps"]), int(budget["c_total_steps"]))
    support_x, support_y, query_x, query_y, c_task = ctx["c_auxiliary"]

    def c_outer_update():
        result = arm_c_exact_outer_update(c_model, c_optimizer, c_task, support_x, support_y, query_x, query_y)
        c_scheduler.step()
        return result["query_loss"]

    c_aux = measured_loop(device, c_outer_update)

    c_target_head = fresh_target_head().to(device)

    def c_target_adapt():
        result = arm_c_target_adapt_and_score(c_model.encoder, c_target_head, train_x, train_y, val_x, val_y)
        return result["validation_score"]

    c_target = measured_loop(device, c_target_adapt)

    projections = project_workload(
        b_aux_seconds=b_aux["timing"]["mean_seconds"],
        c_aux_seconds=c_aux["timing"]["mean_seconds"],
        b_target_update_seconds=b_target["timing"]["mean_seconds"],
        c_target_adapt_seconds=c_target["timing"]["mean_seconds"],
        b_u=b_u,
        c_u=c_u,
        target_updates_per_cell=TARGET_MAX_EPOCHS * math.ceil(
            ctx["target_counts"]["manifest_training_count"] / TARGET_BATCH_SIZE
        ),
    )
    return {
        "artifact_type": "D012_complete_update_runtime_benchmark_v2",
        "manifest_id": manifest_id,
        "training_seed": training_seed,
        "environment": environment(device),
        "memory": memory_snapshot(device),
        "qualified_scheduler_budget": budget,
        "target_counts": ctx["target_counts"],
        "protocol": {
            "warmup_complete_updates": WARMUP_COMPLETE_UPDATES,
            "measured_complete_updates": MEASURED_COMPLETE_UPDATES,
            "optimizer_scheduler_reset_between_warmup_and_measurement": False,
            "cuda_synchronization_around_measured_timing": device.type == "cuda",
        },
        "arm_b_auxiliary_update": b_aux,
        "arm_b_target_transfer_update": b_target,
        "arm_c_auxiliary_outer_update": c_aux,
        "arm_c_target_adaptation": c_target,
        "projections": projections,
        "method_boundaries": {
            "production_training_run": False,
            "slurm_submission_performed": False,
            "hyperparameter_tuning_performed": False,
            "methodology_changed": False,
        },
    }


def project_workload(
    *,
    b_aux_seconds: float,
    c_aux_seconds: float,
    b_target_update_seconds: float,
    c_target_adapt_seconds: float,
    b_u: int,
    c_u: int,
    target_updates_per_cell: int,
) -> dict[str, object]:
    b_aux_cell = D012_AUX_EPOCHS * b_u * b_aux_seconds
    c_aux_cell = D012_AUX_EPOCHS * c_u * c_aux_seconds
    b_target_cell = target_updates_per_cell * b_target_update_seconds
    c_target_cell = c_target_adapt_seconds
    return {
        "basis": {
            "b_updates_per_epoch": b_u,
            "c_updates_per_epoch": c_u,
            "auxiliary_epochs": D012_AUX_EPOCHS,
            "target_cells": TARGET_AUXILIARY_CELLS,
            "b_target_updates_per_cell_proxy": target_updates_per_cell,
            "c_target_adaptations_per_cell": 1,
        },
        "single_cell_seconds": {
            "b_50_epoch_auxiliary": b_aux_cell,
            "c_50_epoch_auxiliary": c_aux_cell,
            "b_target_transfer": b_target_cell,
            "c_target_adaptation": c_target_cell,
            "b_total": b_aux_cell + b_target_cell,
            "c_total": c_aux_cell + c_target_cell,
        },
        "full_150_cell_seconds": {
            "b_auxiliary": TARGET_AUXILIARY_CELLS * b_aux_cell,
            "c_auxiliary": TARGET_AUXILIARY_CELLS * c_aux_cell,
            "b_target_transfer": TARGET_AUXILIARY_CELLS * b_target_cell,
            "c_target_adaptation": TARGET_AUXILIARY_CELLS * c_target_cell,
            "b_total": TARGET_AUXILIARY_CELLS * (b_aux_cell + b_target_cell),
            "c_total": TARGET_AUXILIARY_CELLS * (c_aux_cell + c_target_cell),
        },
        "projection_caveat": "Projection scales measured bounded complete-update timings by qualified update counts; production qualification depends on running this benchmark on the intended CoreHPC L40S hardware.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D012 B/C complete-update runtime benchmark.")
    parser.add_argument("--manifest-id", default="rep_01_fold_01")
    parser.add_argument("--training-seed", default=20261101, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA benchmark requested but CUDA is unavailable; refusing CPU projection without --allow-cpu")
        device = torch.device("cpu")
    result = run_benchmark(manifest_id=args.manifest_id, training_seed=args.training_seed, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
