from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from chemprop import data
import torch
from torch import Tensor, nn
from torch.func import functional_call

from cyp002_config import CYP002ModelConfig
from d012_arms import (
    D012_AUX_EPOCHS,
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
from p2_anil_model import build_p2_aggregation, build_p2_message_passing


D012_POP = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_auxiliary_population/d012_task_identity_instances.tsv"
D012_PLAN = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_implementation_qualification"

WARMUP_COMPLETE_UPDATES = 10
MEASURED_COMPLETE_UPDATES = 100
TARGET_AUXILIARY_CELLS = 150
TARGET_MAX_EPOCHS = 50
TARGET_BATCH_SIZE = 16


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


def d012_config(seed: int) -> CYP002ModelConfig:
    return CYP002ModelConfig(
        name="D012-Chemprop-2.3.1-DMPNN",
        message_passing_depth=4,
        message_hidden_dim=600,
        ffn_hidden_dim=2000,
        ffn_num_layers=1,
        dropout=0.4,
        max_lr=D012_MAX_LR,
        batch_size=16,
        max_epochs=50,
        warmup_epochs=10,
        initial_lr=D012_INITIAL_LR,
        final_lr=D012_FINAL_LR,
        seed=seed,
    )


class D012ScalarHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(600, 2000)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(2000, 1)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.output(self.dropout(self.activation(self.hidden(embeddings)))).squeeze(-1)


class D012ChempropEncoder(nn.Module):
    def __init__(self, *, seed: int) -> None:
        super().__init__()
        config = d012_config(seed)
        self.message_passing = build_p2_message_passing(config)
        self.aggregation = build_p2_aggregation(config)

    def forward_dataset(self, dataset: data.MoleculeDataset, *, device: torch.device, batch_size: int) -> Tensor:
        loader = data.build_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )
        pieces = []
        for batch in loader:
            batch_mol_graph = batch[0]
            moved = batch_mol_graph.to(device)
            if moved is not None:
                batch_mol_graph = moved
            atom_embeddings = self.message_passing(batch_mol_graph)
            pieces.append(self.aggregation(atom_embeddings, batch_mol_graph.batch))
        return torch.cat(pieces, dim=0)


class ArmBCompleteModel(nn.Module):
    def __init__(self, task_unit_ids: list[str], *, seed: int) -> None:
        super().__init__()
        self.encoder = D012ChempropEncoder(seed=seed)
        self.heads = nn.ModuleDict({task: D012ScalarHead() for task in sorted(task_unit_ids)})

    def predict(self, dataset: data.MoleculeDataset, task_unit_id: str, *, device: torch.device, batch_size: int) -> Tensor:
        return self.heads[task_unit_id](self.encoder.forward_dataset(dataset, device=device, batch_size=batch_size))


class ArmCCompleteModel(nn.Module):
    def __init__(self, task_unit_ids: list[str], *, seed: int) -> None:
        super().__init__()
        self.encoder = D012ChempropEncoder(seed=seed)
        self.heads = nn.ModuleDict({task: D012ScalarHead() for task in sorted(task_unit_ids)})


@dataclass
class LinearAnchorScheduler:
    optimizer: torch.optim.Optimizer
    warmup_steps: int
    total_steps: int
    step_count: int = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            frac = self.step_count / max(1, self.warmup_steps)
            lr = D012_INITIAL_LR + frac * (D012_MAX_LR - D012_INITIAL_LR)
        else:
            frac = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
            lr = D012_MAX_LR + frac * (D012_FINAL_LR - D012_MAX_LR)
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value > 10_000_000:
        return value / (1024 * 1024)
    return value / 1024


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
        "chemprop_version": __import__("chemprop").__version__,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = 0 if device.index is None else device.index
        result["device_index"] = index
        result["gpu_model"] = torch.cuda.get_device_name(index)
        result["cuda_capability"] = torch.cuda.get_device_capability(index)
    return result


def memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_cuda_allocated_mb": None, "peak_cuda_reserved_mb": None, "max_rss_mb": max_rss_mb()}
    return {
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "max_rss_mb": max_rss_mb(),
    }


def molecule_dataset(smiles_values: list[str], names: list[str] | None = None) -> data.MoleculeDataset:
    if names is None:
        names = [f"mol_{i}" for i in range(len(smiles_values))]
    return data.MoleculeDataset([
        data.MoleculeDatapoint.from_smi(smiles, name=name)
        for smiles, name in zip(smiles_values, names, strict=True)
    ])


def mse(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).square().mean()


def summarize(values: list[float]) -> dict[str, float]:
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
    measured_losses = []
    measured_seconds = []
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
        "optimizer_scheduler_reset_between_warmup_and_measurement": False,
        "warmup_loss_first_last": [warmup_losses[0], warmup_losses[-1]],
        "measured_loss_first_last": [measured_losses[0], measured_losses[-1]],
        "timing": summarize(measured_seconds),
    }


def first_smiles(row: dict[str, str]) -> str:
    values = json.loads(row["contributing_source_smiles"])
    if not values:
        raise ValueError("missing frozen D012 source SMILES")
    return str(values[0])


def load_instances() -> list[dict[str, str]]:
    rows = read_tsv(D012_POP)
    if len(rows) != 4664:
        raise RuntimeError("D012 population changed")
    return rows


def instance_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {f"{row['task_unit_id']}|{row['full_standard_inchikey']}": row for row in rows}


def manifest_removed_keys(manifest_id: str) -> set[str]:
    for row in read_tsv(D012_PLAN / "d012_manifest_firewall.tsv"):
        if row["manifest_id"] == manifest_id:
            return set(split_semicolon(row["removed_full_standard_inchikeys"]))
    raise KeyError(manifest_id)


def select_plan_row(path: Path, manifest_id: str, training_seed: int) -> dict[str, str]:
    for row in read_tsv(path):
        if row["manifest_id"] == manifest_id and int(row["training_seed"]) == int(training_seed):
            return row
    raise KeyError((path, manifest_id, training_seed))


def auxiliary_dataset_from_ids(ids: list[str], lookup: dict[str, dict[str, str]]) -> tuple[data.MoleculeDataset, Tensor, str]:
    rows = [lookup[item] for item in ids]
    task_ids = {row["task_unit_id"] for row in rows}
    if len(task_ids) != 1:
        raise ValueError("benchmark auxiliary update must be task-local")
    dataset = molecule_dataset([first_smiles(row) for row in rows], names=ids)
    targets = torch.tensor([float(row["median_derived_pic50"]) for row in rows], dtype=torch.float32)
    return dataset, targets, next(iter(task_ids))


def target_scaling(rows: list[dict[str, str]], train_names: list[str], task: str) -> tuple[float, float]:
    by_name = {row["Molecule_Name"]: row for row in rows}
    values = [float(by_name[name][task]) for name in train_names if by_name[name].get(task, "") != ""]
    mean = float(statistics.fmean(values))
    scale = float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)
    return mean, scale


def target_dataset(
    rows: list[dict[str, str]],
    names: list[str],
    task: str,
    mean: float,
    scale: float,
    *,
    require_labels: bool,
) -> tuple[data.MoleculeDataset, Tensor, int]:
    by_name = {row["Molecule_Name"]: row for row in rows}
    selected = [name for name in names if (not require_labels) or by_name[name].get(task, "") != ""]
    smiles = [by_name[name]["SMILES"] for name in selected]
    dataset = molecule_dataset(smiles, names=selected)
    labels = torch.empty(0, dtype=torch.float32)
    if require_labels:
        labels = torch.tensor([(float(by_name[name][task]) - mean) / scale for name in selected], dtype=torch.float32)
    return dataset, labels, len(selected)


def adapt_head_exact(head: D012ScalarHead, support_embeddings: Tensor, support_y: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    params = dict(head.named_parameters())
    pred = functional_call(head, params, (support_embeddings,))
    loss = mse(pred, support_y)
    grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=True, retain_graph=True)
    adapted = {
        name: param - D012_INNER_LR * grad
        for (name, param), grad in zip(params.items(), grads, strict=True)
    }
    return adapted, loss


def run_benchmark(*, manifest_id: str, training_seed: int, device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    configure_determinism(stable_seed("D012-complete-model-runtime", manifest_id, training_seed))
    instances = load_instances()
    lookup = instance_lookup(instances)
    task_ids = sorted({row["task_unit_id"] for row in instances})
    budget = scheduler_budget_for_cell(manifest_id, training_seed)
    b_u = int(budget["b_updates_per_epoch"])
    c_u = int(budget["c_updates_per_epoch"])

    b_plan = select_plan_row(D012_PLAN / "d012_arm_b_epoch_plan.tsv", manifest_id, training_seed)
    b_dataset, b_y, b_task = auxiliary_dataset_from_ids(split_semicolon(b_plan["instance_ids"]), lookup)
    b_y = b_y.to(device)

    c_plan = select_plan_row(D012_PLAN / "d012_arm_c_epoch_plan.tsv", manifest_id, training_seed)
    query_ids = split_semicolon(c_plan["query_instance_ids"])
    removed = manifest_removed_keys(manifest_id)
    task_rows = [
        row for row in instances
        if row["task_unit_id"] == c_plan["task_unit_id"]
        and row["full_standard_inchikey"] not in removed
    ]
    query_set = set(query_ids)
    support_ids = [
        f"{row['task_unit_id']}|{row['full_standard_inchikey']}"
        for row in task_rows
        if f"{row['task_unit_id']}|{row['full_standard_inchikey']}" not in query_set
    ]
    support_dataset, support_y, c_task = auxiliary_dataset_from_ids(support_ids, lookup)
    query_dataset, query_y, _ = auxiliary_dataset_from_ids(query_ids, lookup)
    support_y = support_y.to(device)
    query_y = query_y.to(device)

    manifest = load_manifest(manifest_id)
    target_rows = read_csv(TARGET_DATA)
    mean, scale = target_scaling(target_rows, list(manifest["training_names"]), TARGET_TASKS[0])
    train_dataset, train_y, train_observed = target_dataset(target_rows, list(manifest["training_names"]), TARGET_TASKS[0], mean, scale, require_labels=True)
    validation_dataset, validation_y, validation_observed = target_dataset(target_rows, list(manifest["validation_names"]), TARGET_TASKS[0], mean, scale, require_labels=True)
    validation_query_dataset, _empty, validation_input_count = target_dataset(target_rows, list(manifest["validation_names"]), TARGET_TASKS[0], mean, scale, require_labels=False)
    train_y = train_y.to(device)
    validation_y = validation_y.to(device)

    b_model = ArmBCompleteModel(task_ids, seed=training_seed).to(device)
    b_optimizer = torch.optim.Adam(b_model.parameters(), lr=D012_INITIAL_LR)
    b_scheduler = LinearAnchorScheduler(b_optimizer, int(budget["b_warmup_steps"]), int(budget["b_total_steps"]))

    def b_aux_update():
        b_model.train()
        b_optimizer.zero_grad(set_to_none=True)
        loss = mse(b_model.predict(b_dataset, b_task, device=device, batch_size=16), b_y)
        loss.backward()
        b_optimizer.step()
        b_scheduler.step()
        return float(loss.detach().cpu())

    b_aux = measured_loop(device, b_aux_update)

    b_target_head = D012ScalarHead().to(device)
    b_target_optimizer = torch.optim.Adam([*b_model.encoder.parameters(), *b_target_head.parameters()], lr=D012_INITIAL_LR)
    target_updates_per_cell = TARGET_MAX_EPOCHS * math.ceil(train_observed / TARGET_BATCH_SIZE)
    b_target_scheduler = LinearAnchorScheduler(b_target_optimizer, 10 * math.ceil(train_observed / TARGET_BATCH_SIZE), target_updates_per_cell)

    def b_target_update():
        b_model.encoder.train()
        b_target_head.train()
        b_target_optimizer.zero_grad(set_to_none=True)
        pred = b_target_head(b_model.encoder.forward_dataset(train_dataset, device=device, batch_size=16))
        loss = mse(pred, train_y)
        loss.backward()
        b_target_optimizer.step()
        b_target_scheduler.step()
        return float(loss.detach().cpu())

    b_target = measured_loop(device, b_target_update)

    def b_validation_score():
        b_model.encoder.eval()
        b_target_head.eval()
        with torch.no_grad():
            pred = b_target_head(b_model.encoder.forward_dataset(validation_dataset, device=device, batch_size=16))
            return float(mse(pred, validation_y).detach().cpu())

    b_validation = measured_loop(device, b_validation_score)

    c_model = ArmCCompleteModel(task_ids, seed=training_seed).to(device)
    c_optimizer = torch.optim.Adam(c_model.parameters(), lr=D012_INITIAL_LR)
    c_scheduler = LinearAnchorScheduler(c_optimizer, int(budget["c_warmup_steps"]), int(budget["c_total_steps"]))

    def c_aux_update():
        c_model.train()
        c_optimizer.zero_grad(set_to_none=True)
        support_embeddings = c_model.encoder.forward_dataset(support_dataset, device=device, batch_size=16)
        adapted, _support_loss = adapt_head_exact(c_model.heads[c_task], support_embeddings, support_y)
        query_embeddings = c_model.encoder.forward_dataset(query_dataset, device=device, batch_size=16)
        query_loss = mse(functional_call(c_model.heads[c_task], adapted, (query_embeddings,)), query_y)
        query_loss.backward()
        c_optimizer.step()
        c_scheduler.step()
        return float(query_loss.detach().cpu())

    c_aux = measured_loop(device, c_aux_update)

    c_target_head = D012ScalarHead().to(device)

    def c_target_adapt_validate():
        c_model.encoder.eval()
        support_embeddings = c_model.encoder.forward_dataset(train_dataset, device=device, batch_size=16).detach()
        adapted, _support_loss = adapt_head_exact(c_target_head, support_embeddings, train_y)
        validation_embeddings = c_model.encoder.forward_dataset(validation_query_dataset, device=device, batch_size=16).detach()
        pred = functional_call(c_target_head, adapted, (validation_embeddings,))
        if validation_y.numel() == pred.numel():
            return float(mse(pred, validation_y).detach().cpu())
        return float(pred.square().mean().detach().cpu())

    c_target = measured_loop(device, c_target_adapt_validate)

    b_aux_mean = b_aux["timing"]["mean_seconds"]
    c_aux_mean = c_aux["timing"]["mean_seconds"]
    b_target_mean = b_target["timing"]["mean_seconds"]
    b_validation_mean = b_validation["timing"]["mean_seconds"]
    c_target_mean = c_target["timing"]["mean_seconds"]
    b_aux_cell = D012_AUX_EPOCHS * b_u * b_aux_mean
    c_aux_cell = D012_AUX_EPOCHS * c_u * c_aux_mean
    b_target_cell = target_updates_per_cell * b_target_mean + 50 * b_validation_mean
    c_target_cell = c_target_mean

    return {
        "artifact_type": "D012_Chemprop_2_3_1_DMPNN_complete_model_runtime_benchmark_v1",
        "manifest_id": manifest_id,
        "training_seed": training_seed,
        "environment": environment(device),
        "input_hashes": {
            "d012_population": sha256_file(D012_POP),
            "b_epoch_plan": sha256_file(D012_PLAN / "d012_arm_b_epoch_plan.tsv"),
            "c_epoch_plan": sha256_file(D012_PLAN / "d012_arm_c_epoch_plan.tsv"),
            "scheduler_budgets": sha256_file(D012_PLAN / "d012_cell_scheduler_budgets.tsv"),
            "target_data": sha256_file(TARGET_DATA),
        },
        "architecture": {
            "chemprop_version": __import__("chemprop").__version__,
            "message_passing": "Chemprop BondMessagePassing D-MPNN",
            "depth": 4,
            "message_hidden_dim": 600,
            "aggregation": "NormAggregation(norm=100)",
            "ffn_hidden_dim": 2000,
            "ffn_layers": 1,
            "dropout": 0.4,
            "arm_b_heads": "assay/task-specific scalar heads",
            "arm_c_heads": "learned scalar-head initialization with exact K=1 LR=0.01 head-only adaptation",
            "uses_ecfp4": False,
        },
        "qualified_budget": dict(budget),
        "workload_counts": {
            "b_auxiliary_batch_size": len(split_semicolon(b_plan["instance_ids"])),
            "c_support_count": len(support_ids),
            "c_query_count": len(query_ids),
            "target_training_labeled_support_count": train_observed,
            "target_validation_labeled_count": validation_observed,
            "target_validation_input_count": validation_input_count,
        },
        "timed_operations": {
            "included": [
                "Chemprop MoleculeDataset DataLoader graph batching/collation",
                "D-MPNN message-passing forward",
                "NormAggregation forward",
                "D012 FFN/scalar-head forward",
                "MSE loss computation",
                "backward pass where applicable",
                "Adam optimizer step where applicable",
                "scheduler step where applicable",
                "CUDA synchronization around measured timing",
            ],
            "excluded": [
                "TSV/CSV/JSON file loading",
                "qualified firewall/epoch-plan construction",
                "SMILES-to-MoleculeDatapoint preprocessing and dataset object construction",
                "model/module construction",
                "artifact JSON writing",
                "SLURM queue time",
            ],
        },
        "protocol": {
            "warmup_complete_updates": WARMUP_COMPLETE_UPDATES,
            "measured_complete_updates": MEASURED_COMPLETE_UPDATES,
            "optimizer_scheduler_reset_between_warmup_and_measurement": False,
            "cuda_synchronization_around_measured_timing": device.type == "cuda",
        },
        "arm_b_auxiliary_complete_update": b_aux,
        "arm_b_target_transfer_complete_update": b_target,
        "arm_b_validation_scoring_path": b_validation,
        "arm_c_exact_anil_auxiliary_outer_update": c_aux,
        "arm_c_complete_target_support_adaptation_validation_path": c_target,
        "memory": memory(device),
        "projections": {
            "single_cell_seconds": {
                "b_50_epoch_auxiliary": b_aux_cell,
                "c_50_epoch_auxiliary": c_aux_cell,
                "b_target_transfer": b_target_cell,
                "c_target_adaptation": c_target_cell,
                "b_total": b_aux_cell + b_target_cell,
                "c_total": c_aux_cell + c_target_cell,
            },
            "full_150_cell_seconds": {
                "b_auxiliary": 150 * b_aux_cell,
                "c_auxiliary": 150 * c_aux_cell,
                "b_target_transfer": 150 * b_target_cell,
                "c_target_adaptation": 150 * c_target_cell,
                "b_total": 150 * (b_aux_cell + b_target_cell),
                "c_total": 150 * (c_aux_cell + c_target_cell),
            },
            "projection_basis": {
                "b_updates_per_epoch": b_u,
                "c_updates_per_epoch": c_u,
                "auxiliary_epochs": D012_AUX_EPOCHS,
                "b_target_updates_per_cell": target_updates_per_cell,
                "b_validation_checks_per_cell_proxy": 50,
                "c_target_adaptations_per_cell": 1,
                "cell_count": 150,
            },
        },
        "boundaries": {
            "production_training_run": False,
            "tuning_performed": False,
            "methodology_changed": False,
            "d011_modified": False,
            "t121_modified": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D012 Chemprop D-MPNN complete-model runtime benchmark.")
    parser.add_argument("--manifest-id", default="rep_01_fold_01")
    parser.add_argument("--training-seed", default=20261101, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
        device = torch.device("cpu")
    result = run_benchmark(manifest_id=args.manifest_id, training_seed=args.training_seed, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
