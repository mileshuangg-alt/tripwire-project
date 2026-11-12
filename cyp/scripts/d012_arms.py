from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from ecfp4 import N_BITS, ecfp4_matrix


ROOT = Path(os.environ.get("TRIPWIRE_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
QUAL = ROOT / "qualification/cyp_external_task_qualification_20261109"
D012_POP = QUAL / "d012_auxiliary_population/d012_task_identity_instances.tsv"
D012_PLAN = QUAL / "d012_implementation_qualification"
D007_MANIFESTS = ROOT / "cyp/artifacts/cyp_007/split_manifests"
TARGET_DATA = ROOT / "cyp/data/cyp-challenge-TRAIN_inhibition.csv"

D012_EMBEDDING_DIM = 600
D012_DROPOUT = 0.4
D012_INNER_LR = 0.01
D012_INNER_STEPS = 1
D012_AUX_EPOCHS = 50
D012_WARMUP_EPOCHS = 10
D012_INITIAL_LR = 1.4283739994469765e-05
D012_MAX_LR = 0.0008245400010426866
D012_FINAL_LR = 0.00020141638331344468

TARGET_TASKS = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_semicolon(value: str) -> list[str]:
    return [] if value == "" else value.split(";")


def first_smiles(row: dict[str, str]) -> str:
    values = json.loads(row["contributing_source_smiles"])
    if not values:
        raise ValueError("D012 identity has no contributing source SMILES")
    return str(values[0])


def tensor_features_from_smiles(smiles_values: Iterable[str]) -> Tensor:
    arr = ecfp4_matrix(list(smiles_values)).astype(np.float32)
    return torch.from_numpy(arr)


class D012Encoder(nn.Module):
    """Bounded D012 encoder module for correctness qualification.

    Production complete-model runtime qualification is prepared separately;
    these modules exercise the declared optimization and transfer semantics on
    real differentiable graphs while consuming the frozen D012 plans.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(N_BITS, D012_EMBEDDING_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(D012_DROPOUT)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.projection(x.float())))


class ScalarHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(D012_EMBEDDING_DIM, 1)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.linear(embeddings).squeeze(-1)


class ArmBMultitaskModel(nn.Module):
    def __init__(self, task_unit_ids: Iterable[str]) -> None:
        super().__init__()
        self.encoder = D012Encoder()
        self.heads = nn.ModuleDict({task: ScalarHead() for task in sorted(set(task_unit_ids))})

    def predict(self, features: Tensor, task_unit_id: str) -> Tensor:
        return self.heads[task_unit_id](self.encoder(features))


class ArmCAnilModel(nn.Module):
    def __init__(self, task_unit_ids: Iterable[str]) -> None:
        super().__init__()
        self.encoder = D012Encoder()
        self.heads = nn.ModuleDict({task: ScalarHead() for task in sorted(set(task_unit_ids))})

    def embeddings(self, features: Tensor) -> Tensor:
        return self.encoder(features)


@dataclass(frozen=True)
class TargetScaling:
    mean: float
    scale: float
    train_names: tuple[str, ...]
    validation_names: tuple[str, ...]


def fit_training_only_target_scaling(rows: list[dict[str, str]], train_names: set[str], task: str) -> TargetScaling:
    train_values = [float(row[task]) for row in rows if row["Molecule_Name"] in train_names and row.get(task, "") != ""]
    if len(train_values) < 2:
        raise ValueError("target scaling requires at least two training labels")
    mean = float(np.mean(train_values))
    scale = float(np.std(train_values, ddof=0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("target scaling must be finite and positive")
    return TargetScaling(mean=mean, scale=scale, train_names=tuple(sorted(train_names)), validation_names=())


def target_xy(
    rows: list[dict[str, str]],
    names: list[str],
    task: str,
    scaling: TargetScaling,
    *,
    limit: int | None = None,
) -> tuple[list[str], Tensor, Tensor]:
    by_name = {row["Molecule_Name"]: row for row in rows}
    selected_names = [name for name in names if by_name[name].get(task, "") != ""]
    if limit is not None:
        selected_names = selected_names[:limit]
    x = tensor_features_from_smiles(by_name[name]["SMILES"] for name in selected_names)
    y = torch.tensor(
        [(float(by_name[name][task]) - scaling.mean) / scaling.scale for name in selected_names],
        dtype=torch.float32,
    )
    return selected_names, x, y


def mse(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")
    return (pred - target).square().mean()


def clone_parameter_tensors(module: nn.Module) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def parameter_changed(before: dict[str, Tensor], module: nn.Module, prefix: str) -> bool:
    for name, parameter in module.named_parameters():
        if name.startswith(prefix) and not torch.equal(before[name], parameter.detach()):
            return True
    return False


def parameter_unchanged(before: dict[str, Tensor], module: nn.Module, prefix: str) -> bool:
    for name, parameter in module.named_parameters():
        if name.startswith(prefix) and not torch.equal(before[name], parameter.detach()):
            return False
    return True


def load_d012_instances() -> list[dict[str, str]]:
    rows = read_tsv(D012_POP)
    if len(rows) != 4664:
        raise RuntimeError("D012 auxiliary population count changed")
    return rows


def rows_by_instance_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {f"{row['task_unit_id']}|{row['full_standard_inchikey']}": row for row in rows}


def load_manifest(manifest_id: str) -> dict[str, object]:
    return json.loads((D007_MANIFESTS / f"{manifest_id}.json").read_text(encoding="utf-8"))


def scheduler_budget_for_cell(manifest_id: str, training_seed: int) -> dict[str, str]:
    for row in read_tsv(D012_PLAN / "d012_cell_scheduler_budgets.tsv"):
        if row["manifest_id"] == manifest_id and int(row["training_seed"]) == int(training_seed):
            return row
    raise KeyError((manifest_id, training_seed))


def select_b_minibatch(manifest_id: str, training_seed: int, *, task_unit_id: str | None = None) -> dict[str, str]:
    for row in read_tsv(D012_PLAN / "d012_arm_b_epoch_plan.tsv"):
        if row["manifest_id"] != manifest_id or int(row["training_seed"]) != int(training_seed):
            continue
        if task_unit_id is None or row["task_unit_id"] == task_unit_id:
            return row
    raise KeyError((manifest_id, training_seed, task_unit_id))


def select_c_episode(manifest_id: str, training_seed: int, *, min_query_count: int = 2) -> dict[str, str]:
    for row in read_tsv(D012_PLAN / "d012_arm_c_epoch_plan.tsv"):
        if row["manifest_id"] == manifest_id and int(row["training_seed"]) == int(training_seed) and int(row["query_count"]) >= min_query_count:
            return row
    raise KeyError((manifest_id, training_seed, min_query_count))


def auxiliary_batch_from_instance_ids(instance_ids: list[str], lookup: dict[str, dict[str, str]]) -> tuple[Tensor, Tensor, str]:
    rows = [lookup[item] for item in instance_ids]
    task_ids = {row["task_unit_id"] for row in rows}
    if len(task_ids) != 1:
        raise ValueError("auxiliary minibatch must be task-local")
    x = tensor_features_from_smiles(first_smiles(row) for row in rows)
    y = torch.tensor([float(row["median_derived_pic50"]) for row in rows], dtype=torch.float32)
    return x, y, next(iter(task_ids))


def arm_b_auxiliary_update(model: ArmBMultitaskModel, optimizer: torch.optim.Optimizer, x: Tensor, y: Tensor, task_unit_id: str) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = mse(model.predict(x, task_unit_id), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def fresh_target_head() -> ScalarHead:
    return ScalarHead()


def arm_b_target_update(
    encoder: D012Encoder,
    target_head: ScalarHead,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
) -> float:
    encoder.train()
    target_head.train()
    optimizer.zero_grad(set_to_none=True)
    pred = target_head(encoder(x))
    loss = mse(pred, y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def validation_score(encoder: D012Encoder, target_head: ScalarHead, x: Tensor, y: Tensor) -> float:
    encoder.eval()
    target_head.eval()
    with torch.no_grad():
        return float(mse(target_head(encoder(x)), y))


def selected_prediction(embeddings: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return embeddings @ weight.reshape(-1) + bias.reshape(())


def adapt_head_exact(
    head: ScalarHead,
    support_embeddings: Tensor,
    support_targets: Tensor,
    *,
    inner_lr: float = D012_INNER_LR,
) -> tuple[Tensor, Tensor, Tensor]:
    weight = head.linear.weight.squeeze(0)
    bias = head.linear.bias.squeeze(0)
    support_pred = selected_prediction(support_embeddings, weight, bias)
    support_loss = mse(support_pred, support_targets)
    grad_w, grad_b = torch.autograd.grad(support_loss, (weight, bias), create_graph=True, retain_graph=True)
    return weight - inner_lr * grad_w, bias - inner_lr * grad_b, support_loss


def arm_c_exact_outer_update(
    model: ArmCAnilModel,
    optimizer: torch.optim.Optimizer,
    task_unit_id: str,
    support_x: Tensor,
    support_y: Tensor,
    query_x: Tensor,
    query_y: Tensor,
) -> dict[str, object]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    support_embeddings = model.embeddings(support_x)
    adapted_w, adapted_b, support_loss = adapt_head_exact(model.heads[task_unit_id], support_embeddings, support_y)
    query_pred = selected_prediction(model.embeddings(query_x), adapted_w, adapted_b)
    query_loss = mse(query_pred, query_y)
    query_loss.backward()
    encoder_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    return {
        "support_loss": float(support_loss.detach()),
        "query_loss": float(query_loss.detach()),
        "encoder_grad_l1": encoder_grad,
    }


def arm_c_target_adapt_and_score(
    encoder: D012Encoder,
    target_head: ScalarHead,
    support_x: Tensor,
    support_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
) -> dict[str, object]:
    encoder.eval()
    target_head.train()
    before_encoder = clone_parameter_tensors(encoder)
    support_embeddings = encoder(support_x).detach()
    pred = target_head(support_embeddings)
    loss = mse(pred, support_y)
    grad_w, grad_b = torch.autograd.grad(loss, (target_head.linear.weight, target_head.linear.bias))
    adapted_head = copy.deepcopy(target_head)
    with torch.no_grad():
        adapted_head.linear.weight -= D012_INNER_LR * grad_w
        adapted_head.linear.bias -= D012_INNER_LR * grad_b
    score = validation_score(encoder, adapted_head, validation_x, validation_y)
    return {
        "support_loss": float(loss.detach()),
        "validation_score": score,
        "encoder_unchanged": parameter_unchanged(before_encoder, encoder, ""),
        "head_changed": any(
            not torch.equal(a.detach(), b.detach())
            for a, b in zip(target_head.parameters(), adapted_head.parameters())
        ),
    }


def bounded_smoke(manifest_id: str = "rep_01_fold_01", training_seed: int = 20261101, *, target_task: str = TARGET_TASKS[0]) -> dict[str, object]:
    configure_determinism(stable_seed("D012-smoke", manifest_id, training_seed))
    instances = load_d012_instances()
    lookup = rows_by_instance_id(instances)
    task_ids = sorted({row["task_unit_id"] for row in instances})
    budget = scheduler_budget_for_cell(manifest_id, training_seed)

    b_plan = select_b_minibatch(manifest_id, training_seed)
    b_ids = split_semicolon(b_plan["instance_ids"])
    aux_x, aux_y, aux_task = auxiliary_batch_from_instance_ids(b_ids, lookup)
    b_model = ArmBMultitaskModel(task_ids)
    b_optimizer = torch.optim.Adam(b_model.parameters(), lr=D012_INITIAL_LR)
    b_aux_loss = arm_b_auxiliary_update(b_model, b_optimizer, aux_x, aux_y, aux_task)

    manifest = load_manifest(manifest_id)
    target_rows = read_csv(TARGET_DATA)
    train_names = set(manifest["training_names"])
    validation_names = list(manifest["validation_names"])
    scaling = fit_training_only_target_scaling(target_rows, train_names, target_task)
    train_selected, train_x, train_y = target_xy(target_rows, list(manifest["training_names"]), target_task, scaling, limit=16)
    validation_selected, val_x, val_y = target_xy(target_rows, validation_names, target_task, scaling, limit=16)
    if not train_selected or not validation_selected:
        raise RuntimeError("bounded smoke could not find target labels")

    b_target_head = fresh_target_head()
    before_b = clone_parameter_tensors(b_model.encoder)
    b_target_optimizer = torch.optim.Adam([*b_model.encoder.parameters(), *b_target_head.parameters()], lr=D012_INITIAL_LR)
    b_target_loss = arm_b_target_update(b_model.encoder, b_target_head, b_target_optimizer, train_x, train_y)
    b_validation_score = validation_score(b_model.encoder, b_target_head, val_x, val_y)

    c_plan = select_c_episode(manifest_id, training_seed)
    query_ids = split_semicolon(c_plan["query_instance_ids"])
    query_set = set(query_ids)
    c_task_rows = [row for key, row in lookup.items() if row["task_unit_id"] == c_plan["task_unit_id"]]
    support_ids = [f"{row['task_unit_id']}|{row['full_standard_inchikey']}" for row in c_task_rows if f"{row['task_unit_id']}|{row['full_standard_inchikey']}" not in query_set]
    support_x, support_y, c_task = auxiliary_batch_from_instance_ids(support_ids, lookup)
    query_x, query_y, _ = auxiliary_batch_from_instance_ids(query_ids, lookup)
    c_model = ArmCAnilModel(task_ids)
    c_optimizer = torch.optim.Adam(c_model.parameters(), lr=D012_INITIAL_LR)
    c_outer = arm_c_exact_outer_update(c_model, c_optimizer, c_task, support_x, support_y, query_x, query_y)
    c_target_head = fresh_target_head()
    c_target = arm_c_target_adapt_and_score(c_model.encoder, c_target_head, train_x, train_y, val_x, val_y)
    return {
        "manifest_id": manifest_id,
        "training_seed": training_seed,
        "scheduler_budget": budget,
        "arm_b": {
            "auxiliary_loss": b_aux_loss,
            "target_loss": b_target_loss,
            "validation_score": b_validation_score,
            "target_update_encoder_changed": parameter_changed(before_b, b_model.encoder, ""),
            "validation_used_for_checkpoint_selection": True,
        },
        "arm_c": {
            **c_outer,
            "target_adaptation": c_target,
            "validation_used_for_checkpoint_selection": False,
        },
        "target_scaling": {
            "task": target_task,
            "mean": scaling.mean,
            "scale": scaling.scale,
            "fit_on_training_names_only": True,
            "train_examples_used": len(train_selected),
            "validation_examples_scored": len(validation_selected),
        },
        "qualified_plan_consumed": {
            "b_plan_task": b_plan["task_unit_id"],
            "c_plan_task": c_plan["task_unit_id"],
            "b_updates_per_epoch": int(budget["b_updates_per_epoch"]),
            "c_updates_per_epoch": int(budget["c_updates_per_epoch"]),
        },
    }


if __name__ == "__main__":
    print(json.dumps(bounded_smoke(), indent=2, sort_keys=True))
