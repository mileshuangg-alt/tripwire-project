from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from chemprop import schedulers
import numpy as np
import torch
from torch import Tensor

from cyp002_config import CYP002_TASK_NAMES
from cyp002_validation import compute_validation_macro_st_rae
from cyp007_infrastructure import sha256_file
from p1_pairwise import (
    PairwisePartitionContract,
    compute_pairwise_partition_contract,
    sample_supervised_pairs,
)
from p1_pairwise_chemprop import count_parameters
from p1_pairwise_model import (
    P1AntisymmetricPairHead,
    normalized_weighted_pair_mse,
    reconstruct_absolute_from_references,
)


P1_ACCEPTED_PAIR_BATCH_SIZE = 16
P1_UPDATES_PER_PSEUDO_EPOCH = 235
P1_MAX_PSEUDO_EPOCHS = 50
P1_MAX_UPDATES = P1_UPDATES_PER_PSEUDO_EPOCH * P1_MAX_PSEUDO_EPOCHS
P1_WARMUP_UPDATES = 2350
P1_VALIDATION_INTERVAL_UPDATES = 235
P1_EARLY_STOPPING_PATIENCE = 20
P1_INITIAL_LR = 1.4283739994469765e-05
P1_MAX_LR = 0.0008245400010426866
P1_FINAL_LR = 0.00020141638331344468


@dataclass(frozen=True)
class P1TrainingSchedule:
    accepted_pair_batch_size: int = P1_ACCEPTED_PAIR_BATCH_SIZE
    updates_per_pseudo_epoch: int = P1_UPDATES_PER_PSEUDO_EPOCH
    max_pseudo_epochs: int = P1_MAX_PSEUDO_EPOCHS
    max_updates: int = P1_MAX_UPDATES
    warmup_updates: int = P1_WARMUP_UPDATES
    validation_interval_updates: int = P1_VALIDATION_INTERVAL_UPDATES
    early_stopping_patience: int = P1_EARLY_STOPPING_PATIENCE
    initial_lr: float = P1_INITIAL_LR
    max_lr: float = P1_MAX_LR
    final_lr: float = P1_FINAL_LR
    optimizer: str = "Adam"
    lr_schedule: str = "Chemprop build_NoamLike_LRSched"


@dataclass(frozen=True)
class P1TrainingAccounting:
    update_count: int
    validation_checks: int
    pair_attempts: int
    accepted_pair_rows: int
    no_supervision_rejections: int
    eligible_target_count: tuple[int, int, int, int]
    weighted_target_contribution: tuple[float, float, float, float]
    best_validation_macro_st_rae: float | None
    best_update: int | None
    bad_validation_checks_since_best: int
    stopped_early: bool


@dataclass(frozen=True)
class P1TrainingProvenance:
    dataset_path: str
    dataset_sha256: str
    split_manifest_path: str
    split_manifest_sha256: str
    t121_config_path: str
    t121_config_sha256: str
    training_seed: int
    endpoint_order: tuple[str, ...]
    scaler_means: tuple[float, float, float, float]
    scaler_scales: tuple[float, float, float, float]
    n_t: tuple[int, int, int, int]
    q_t: tuple[float, float, float, float]
    w_t: tuple[float | None, float | None, float | None, float | None]
    schedule: P1TrainingSchedule
    parameter_counts: Mapping[str, Mapping[str, int]]


class P1ValidationController:
    def __init__(self, *, patience: int = P1_EARLY_STOPPING_PATIENCE) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive.")
        self.patience = int(patience)
        self.best_score: float | None = None
        self.best_update: int | None = None
        self.best_state_dict: dict[str, Tensor] | None = None
        self.bad_checks = 0
        self.history: list[dict[str, object]] = []
        self.stopped = False

    def observe(
        self,
        *,
        update: int,
        validation_result: Mapping[str, object],
        state_dict: Mapping[str, Tensor] | None = None,
    ) -> bool:
        score = float(validation_result["macro_st_rae"])
        if not np.isfinite(score):
            raise ValueError("validation macro ST-RAE must be finite.")

        improved = self.best_score is None or score < self.best_score
        if improved:
            self.best_score = score
            self.best_update = int(update)
            self.bad_checks = 0
            if state_dict is not None:
                self.best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in state_dict.items()
                }
        else:
            self.bad_checks += 1

        record = dict(validation_result)
        record["update"] = int(update)
        record["improved"] = bool(improved)
        record["bad_checks_since_best"] = self.bad_checks
        self.history.append(record)

        if self.bad_checks >= self.patience:
            self.stopped = True

        return improved


def build_p1_optimizer_and_scheduler(parameters) -> tuple[torch.optim.Optimizer, object]:
    optimizer = torch.optim.Adam(
        parameters,
        lr=P1_INITIAL_LR,
    )
    scheduler = schedulers.build_NoamLike_LRSched(
        optimizer,
        warmup_steps=P1_WARMUP_UPDATES,
        cooldown_steps=P1_MAX_UPDATES - P1_WARMUP_UPDATES,
        init_lr=P1_INITIAL_LR,
        max_lr=P1_MAX_LR,
        final_lr=P1_FINAL_LR,
    )
    return optimizer, scheduler


def should_validate_after_update(update_count: int) -> bool:
    if update_count <= 0:
        return False
    return update_count % P1_VALIDATION_INTERVAL_UPDATES == 0


def pseudo_epoch_for_update(update_count: int) -> int:
    if update_count <= 0:
        raise ValueError("update_count must be positive.")
    return (update_count - 1) // P1_UPDATES_PER_PSEUDO_EPOCH + 1


def endpoint_weights_tensor(
    contract: PairwisePartitionContract,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    weights = []
    for endpoint in contract.endpoints:
        weights.append(1.0 if endpoint.weight is None else float(endpoint.weight))
    result = torch.tensor(weights, dtype=dtype, device=device)
    if not torch.isfinite(result).all() or (result <= 0).any():
        raise ValueError("endpoint weights must be finite and positive.")
    return result


def draw_p1_training_pair_batch(
    *,
    contract: PairwisePartitionContract,
    molecule_ids: Sequence[str],
    labels: Sequence[Sequence[object]],
    scaler_means: Sequence[float | None],
    scaler_scales: Sequence[float | None],
    rng: np.random.Generator,
    accepted_pair_count: int = P1_ACCEPTED_PAIR_BATCH_SIZE,
):
    return sample_supervised_pairs(
        contract,
        molecule_ids,
        labels,
        endpoint_order=CYP002_TASK_NAMES,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
        accepted_pair_count=accepted_pair_count,
        rng=rng,
    )


def tensors_from_pair_batch(
    batch,
    embedding_table: Mapping[str, Tensor],
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    h_a = torch.stack([embedding_table[name] for name in batch.molecule_a_ids])
    h_b = torch.stack([embedding_table[name] for name in batch.molecule_b_ids])
    if device is not None:
        h_a = h_a.to(device)
        h_b = h_b.to(device)

    mask = torch.tensor(batch.eligibility_masks, dtype=torch.bool, device=h_a.device)
    targets = torch.tensor(batch.delta_targets, dtype=h_a.dtype, device=h_a.device)

    if not mask.any():
        raise ValueError("pair batch contains no eligible targets.")
    if not torch.isfinite(targets[mask]).all():
        raise ValueError("eligible pair targets must be finite.")
    return h_a, h_b, targets, mask


def p1_forward_loss_backward_step(
    *,
    pair_head: P1AntisymmetricPairHead,
    optimizer: torch.optim.Optimizer,
    scheduler,
    h_a: Tensor,
    h_b: Tensor,
    targets: Tensor,
    mask: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor]:
    pair_head.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = pair_head(h_a, h_b)
    loss = normalized_weighted_pair_mse(
        predictions,
        targets,
        mask,
        weights,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("P1 training loss is non-finite.")
    loss.backward()

    for parameter in pair_head.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError("trainable parameter is missing a gradient.")
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("P1 parameter gradient is non-finite.")

    optimizer.step()
    scheduler.step()

    for parameter in pair_head.parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError("P1 parameter became non-finite after optimizer step.")
    return loss.detach(), predictions.detach()


def reconstruct_and_score_p1_validation(
    *,
    predicted_delta_z: Tensor,
    reference_y: Tensor,
    reference_mask: Tensor,
    scaler_scales: Tensor,
    validation_y_true: Mapping[str, np.ndarray],
    validation_conf_low: Mapping[str, np.ndarray],
    validation_conf_high: Mapping[str, np.ndarray],
    task_names: Sequence[str] = CYP002_TASK_NAMES,
) -> dict[str, object]:
    absolute_mean, reference_dispersion = reconstruct_absolute_from_references(
        predicted_delta_z,
        reference_y,
        reference_mask,
        scaler_scales,
    )
    if not torch.isfinite(absolute_mean).all():
        raise FloatingPointError("validation reconstruction is non-finite.")
    if not torch.isfinite(reference_dispersion).all():
        raise FloatingPointError("validation reference dispersion is non-finite.")

    predictions_np = absolute_mean.detach().cpu().numpy()
    y_pred = {
        task_name: predictions_np[:, task_index]
        for task_index, task_name in enumerate(task_names)
    }
    result = compute_validation_macro_st_rae(
        y_true=validation_y_true,
        y_pred=y_pred,
        conf_low=validation_conf_low,
        conf_high=validation_conf_high,
        task_names=task_names,
    )
    result["reference_dispersion_diagnostic_shape"] = tuple(reference_dispersion.shape)
    return result


def build_pairwise_contract_from_training_labels(
    *,
    molecule_ids: Sequence[str],
    labels: Sequence[Sequence[object]],
    scaler_means: Sequence[float | None],
    scaler_scales: Sequence[float | None],
) -> PairwisePartitionContract:
    return compute_pairwise_partition_contract(
        molecule_ids,
        labels,
        endpoint_order=CYP002_TASK_NAMES,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
    )


def provenance_from_contract(
    *,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    t121_config_path: str | Path,
    training_seed: int,
    scaler_means: Sequence[float],
    scaler_scales: Sequence[float],
    contract: PairwisePartitionContract,
    modules: Mapping[str, torch.nn.Module],
) -> P1TrainingProvenance:
    return P1TrainingProvenance(
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
        split_manifest_path=str(split_manifest_path),
        split_manifest_sha256=sha256_file(split_manifest_path),
        t121_config_path=str(t121_config_path),
        t121_config_sha256=sha256_file(t121_config_path),
        training_seed=int(training_seed),
        endpoint_order=tuple(CYP002_TASK_NAMES),
        scaler_means=tuple(float(value) for value in scaler_means),
        scaler_scales=tuple(float(value) for value in scaler_scales),
        n_t=tuple(endpoint.n_labeled for endpoint in contract.endpoints),
        q_t=tuple(float(endpoint.q_pair_eligible) for endpoint in contract.endpoints),
        w_t=tuple(
            None if endpoint.weight is None else float(endpoint.weight)
            for endpoint in contract.endpoints
        ),
        schedule=P1TrainingSchedule(),
        parameter_counts={
            name: asdict(count_parameters(module))
            for name, module in modules.items()
        },
    )


def accounting_after_batch(
    *,
    update_count: int,
    validation_checks: int,
    batch,
    contract: PairwisePartitionContract,
    controller: P1ValidationController,
) -> P1TrainingAccounting:
    weights = [1.0 if endpoint.weight is None else float(endpoint.weight) for endpoint in contract.endpoints]
    weighted = tuple(
        float(count * weight)
        for count, weight in zip(batch.eligible_target_count, weights, strict=True)
    )
    return P1TrainingAccounting(
        update_count=int(update_count),
        validation_checks=int(validation_checks),
        pair_attempts=int(batch.attempts),
        accepted_pair_rows=int(batch.accepted_count),
        no_supervision_rejections=int(batch.no_supervision_rejections),
        eligible_target_count=tuple(int(value) for value in batch.eligible_target_count),
        weighted_target_contribution=weighted,
        best_validation_macro_st_rae=controller.best_score,
        best_update=controller.best_update,
        bad_validation_checks_since_best=controller.bad_checks,
        stopped_early=controller.stopped,
    )
