from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from chemprop import schedulers
import numpy as np
import torch
from torch import Tensor

from cyp002_config import CYP002_TASK_NAMES
from p1_pairwise_training import P1ValidationController
from p2_anil_model import (
    P2MetaHead,
    P2_INNER_LR,
    P2_N_TASKS,
    query_loss_after_exact_adaptation,
    selected_task_prediction,
    standardized_mse,
    validate_inner_adaptation_settings,
)


P2_SUPPORT_COUNT = 13
P2_QUERY_COUNT = 3
P2_EPISODES_PER_OUTER_UPDATE = 4
P2_OUTER_UPDATES_PER_PSEUDO_EPOCH = 235
P2_TASK_EPISODES_PER_PSEUDO_EPOCH = 940
P2_VALIDATION_INTERVAL_UPDATES = 235
P2_WARMUP_UPDATES = 2350
P2_MAX_UPDATES = 11750
P2_MAX_VALIDATION_OPPORTUNITIES = 50
P2_EARLY_STOPPING_PATIENCE = 20
P2_INITIAL_LR = 1.4283739994469765e-05
P2_MAX_LR = 0.0008245400010426866
P2_FINAL_LR = 0.00020141638331344468
T121_O150 = 0.701212576796


@dataclass(frozen=True)
class P2Episode:
    task_index: int
    task_name: str
    support_molecule_ids: tuple[str, ...]
    query_molecule_ids: tuple[str, ...]
    support_targets_z: tuple[float, ...]
    query_targets_z: tuple[float, ...]


@dataclass(frozen=True)
class P2OuterUpdateResult:
    outer_loss: Tensor
    query_losses: tuple[Tensor, Tensor, Tensor, Tensor]
    support_losses: tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True, order=True)
class D011Candidate:
    inner_lr: float
    inner_steps: int

    def __post_init__(self) -> None:
        validate_inner_adaptation_settings(self.inner_lr, self.inner_steps)


def build_p2_optimizer_and_scheduler(parameters) -> tuple[torch.optim.Optimizer, object]:
    optimizer = torch.optim.Adam(parameters, lr=P2_INITIAL_LR)
    scheduler = schedulers.build_NoamLike_LRSched(
        optimizer,
        warmup_steps=P2_WARMUP_UPDATES,
        cooldown_steps=P2_MAX_UPDATES - P2_WARMUP_UPDATES,
        init_lr=P2_INITIAL_LR,
        max_lr=P2_MAX_LR,
        final_lr=P2_FINAL_LR,
    )
    return optimizer, scheduler


def should_validate_after_update(update_count: int) -> bool:
    return update_count > 0 and update_count % P2_VALIDATION_INTERVAL_UPDATES == 0


def enumerate_p2_cell_keys():
    from p1_pairwise_production import enumerate_p1_cells

    return enumerate_p1_cells()


def fit_training_only_scaler(labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if labels.shape != mask.shape or labels.ndim != 2 or labels.shape[1] != P2_N_TASKS:
        raise ValueError("P2 labels/mask must have shape (n, 4).")
    means = np.empty(P2_N_TASKS, dtype=np.float64)
    scales = np.empty(P2_N_TASKS, dtype=np.float64)
    for task_index in range(P2_N_TASKS):
        observed = labels[mask[:, task_index], task_index]
        if observed.size < P2_SUPPORT_COUNT + P2_QUERY_COUNT:
            raise ValueError("P2 requires at least 16 observed training labels per task.")
        if not np.isfinite(observed).all():
            raise ValueError("P2 observed training labels must be finite.")
        means[task_index] = float(observed.mean())
        scale = float(observed.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("P2 training-only scaler scale must be finite and positive.")
        scales[task_index] = scale
    return means, scales


def draw_p2_episodes(
    *,
    molecule_ids: Sequence[str],
    labels: np.ndarray,
    mask: np.ndarray,
    scaler_means: Sequence[float],
    scaler_scales: Sequence[float],
    rng: np.random.Generator,
) -> tuple[P2Episode, P2Episode, P2Episode, P2Episode]:
    ids = tuple(str(value) for value in molecule_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("P2 training molecule identities must be unique.")
    labels = np.asarray(labels, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    means = np.asarray(scaler_means, dtype=np.float64)
    scales = np.asarray(scaler_scales, dtype=np.float64)
    if labels.shape != mask.shape or labels.shape != (len(ids), P2_N_TASKS):
        raise ValueError("P2 labels and mask must align with molecule identities and four tasks.")
    if means.shape != (P2_N_TASKS,) or scales.shape != (P2_N_TASKS,):
        raise ValueError("P2 scaler values must align with the four tasks.")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("P2 scaler values must be finite and positive where required.")
    episodes = []
    for task_index, task_name in enumerate(CYP002_TASK_NAMES):
        eligible = np.flatnonzero(mask[:, task_index])
        if eligible.size < P2_SUPPORT_COUNT + P2_QUERY_COUNT:
            raise ValueError(f"P2 task {task_name} has fewer than 16 eligible training molecules.")
        selected = rng.choice(eligible, size=P2_SUPPORT_COUNT + P2_QUERY_COUNT, replace=False)
        support = selected[:P2_SUPPORT_COUNT]
        query = selected[P2_SUPPORT_COUNT:]
        if set(support) & set(query):
            raise RuntimeError("P2 support/query molecule boundary failed.")
        support_y = labels[support, task_index]
        query_y = labels[query, task_index]
        if not np.isfinite(support_y).all() or not np.isfinite(query_y).all():
            raise ValueError("P2 selected task labels must be observed and finite; no imputation is allowed.")
        episodes.append(
            P2Episode(
                task_index=task_index,
                task_name=task_name,
                support_molecule_ids=tuple(ids[index] for index in support),
                query_molecule_ids=tuple(ids[index] for index in query),
                support_targets_z=tuple(((support_y - means[task_index]) / scales[task_index]).tolist()),
                query_targets_z=tuple(((query_y - means[task_index]) / scales[task_index]).tolist()),
            )
        )
    return tuple(episodes)  # type: ignore[return-value]


def p2_outer_loss_from_episode_embeddings(
    *,
    meta_head: P2MetaHead,
    episodes: Sequence[P2Episode],
    embedding_table: Mapping[str, Tensor],
    inner_lr: float = P2_INNER_LR,
    inner_steps: int = 1,
) -> P2OuterUpdateResult:
    validate_inner_adaptation_settings(inner_lr, inner_steps)
    if tuple(episode.task_name for episode in episodes) != tuple(CYP002_TASK_NAMES):
        raise ValueError("P2 episodes must appear in the frozen four-task order.")
    meta_head.train()
    query_losses = []
    support_losses = []
    for episode in episodes:
        if len(episode.support_molecule_ids) != P2_SUPPORT_COUNT:
            raise ValueError("P2 episode must contain exactly 13 support molecules.")
        if len(episode.query_molecule_ids) != P2_QUERY_COUNT:
            raise ValueError("P2 episode must contain exactly 3 query molecules.")
        if set(episode.support_molecule_ids) & set(episode.query_molecule_ids):
            raise ValueError("P2 support/query molecules must be disjoint.")
        support_embeddings = torch.stack([embedding_table[name] for name in episode.support_molecule_ids])
        query_embeddings = torch.stack([embedding_table[name] for name in episode.query_molecule_ids])
        support_targets = torch.tensor(episode.support_targets_z, dtype=support_embeddings.dtype, device=support_embeddings.device)
        query_targets = torch.tensor(episode.query_targets_z, dtype=query_embeddings.dtype, device=query_embeddings.device)
        query_loss, support_loss = query_loss_after_exact_adaptation(
            meta_head,
            support_embeddings,
            support_targets,
            query_embeddings,
            query_targets,
            task_index=episode.task_index,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
        )
        query_losses.append(query_loss)
        support_losses.append(support_loss)
    outer_loss = sum(query_losses) / P2_EPISODES_PER_OUTER_UPDATE
    if not torch.isfinite(outer_loss):
        raise FloatingPointError("P2 outer loss is non-finite.")
    return P2OuterUpdateResult(
        outer_loss=outer_loss,
        query_losses=tuple(query_losses),  # type: ignore[arg-type]
        support_losses=tuple(support_losses),  # type: ignore[arg-type]
    )


def p2_exact_outer_update_step(
    *,
    meta_head: P2MetaHead,
    optimizer: torch.optim.Optimizer,
    scheduler,
    episodes: Sequence[P2Episode],
    embedding_table: Mapping[str, Tensor],
    inner_lr: float = P2_INNER_LR,
    inner_steps: int = 1,
) -> P2OuterUpdateResult:
    optimizer.zero_grad(set_to_none=True)
    result = p2_outer_loss_from_episode_embeddings(
        meta_head=meta_head,
        episodes=episodes,
        embedding_table=embedding_table,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
    )
    result.outer_loss.backward()
    for parameter in meta_head.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError("P2 trainable parameter is missing exact meta-gradient.")
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("P2 exact meta-gradient is non-finite.")
    optimizer.step()
    scheduler.step()
    return result


def full_support_task_adaptation_prediction(
    *,
    meta_head: P2MetaHead,
    train_embeddings: Tensor,
    train_labels: Tensor,
    train_mask: Tensor,
    validation_embeddings: Tensor,
    scaler_means: Tensor,
    scaler_scales: Tensor,
    task_index: int,
    support_batch_size: int | None = None,
    inner_lr: float = P2_INNER_LR,
    inner_steps: int = 1,
) -> Tensor:
    validate_inner_adaptation_settings(inner_lr, inner_steps)
    if task_index < 0 or task_index >= P2_N_TASKS:
        raise ValueError("task_index is outside the frozen four-task range.")
    meta_head.eval()
    eligible = train_mask[:, task_index]
    if not eligible.any():
        raise ValueError("P2 evaluation adaptation requires eligible training support.")
    support_embeddings = train_embeddings[eligible]
    support_y = train_labels[eligible, task_index]
    if not torch.isfinite(support_y).all():
        raise ValueError("P2 evaluation support labels must be finite.")
    support_z = (support_y - scaler_means[task_index]) / scaler_scales[task_index]
    weight = meta_head.output.weight[task_index]
    bias = meta_head.output.bias[task_index]
    if support_batch_size is None:
        support_batch_size = support_embeddings.shape[0]
    if support_batch_size <= 0:
        raise ValueError("support_batch_size must be positive.")
    adapted_w = weight
    adapted_b = bias
    for _step in range(int(inner_steps)):
        squared_sum = None
        count = 0
        for start in range(0, support_embeddings.shape[0], support_batch_size):
            stop = min(start + support_batch_size, support_embeddings.shape[0])
            features = meta_head.features(support_embeddings[start:stop])
            pred = selected_task_prediction(features, task_index=task_index, weight=adapted_w, bias=adapted_b)
            errors = (pred - support_z[start:stop]).square().sum()
            squared_sum = errors if squared_sum is None else squared_sum + errors
            count += stop - start
        support_loss = squared_sum / count
        grad_w, grad_b = torch.autograd.grad(support_loss, (adapted_w, adapted_b), create_graph=False)
        adapted_w = adapted_w - float(inner_lr) * grad_w
        adapted_b = adapted_b - float(inner_lr) * grad_b
    val_features = meta_head.features(validation_embeddings)
    pred_z = selected_task_prediction(val_features, task_index=task_index, weight=adapted_w, bias=adapted_b)
    return pred_z * scaler_scales[task_index] + scaler_means[task_index]


def p2_strict_o150_advancement(cell_macro_scores: Sequence[float]) -> dict[str, object]:
    if len(cell_macro_scores) != 150:
        raise ValueError("P2 O150 requires exactly 150 cell macro ST-RAE values.")
    values = np.asarray(cell_macro_scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("P2 O150 cell values must be finite.")
    o150 = float(values.mean())
    return {
        "cell_count": 150,
        "o150": o150,
        "t121_o150": T121_O150,
        "p2_advances": bool(o150 < T121_O150),
        "strict_rule": "P2 advances iff O150(P2) < 0.701212576796",
    }


P2ValidationController = P1ValidationController
