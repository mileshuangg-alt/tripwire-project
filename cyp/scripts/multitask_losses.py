from __future__ import annotations

import torch
from torch import Tensor

from chemprop.nn import ChempropMetric, LossFunctionRegistry

N_TASKS = 4


def task_balanced_masked_mse(
    preds: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute the D001 task-balanced masked MSE objective.

    Parameters
    ----------
    preds:
        Prediction tensor of shape (batch, tasks).
    targets:
        Target tensor of shape (batch, tasks).
    mask:
        Boolean observation mask of shape (batch, tasks).

    Returns
    -------
    Tensor
        Scalar task-balanced loss:
        1/4 * sum_t mean(observed squared errors for task t).

    Notes
    -----
    A task with zero observed labels in the batch contributes zero to the
    fixed four-task average. The denominator remains exactly 4; it is never
    renormalized over tasks present in the batch.
    """
    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D, got shape {tuple(preds.shape)}")

    if targets.shape != preds.shape:
        raise ValueError(
            "targets must have the same shape as preds: "
            f"preds={tuple(preds.shape)}, targets={tuple(targets.shape)}"
        )

    if mask.shape != preds.shape:
        raise ValueError(
            "mask must have the same shape as preds: "
            f"preds={tuple(preds.shape)}, mask={tuple(mask.shape)}"
        )

    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be boolean, got {mask.dtype}")

    squared_error = (preds - targets).square()

    task_losses = []

    for task_index in range(preds.shape[1]):
        present = mask[:, task_index]

        if present.any():
            task_loss = squared_error[present, task_index].mean()
        else:
            task_loss = squared_error.new_zeros(())

        task_losses.append(task_loss)

    task_losses_tensor = torch.stack(task_losses)

    result = task_losses_tensor.mean()
    return result


@LossFunctionRegistry.register("task-balanced-mse")
class TaskBalancedMSE(ChempropMetric):
    """D001 task-balanced masked MSE criterion."""

    def _calc_unreduced_loss(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        weights: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ) -> Tensor:
        result = (preds - targets).square()
        return result

    def task_losses(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        if preds.ndim != 2:
            raise ValueError(
                f"preds must be 2D, got shape {tuple(preds.shape)}"
            )

        if targets.shape != preds.shape:
            raise ValueError(
                "targets must have the same shape as preds: "
                f"preds={tuple(preds.shape)}, "
                f"targets={tuple(targets.shape)}"
            )

        if mask.shape != preds.shape:
            raise ValueError(
                "mask must have the same shape as preds: "
                f"preds={tuple(preds.shape)}, "
                f"mask={tuple(mask.shape)}"
            )

        if mask.dtype != torch.bool:
            raise TypeError(
                f"mask must be boolean, got {mask.dtype}"
            )

        if weights is None:
            weights = torch.ones(
                preds.shape[0],
                dtype=preds.dtype,
                device=preds.device,
            )
        else:
            weights = weights.to(
                dtype=preds.dtype,
                device=preds.device,
            ).reshape(-1)

        squared_error = self._calc_unreduced_loss(
            preds,
            targets,
            mask,
            weights,
            torch.zeros_like(mask),
            torch.zeros_like(mask),
        )

        losses = []

        for task_index in range(preds.shape[1]):
            present = mask[:, task_index]

            if present.any():
                task_loss = (
                    squared_error[present, task_index]
                    * weights[present]
                ).mean()
            else:
                task_loss = squared_error[:, task_index].sum() * 0.0

            losses.append(task_loss)

        result = torch.stack(losses)

        if result.numel() != N_TASKS:
            raise ValueError(
                f"D001 requires exactly {N_TASKS} CYP tasks, "
                f"got {result.numel()}"
            )

        return result

    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        lt_mask: Tensor | None = None,
        gt_mask: Tensor | None = None,
    ) -> None:
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)

        task_losses = self.task_losses(
            preds,
            targets,
            mask,
            weights,
        )

        if task_losses.numel() != N_TASKS:
            raise ValueError(
                f"D001 requires exactly {N_TASKS} CYP tasks, "
                f"got {task_losses.numel()}"
            )

        # D001: fixed equal weighting of all four tasks.
        result = task_losses.sum() / float(N_TASKS)

        self.total_loss += result
        self.num_samples += 1

    def compute(self) -> Tensor:
        result = self.total_loss / self.num_samples
        return result
