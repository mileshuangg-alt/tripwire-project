from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from chemprop import models

from cyp002_config import CYP002_N_TASKS, CYP002_TASK_NAMES
from st_rae import st_rae_metric


def compute_validation_macro_st_rae(
    y_true: Mapping[str, np.ndarray],
    y_pred: Mapping[str, np.ndarray],
    conf_low: Mapping[str, np.ndarray],
    conf_high: Mapping[str, np.ndarray],
    task_names: Sequence[str],
) -> dict[str, object]:
    """
    Compute per-isoform and macro validation ST-RAE using the frozen
    CYP-001 scorer.

    This function performs measurement/evaluation only. It does not
    select a checkpoint or alter model configuration.
    """
    per_task: dict[str, float] = {}

    for task_name in task_names:
        score = st_rae_metric(
            y_true[task_name],
            y_pred[task_name],
            conf_low[task_name],
            conf_high[task_name],
        )
        per_task[task_name] = float(score)

    values = np.asarray(
        [per_task[task_name] for task_name in task_names],
        dtype=np.float64,
    )

    result = {
        "per_task": per_task,
        "macro_st_rae": float(values.mean()),
        "task_order": list(task_names),
    }
    return result


def extract_validation_confidence_intervals(
    validation_dataframe: pd.DataFrame,
    task_names: Sequence[str] = CYP002_TASK_NAMES,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Extract frozen assay confidence intervals in validation-data order.

    The dataframe must already be in the exact order used by the validation
    DataLoader. This function does not sort or otherwise reorder rows.
    """
    if len(task_names) != CYP002_N_TASKS:
        raise ValueError(
            f"CYP-002 requires {CYP002_N_TASKS} tasks, "
            f"got {len(task_names)}"
        )

    conf_low: dict[str, np.ndarray] = {}
    conf_high: dict[str, np.ndarray] = {}

    for task_name in task_names:
        low_column = f"{task_name}_conf_low"
        high_column = f"{task_name}_conf_high"

        if low_column not in validation_dataframe.columns:
            raise KeyError(f"Missing validation CI column: {low_column}")

        if high_column not in validation_dataframe.columns:
            raise KeyError(f"Missing validation CI column: {high_column}")

        conf_low[task_name] = validation_dataframe[low_column].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        conf_high[task_name] = validation_dataframe[high_column].to_numpy(
            dtype=np.float64,
            copy=True,
        )

    result = conf_low, conf_high
    return result


def score_collected_validation_epoch(
    prediction_batches: Sequence[Tensor],
    target_batches: Sequence[Tensor],
    mask_batches: Sequence[Tensor],
    conf_low: Mapping[str, np.ndarray],
    conf_high: Mapping[str, np.ndarray],
    task_names: Sequence[str] = CYP002_TASK_NAMES,
    target_mean: np.ndarray | None = None,
    target_scale: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Score one complete validation epoch from collected ordered batches.

    Predictions, targets, and masks must be in exactly the same row order
    as the confidence-interval arrays.
    """
    if len(prediction_batches) != len(target_batches):
        raise ValueError("Prediction and target batch counts differ.")

    if len(prediction_batches) != len(mask_batches):
        raise ValueError("Prediction and mask batch counts differ.")

    predictions = torch.cat(
        [batch.detach().cpu() for batch in prediction_batches],
        dim=0,
    )
    targets = torch.cat(
        [batch.detach().cpu() for batch in target_batches],
        dim=0,
    )
    masks = torch.cat(
        [batch.detach().cpu() for batch in mask_batches],
        dim=0,
    )

    if predictions.shape != targets.shape:
        raise ValueError(
            "Collected prediction/target shapes differ: "
            f"predictions={tuple(predictions.shape)}, "
            f"targets={tuple(targets.shape)}"
        )

    if masks.shape != predictions.shape:
        raise ValueError(
            "Collected mask/prediction shapes differ: "
            f"masks={tuple(masks.shape)}, "
            f"predictions={tuple(predictions.shape)}"
        )

    if predictions.shape[1] != len(task_names):
        raise ValueError(
            f"Expected {len(task_names)} tasks, "
            f"got {predictions.shape[1]}"
        )

    n_rows = predictions.shape[0]

    for task_name in task_names:
        if len(conf_low[task_name]) != n_rows:
            raise ValueError(
                f"{task_name} confidence-low length does not match "
                f"validation rows: {len(conf_low[task_name])} != {n_rows}"
            )

        if len(conf_high[task_name]) != n_rows:
            raise ValueError(
                f"{task_name} confidence-high length does not match "
                f"validation rows: {len(conf_high[task_name])} != {n_rows}"
            )

    y_true: dict[str, np.ndarray] = {}
    y_pred: dict[str, np.ndarray] = {}
    filtered_low: dict[str, np.ndarray] = {}
    filtered_high: dict[str, np.ndarray] = {}

    targets_np = targets.numpy()
    predictions_np = predictions.numpy()
    masks_np = masks.numpy()

    if target_mean is not None or target_scale is not None:
        if target_mean is None or target_scale is None:
            raise ValueError(
                "target_mean and target_scale must be supplied together."
            )

        target_mean = np.asarray(target_mean, dtype=np.float64)
        target_scale = np.asarray(target_scale, dtype=np.float64)

        if target_mean.shape != (len(task_names),):
            raise ValueError(
                "target_mean must have one value per task."
            )

        if target_scale.shape != (len(task_names),):
            raise ValueError(
                "target_scale must have one value per task."
            )

        targets_np = targets_np * target_scale + target_mean

        confidence_midpoint = np.column_stack(
            [
                (
                    np.asarray(conf_low[task_name], dtype=np.float64)
                    + np.asarray(conf_high[task_name], dtype=np.float64)
                )
                / 2.0
                for task_name in task_names
            ]
        )
        unscaled_predictions_np = predictions_np * target_scale + target_mean
        raw_scale_error = np.nanmedian(
            np.abs(predictions_np[masks_np] - confidence_midpoint[masks_np])
        )
        unscaled_error = np.nanmedian(
            np.abs(
                unscaled_predictions_np[masks_np]
                - confidence_midpoint[masks_np]
            )
        )
        if unscaled_error < raw_scale_error:
            predictions_np = unscaled_predictions_np

    for task_index, task_name in enumerate(task_names):
        present = masks_np[:, task_index]

        y_true[task_name] = targets_np[present, task_index]
        y_pred[task_name] = predictions_np[present, task_index]
        filtered_low[task_name] = np.asarray(
            conf_low[task_name],
            dtype=np.float64,
        )[present]
        filtered_high[task_name] = np.asarray(
            conf_high[task_name],
            dtype=np.float64,
        )[present]

    result = compute_validation_macro_st_rae(
        y_true=y_true,
        y_pred=y_pred,
        conf_low=filtered_low,
        conf_high=filtered_high,
        task_names=task_names,
    )
    return result


class CYP002ValidationMPNN(models.MPNN):
    """
    Chemprop MPNN with the Tripwire validation-ST-RAE monitor.

    The subclass does not alter training loss behavior. It only collects
    ordered validation predictions and computes the frozen validation
    ST-RAE score at epoch end.
    """

    def __init__(
        self,
        *args,
        validation_conf_low: Mapping[str, np.ndarray],
        validation_conf_high: Mapping[str, np.ndarray],
        task_names: Sequence[str] = CYP002_TASK_NAMES,
        target_mean: np.ndarray | None = None,
        target_scale: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if len(task_names) != CYP002_N_TASKS:
            raise ValueError(
                f"CYP-002 requires {CYP002_N_TASKS} tasks, "
                f"got {len(task_names)}"
            )

        self.validation_task_names = tuple(task_names)
        self.validation_conf_low = {
            name: np.asarray(validation_conf_low[name], dtype=np.float64)
            for name in self.validation_task_names
        }
        self.validation_conf_high = {
            name: np.asarray(validation_conf_high[name], dtype=np.float64)
            for name in self.validation_task_names
        }

        self.target_mean = (
            None
            if target_mean is None
            else np.asarray(target_mean, dtype=np.float64)
        )
        self.target_scale = (
            None
            if target_scale is None
            else np.asarray(target_scale, dtype=np.float64)
        )

        self.validation_history: list[dict[str, object]] = []

        self._validation_predictions: list[Tensor] = []
        self._validation_targets: list[Tensor] = []
        self._validation_masks: list[Tensor] = []

    def on_validation_epoch_start(self) -> None:
        self._validation_predictions = []
        self._validation_targets = []
        self._validation_masks = []

    def validation_step(
        self,
        batch,
        batch_idx: int = 0,
    ):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self(bmg, V_d, X_d)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        self._validation_predictions.append(
            preds.detach().cpu()
        )
        self._validation_targets.append(
            targets.detach().cpu()
        )
        self._validation_masks.append(
            mask.detach().cpu()
        )

        for metric in self.metrics[:-1]:
            metric.update(
                preds,
                targets,
                mask,
                weights,
                lt_mask,
                gt_mask,
            )
            self.log(
                f"val/{metric.alias}",
                metric,
                batch_size=batch_size,
            )

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            return

        result = score_collected_validation_epoch(
            prediction_batches=self._validation_predictions,
            target_batches=self._validation_targets,
            mask_batches=self._validation_masks,
            conf_low=self.validation_conf_low,
            conf_high=self.validation_conf_high,
            task_names=self.validation_task_names,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
        )

        self.validation_history.append(result)

        macro_st_rae = torch.tensor(
            result["macro_st_rae"],
            dtype=torch.float32,
            device=self.device,
        )

        self.log(
            "val_macro_st_rae",
            macro_st_rae,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )

        self._last_validation_st_rae = result


def build_validation_callbacks(
    checkpoint_dir: str | Path,
    patience: int = 20,
    min_delta: float = 0.0,
):
    """
    Build the frozen CYP-002 checkpoint/early-stopping callbacks.
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    checkpoint_dir = Path(checkpoint_dir)

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_macro_st_rae",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_macro_st_rae",
        mode="min",
        patience=patience,
        min_delta=min_delta,
    )

    result = [checkpoint, early_stopping]
    return result
