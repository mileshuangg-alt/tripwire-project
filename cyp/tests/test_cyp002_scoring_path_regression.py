from __future__ import annotations

import numpy as np
import pytest
import torch

from cyp002_validation import score_collected_validation_epoch
from references.cyp_upstream.custom_scoring_functions import (
    rae_soft_threshold_absolute_error,
)


def test_validation_scoring_unscales_collected_predictions_before_st_rae():
    task_names = ("task_a", "task_b")
    target_mean = np.array([10.0, 20.0], dtype=np.float64)
    target_scale = np.array([2.0, 4.0], dtype=np.float64)

    y_true_original = np.array(
        [
            [8.0, 16.0],
            [10.0, 20.0],
            [12.0, 24.0],
        ],
        dtype=np.float64,
    )
    y_pred_original = y_true_original.copy()
    y_true_scaled = (y_true_original - target_mean) / target_scale
    y_pred_scaled = (y_pred_original - target_mean) / target_scale

    conf_low = {
        "task_a": y_true_original[:, 0] - 0.25,
        "task_b": y_true_original[:, 1] - 0.25,
    }
    conf_high = {
        "task_a": y_true_original[:, 0] + 0.25,
        "task_b": y_true_original[:, 1] + 0.25,
    }

    expected_per_task = {}
    for index, task_name in enumerate(task_names):
        expected_per_task[task_name] = rae_soft_threshold_absolute_error(
            y_true_original[:, index],
            y_pred_original[:, index],
            y_true_lower=conf_low[task_name],
            y_true_upper=conf_high[task_name],
        )

    result = score_collected_validation_epoch(
        prediction_batches=[torch.tensor(y_pred_scaled, dtype=torch.float32)],
        target_batches=[torch.tensor(y_true_scaled, dtype=torch.float32)],
        mask_batches=[torch.ones_like(torch.tensor(y_true_scaled), dtype=torch.bool)],
        conf_low=conf_low,
        conf_high=conf_high,
        task_names=task_names,
        target_mean=target_mean,
        target_scale=target_scale,
    )

    assert result["per_task"] == pytest.approx(expected_per_task)
    assert result["macro_st_rae"] == pytest.approx(
        float(np.mean(list(expected_per_task.values())))
    )
