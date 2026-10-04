from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from cyp002_config import CYP002_TASK_NAMES
import cyp002_validation as validation_module
from cyp002_validation import (
    CYP002ValidationMPNN,
    build_validation_callbacks,
    extract_validation_confidence_intervals,
    score_collected_validation_epoch,
)


def test_confidence_intervals_preserve_dataframe_order():
    dataframe = pd.DataFrame(
        {
            "CYP1A2_pIC50_direct_inhibition_conf_low": [1.0, 2.0],
            "CYP1A2_pIC50_direct_inhibition_conf_high": [1.5, 2.5],
            "CYP2C9_pIC50_direct_inhibition_conf_low": [3.0, 4.0],
            "CYP2C9_pIC50_direct_inhibition_conf_high": [3.5, 4.5],
            "CYP2D6_pIC50_direct_inhibition_conf_low": [5.0, 6.0],
            "CYP2D6_pIC50_direct_inhibition_conf_high": [5.5, 6.5],
            "CYP3A4_pIC50_direct_inhibition_conf_low": [7.0, 8.0],
            "CYP3A4_pIC50_direct_inhibition_conf_high": [7.5, 8.5],
        }
    )

    low, high = extract_validation_confidence_intervals(
        dataframe,
        CYP002_TASK_NAMES,
    )

    assert low["CYP1A2_pIC50_direct_inhibition"].tolist() == [1.0, 2.0]
    assert high["CYP1A2_pIC50_direct_inhibition"].tolist() == [1.5, 2.5]
    assert low["CYP3A4_pIC50_direct_inhibition"].tolist() == [7.0, 8.0]
    assert high["CYP3A4_pIC50_direct_inhibition"].tolist() == [7.5, 8.5]


def test_validation_epoch_requires_confidence_interval_alignment():
    predictions = [
        torch.tensor(
            [
                [5.0, 6.0, 7.0, 8.0],
                [5.5, 6.5, 7.5, 8.5],
            ]
        )
    ]

    targets = [predictions[0].clone()]
    masks = [
        torch.ones(
            2,
            4,
            dtype=torch.bool,
        )
    ]

    conf_low = {
        task_name: np.array([0.0])
        for task_name in CYP002_TASK_NAMES
    }
    conf_high = {
        task_name: np.array([10.0, 10.0])
        for task_name in CYP002_TASK_NAMES
    }

    try:
        score_collected_validation_epoch(
            predictions,
            targets,
            masks,
            conf_low,
            conf_high,
            CYP002_TASK_NAMES,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected validation CI/prediction alignment failure"
    )


def test_validation_epoch_filters_missing_labels_before_scoring():
    predictions = [
        torch.tensor(
            [
                [5.1, 6.1, 7.1, 8.1],
                [5.2, 6.2, 7.2, 8.2],
            ]
        )
    ]

    targets = [
        torch.tensor(
            [
                [5.0, 6.0, 7.0, 8.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
    ]

    masks = [
        torch.tensor(
            [
                [True, True, False, True],
                [True, False, True, True],
            ],
            dtype=torch.bool,
        )
    ]

    # Deliberately non-degenerate synthetic intervals. They are offset
    # from the true values so the frozen ST-RAE denominator is nonzero.
    conf_low = {
        task_name: np.array([5.2, 5.2])
        for task_name in CYP002_TASK_NAMES
    }

    conf_high = {
        task_name: np.array([5.3, 5.3])
        for task_name in CYP002_TASK_NAMES
    }

    result = score_collected_validation_epoch(
        predictions,
        targets,
        masks,
        conf_low,
        conf_high,
        CYP002_TASK_NAMES,
    )

    assert set(result["per_task"]) == set(CYP002_TASK_NAMES)

    # Every returned task must have been scored only on its observed rows.
    assert result["task_order"] == list(CYP002_TASK_NAMES)
    assert np.isfinite(result["macro_st_rae"])


def test_validation_epoch_unscales_targets_before_scoring():
    predictions = [
        torch.tensor(
            [
                [11.0, 22.0, 33.0, 44.0],
                [12.0, 24.0, 36.0, 48.0],
            ]
        )
    ]
    targets = [
        torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
    ]
    masks = [
        torch.ones(
            2,
            4,
            dtype=torch.bool,
        )
    ]
    conf_low = {
        task_name: np.array([0.0, 0.0])
        for task_name in CYP002_TASK_NAMES
    }
    conf_high = {
        task_name: np.array([10.0, 10.0])
        for task_name in CYP002_TASK_NAMES
    }
    target_mean = np.array([10.0, 20.0, 30.0, 40.0])
    target_scale = np.array([2.0, 4.0, 6.0, 8.0])
    captured = {}

    def fake_compute_validation_macro_st_rae(
        y_true,
        y_pred,
        conf_low,
        conf_high,
        task_names,
    ):
        captured["y_true"] = y_true
        captured["y_pred"] = y_pred
        result = {
            "macro_st_rae": 0.0,
            "per_task": {task_name: 0.0 for task_name in task_names},
            "task_order": list(task_names),
        }
        return result

    original_compute = validation_module.compute_validation_macro_st_rae
    validation_module.compute_validation_macro_st_rae = (
        fake_compute_validation_macro_st_rae
    )
    try:
        result = score_collected_validation_epoch(
            predictions,
            targets,
            masks,
            conf_low,
            conf_high,
            CYP002_TASK_NAMES,
            target_mean=target_mean,
            target_scale=target_scale,
        )
    finally:
        validation_module.compute_validation_macro_st_rae = original_compute

    assert result["macro_st_rae"] == 0.0

    for task_index, task_name in enumerate(CYP002_TASK_NAMES):
        expected_true = (
            targets[0].numpy()[:, task_index] * target_scale[task_index]
            + target_mean[task_index]
        )
        np.testing.assert_allclose(
            captured["y_true"][task_name],
            expected_true,
        )
        np.testing.assert_allclose(
            captured["y_pred"][task_name],
            predictions[0].numpy()[:, task_index],
        )


def test_validation_monitor_skips_sanity_check_and_scores_full_epoch_once():
    class DummyTrainer:
        def __init__(self, sanity_checking):
            self.sanity_checking = sanity_checking

    rows = 736
    sanity_rows = 128
    model = object.__new__(CYP002ValidationMPNN)
    pl.LightningModule.__init__(model)
    model._trainer = DummyTrainer(sanity_checking=True)
    model.validation_task_names = tuple(CYP002_TASK_NAMES)
    model.validation_conf_low = {
        task_name: np.zeros(rows, dtype=np.float64)
        for task_name in CYP002_TASK_NAMES
    }
    model.validation_conf_high = {
        task_name: np.ones(rows, dtype=np.float64)
        for task_name in CYP002_TASK_NAMES
    }
    model.target_mean = np.array([1.0, 2.0, 3.0, 4.0])
    model.target_scale = np.array([0.5, 0.6, 0.7, 0.8])
    model.validation_history = []
    model._validation_predictions = [torch.zeros(sanity_rows, 4)]
    model._validation_targets = [torch.zeros(sanity_rows, 4)]
    model._validation_masks = [
        torch.ones(
            sanity_rows,
            4,
            dtype=torch.bool,
        )
    ]

    calls = []
    logged = []

    def fake_score_collected_validation_epoch(
        prediction_batches,
        target_batches,
        mask_batches,
        conf_low,
        conf_high,
        task_names,
        target_mean=None,
        target_scale=None,
    ):
        predictions = torch.cat(prediction_batches, dim=0)
        targets = torch.cat(target_batches, dim=0)
        masks = torch.cat(mask_batches, dim=0)
        calls.append(
            {
                "rows": predictions.shape[0],
                "target_rows": targets.shape[0],
                "mask_rows": masks.shape[0],
                "mask_value": bool(masks[0, 1]),
                "conf_rows": {
                    task_name: len(conf_low[task_name])
                    for task_name in task_names
                },
                "target_mean": target_mean,
                "target_scale": target_scale,
            }
        )
        result = {
            "macro_st_rae": 1.25,
            "per_task": {task_name: 1.25 for task_name in task_names},
            "task_order": list(task_names),
        }
        return result

    def fake_log(self, name, value, **kwargs):
        logged.append((name, value, kwargs))

    original_score = validation_module.score_collected_validation_epoch
    original_log = CYP002ValidationMPNN.log
    validation_module.score_collected_validation_epoch = (
        fake_score_collected_validation_epoch
    )
    CYP002ValidationMPNN.log = fake_log

    try:
        CYP002ValidationMPNN.on_validation_epoch_end(model)

        assert calls == []
        assert logged == []
        assert model.validation_history == []

        model._trainer.sanity_checking = False
        CYP002ValidationMPNN.on_validation_epoch_start(model)

        batch_sizes = [128, 128, 128, 128, 128, 96]
        model._validation_predictions = [
            torch.full((batch_size, 4), float(index))
            for index, batch_size in enumerate(batch_sizes)
        ]
        model._validation_targets = [
            torch.full((batch_size, 4), float(index + 10))
            for index, batch_size in enumerate(batch_sizes)
        ]
        model._validation_masks = [
            torch.ones(
                batch_size,
                4,
                dtype=torch.bool,
            )
            for batch_size in batch_sizes
        ]
        model._validation_masks[0][0, 1] = False

        CYP002ValidationMPNN.on_validation_epoch_end(model)
    finally:
        validation_module.score_collected_validation_epoch = original_score
        CYP002ValidationMPNN.log = original_log

    assert len(calls) == 1
    assert calls[0]["rows"] == rows
    assert calls[0]["target_rows"] == rows
    assert calls[0]["mask_rows"] == rows
    assert calls[0]["mask_value"] is False
    assert calls[0]["conf_rows"] == {
        task_name: rows for task_name in CYP002_TASK_NAMES
    }
    np.testing.assert_allclose(calls[0]["target_mean"], model.target_mean)
    np.testing.assert_allclose(calls[0]["target_scale"], model.target_scale)
    assert len(model.validation_history) == 1
    assert model.validation_history[0]["macro_st_rae"] == 1.25
    assert len(logged) == 1
    assert logged[0][0] == "val_macro_st_rae"
    assert logged[0][2]["on_epoch"] is True
    assert logged[0][2]["on_step"] is False


def test_checkpoint_callbacks_monitor_macro_st_rae():
    callbacks = build_validation_callbacks(
        "tmp-cyp002-checkpoints",
        patience=20,
        min_delta=0.0,
    )

    assert len(callbacks) == 2

    checkpoint = callbacks[0]
    early_stopping = callbacks[1]

    assert checkpoint.monitor == "val_macro_st_rae"
    assert checkpoint.mode == "min"

    assert early_stopping.monitor == "val_macro_st_rae"
    assert early_stopping.mode == "min"
    assert early_stopping.patience == 20
    assert early_stopping.min_delta == 0.0


def run_tests() -> None:
    tests = [
        test_confidence_intervals_preserve_dataframe_order,
        test_validation_epoch_requires_confidence_interval_alignment,
        test_validation_epoch_filters_missing_labels_before_scoring,
        test_validation_epoch_unscales_targets_before_scoring,
        test_validation_monitor_skips_sanity_check_and_scores_full_epoch_once,
        test_checkpoint_callbacks_monitor_macro_st_rae,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} validation-monitor tests")


if __name__ == "__main__":
    run_tests()
