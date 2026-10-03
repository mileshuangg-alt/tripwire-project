from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cyp002_config import CYP002_TASK_NAMES
from cyp002_validation import (
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
        test_checkpoint_callbacks_monitor_macro_st_rae,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} validation-monitor tests")


if __name__ == "__main__":
    run_tests()
