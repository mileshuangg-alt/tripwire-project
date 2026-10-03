from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from chemprop.nn import MSE
from multitask_targets import CYP_TARGET_COLUMNS
from multitask_losses import TaskBalancedMSE


DATA_PATH = "cyp/data/cyp-challenge-TRAIN_inhibition.csv"
N_TASKS = 4
TASK_OFFSETS = torch.tensor([1.0, 2.0, 3.0, 4.0])


def load_deterministic_real_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    dataframe = pd.read_csv(DATA_PATH)

    labels = dataframe.loc[:, list(CYP_TARGET_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    for n_rows in range(8, min(128, len(labels)) + 1):
        batch = labels.iloc[:n_rows]
        mask_np = ~batch.isna().to_numpy()

        counts = mask_np.sum(axis=0)

        if np.all(counts > 0) and not np.all(counts == counts[0]):
            targets_np = batch.to_numpy(dtype=np.float32)

            # Fill missing targets only for tensor arithmetic.
            # The observation mask remains authoritative.
            targets_filled = np.nan_to_num(
                targets_np,
                nan=0.0,
            )

            targets = torch.tensor(
                targets_filled,
                dtype=torch.float32,
            )
            mask = torch.tensor(
                mask_np,
                dtype=torch.bool,
            )

            # Deliberately introduce task-specific squared errors:
            # task 1 = 1, task 2 = 4, task 3 = 9, task 4 = 16.
            preds = targets + TASK_OFFSETS

            return preds, targets, mask, counts

    raise AssertionError(
        "Could not construct a deterministic real CYP batch with all four "
        "tasks observed and unequal label counts."
    )


def test_stock_and_task_balanced_use_same_real_cyp_batch():
    preds, targets, mask, counts = load_deterministic_real_batch()

    weights = torch.ones(preds.shape[0])
    lt_mask = torch.zeros_like(mask)
    gt_mask = torch.zeros_like(mask)

    stock = MSE()
    challenger = TaskBalancedMSE()

    stock_loss = stock(
        preds,
        targets,
        mask,
        weights,
        lt_mask,
        gt_mask,
    )

    challenger_loss = challenger(
        preds,
        targets,
        mask,
        weights,
        lt_mask,
        gt_mask,
    )

    assert stock_loss.ndim == 0
    assert challenger_loss.ndim == 0

    expected_task_losses = TASK_OFFSETS.square()

    # Variant 2: equal 1/4 weighting across the four tasks.
    expected_balanced = expected_task_losses.mean()

    torch.testing.assert_close(
        challenger_loss,
        expected_balanced,
        atol=1e-6,
        rtol=1e-6,
    )

    # Variant 1: equal weighting of observed labels.
    expected_stock = (
        expected_task_losses
        * torch.tensor(counts, dtype=torch.float32)
    ).sum() / float(counts.sum())

    torch.testing.assert_close(
        stock_loss,
        expected_stock,
        atol=1e-6,
        rtol=1e-6,
    )

    # The real batch must actually exercise unequal label counts.
    assert not np.all(counts == counts[0])

    # The two objectives must differ for this intentionally constructed batch.
    assert not torch.isclose(
        stock_loss,
        challenger_loss,
        atol=1e-6,
        rtol=1e-6,
    )


def run_tests() -> None:
    tests = [
        test_stock_and_task_balanced_use_same_real_cyp_batch,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} real-CYP criterion tests")


if __name__ == "__main__":
    run_tests()
