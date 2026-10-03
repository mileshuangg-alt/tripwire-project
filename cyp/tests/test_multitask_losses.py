from __future__ import annotations

import torch

from multitask_losses import task_balanced_masked_mse


def test_task_balanced_loss_equalizes_task_contributions():
    preds = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0, 1.0],
            [5.0, 1.0, 1.0, 1.0],
            [7.0, 1.0, 1.0, 1.0],
        ]
    )

    targets = torch.zeros_like(preds)

    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [True, False, True, True],
            [True, False, False, True],
        ]
    )

    # Task 1 observed losses: 1, 9, 25, 49 -> mean = 21
    # Task 2 observed loss: 1 -> mean = 1
    # Task 3 observed losses: 1, 1 -> mean = 1
    # Task 4 observed losses: 1, 1 -> mean = 1
    #
    # D001 objective = (21 + 1 + 1 + 1) / 4 = 6.0
    result = task_balanced_masked_mse(preds, targets, mask)

    assert torch.isclose(result, torch.tensor(6.0))


def test_zero_label_task_contributes_zero_without_renormalization():
    preds = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )

    targets = torch.zeros_like(preds)

    mask = torch.tensor(
        [
            [True, False, True, False],
            [True, False, True, False],
        ]
    )

    # Task 1 mean = 4
    # Task 2 mean = 0 (no labels)
    # Task 3 mean = 4
    # Task 4 mean = 0 (no labels)
    #
    # Fixed four-task denominator:
    # (4 + 0 + 4 + 0) / 4 = 2
    result = task_balanced_masked_mse(preds, targets, mask)

    assert torch.isclose(result, torch.tensor(2.0))


def test_missing_labels_do_not_contribute():
    preds = torch.tensor(
        [
            [2.0, 100.0, 2.0, 100.0],
        ]
    )

    targets = torch.zeros_like(preds)

    mask = torch.tensor(
        [
            [True, False, True, False],
        ]
    )

    # Only tasks 1 and 3 contribute:
    # (4 + 0 + 4 + 0) / 4 = 2
    result = task_balanced_masked_mse(preds, targets, mask)

    assert torch.isclose(result, torch.tensor(2.0))


def test_invalid_shapes_are_rejected():
    preds = torch.zeros(2, 4)
    targets = torch.zeros(2, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)

    try:
        task_balanced_masked_mse(preds, targets, mask)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for mismatched target shape")


def run_tests():
    tests = [
        test_task_balanced_loss_equalizes_task_contributions,
        test_zero_label_task_contributes_zero_without_renormalization,
        test_missing_labels_do_not_contribute,
        test_invalid_shapes_are_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} multitask loss tests")


if __name__ == "__main__":
    run_tests()
