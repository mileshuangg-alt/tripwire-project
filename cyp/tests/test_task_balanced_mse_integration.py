from __future__ import annotations

import torch
from torch import nn

from chemprop.nn import MSE
from multitask_losses import TaskBalancedMSE


N_TASKS = 4
ATOL = 1e-7
RTOL = 1e-6


class SharedToyModel(nn.Module):
    """Minimal shared-trunk + task-head model for gradient-path tests."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(1.0))
        self.heads = nn.Parameter(torch.ones(N_TASKS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) * self.shared * self.heads


def build_uneven_fixture():
    """Four tasks with 4, 3, 2, and 1 observed labels respectively.

    All observed examples use x=1 and target=0, so the unweighted
    per-task mean gradients into the shared parameter are equal and nonzero.
    """
    model = SharedToyModel()

    x = torch.ones(4)
    preds = model(x)
    targets = torch.zeros_like(preds)

    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, False],
            [True, False, False, True],
        ],
        dtype=torch.bool,
    )

    weights = torch.ones(4)
    lt_mask = torch.zeros_like(mask)
    gt_mask = torch.zeros_like(mask)

    return model, preds, targets, mask, weights, lt_mask, gt_mask


def unweighted_task_mean_mse_gradients(
    model: SharedToyModel,
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Return each task's unweighted mean-MSE gradient into the shared trunk."""
    gradients = []

    for task_index in range(N_TASKS):
        present = mask[:, task_index]

        if present.any():
            task_loss = (
                (preds[present, task_index] - targets[present, task_index])
                .square()
                .mean()
            )
        else:
            task_loss = preds[:, task_index].sum() * 0.0

        gradient = torch.autograd.grad(
            task_loss,
            model.shared,
            retain_graph=True,
            create_graph=False,
        )[0]

        gradients.append(gradient)

    return gradients


def test_both_variants_use_the_same_chemprop_criterion_interface():
    (
        _model,
        preds,
        targets,
        mask,
        weights,
        lt_mask,
        gt_mask,
    ) = build_uneven_fixture()

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
    assert torch.isfinite(stock_loss)
    assert torch.isfinite(challenger_loss)


def test_fixture_unweighted_task_gradients_are_equal_and_nonzero():
    model, preds, targets, mask, *_ = build_uneven_fixture()

    gradients = unweighted_task_mean_mse_gradients(
        model,
        preds,
        targets,
        mask,
    )

    norms = torch.stack([gradient.abs() for gradient in gradients])

    # This is a fixture precondition, not the property being tested.
    # It must fail loudly if the synthetic fixture becomes degenerate.
    assert torch.all(norms > 0), (
        f"Fixture produced a zero per-task gradient: {norms.tolist()}"
    )

    for i in range(1, N_TASKS):
        torch.testing.assert_close(
            gradients[i],
            gradients[0],
            atol=ATOL,
            rtol=RTOL,
        )


def test_task_balanced_gradient_contributions_are_one_quarter_of_total():
    (
        model,
        preds,
        targets,
        mask,
        weights,
        lt_mask,
        gt_mask,
    ) = build_uneven_fixture()

    task_gradients = unweighted_task_mean_mse_gradients(
        model,
        preds,
        targets,
        mask,
    )

    # Reassert the fixture invariant here so this test cannot pass
    # vacuously if the fixture is accidentally changed later.
    norms = torch.stack([gradient.abs() for gradient in task_gradients])

    assert torch.all(norms > 0)
    for i in range(1, N_TASKS):
        torch.testing.assert_close(
            task_gradients[i],
            task_gradients[0],
            atol=ATOL,
            rtol=RTOL,
        )

    criterion = TaskBalancedMSE()

    loss = criterion(
        preds,
        targets,
        mask,
        weights,
        lt_mask,
        gt_mask,
    )

    combined_gradient = torch.autograd.grad(
        loss,
        model.shared,
        retain_graph=True,
        create_graph=False,
    )[0]

    weighted_task_gradients = [
        gradient / N_TASKS
        for gradient in task_gradients
    ]

    for weighted_gradient in weighted_task_gradients:
        torch.testing.assert_close(
            weighted_gradient,
            combined_gradient / N_TASKS,
            atol=ATOL,
            rtol=RTOL,
        )

    total_norm = combined_gradient.abs()
    for weighted_gradient in weighted_task_gradients:
        torch.testing.assert_close(
            weighted_gradient.abs(),
            total_norm / N_TASKS,
            atol=ATOL,
            rtol=RTOL,
        )


def test_zero_label_task_has_zero_head_and_shared_gradient_contribution():
    model = SharedToyModel()

    x = torch.ones(2)
    preds = model(x)
    targets = torch.zeros_like(preds)

    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
        ],
        dtype=torch.bool,
    )

    weights = torch.ones(2)
    lt_mask = torch.zeros_like(mask)
    gt_mask = torch.zeros_like(mask)

    criterion = TaskBalancedMSE()

    task_losses = criterion.task_losses(
        preds,
        targets,
        mask,
        weights,
    )

    assert task_losses.shape == (N_TASKS,)
    assert task_losses[3].item() == 0.0

    # Inspect the zero-label task contribution BEFORE consuming the graph
    # with the full backward pass.
    zero_task_gradient = torch.autograd.grad(
        task_losses[3],
        model.shared,
        retain_graph=True,
        allow_unused=True,
    )[0]

    assert zero_task_gradient is not None
    assert zero_task_gradient.item() == 0.0

    loss = criterion(
        preds,
        targets,
        mask,
        weights,
        lt_mask,
        gt_mask,
    )

    # The criterion must remain connected to the graph even when one
    # task has no labels, so backward should succeed.
    loss.backward()

    # Task 4 has no observed labels, therefore its head receives
    # exactly zero gradient.
    assert model.heads.grad[3].item() == 0.0

    # The fixed four-task denominator remains:
    # (1 + 1 + 1 + 0) / 4 = 0.75
    assert loss.item() == 0.75


def test_zero_label_task_is_not_used_as_a_present_task_for_renormalization():
    model = SharedToyModel()

    x = torch.ones(2)
    preds = model(x)
    targets = torch.zeros_like(preds)

    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
        ],
        dtype=torch.bool,
    )

    weights = torch.ones(2)
    lt_mask = torch.zeros_like(mask)
    gt_mask = torch.zeros_like(mask)

    criterion = TaskBalancedMSE()

    loss = criterion(
        preds,
        targets,
        mask,
        weights,
        lt_mask,
        gt_mask,
    )

    # Fixed four-task denominator gives:
    # (1 + 1 + 1 + 0) / 4 = 0.75
    # Renormalization over present tasks would incorrectly give 1.0.
    assert loss.item() == 0.75


def run_tests() -> None:
    tests = [
        test_both_variants_use_the_same_chemprop_criterion_interface,
        test_fixture_unweighted_task_gradients_are_equal_and_nonzero,
        test_task_balanced_gradient_contributions_are_one_quarter_of_total,
        test_zero_label_task_has_zero_head_and_shared_gradient_contribution,
        test_zero_label_task_is_not_used_as_a_present_task_for_renormalization,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"PASS: all {len(tests)} Chemprop integration tests")


if __name__ == "__main__":
    run_tests()
