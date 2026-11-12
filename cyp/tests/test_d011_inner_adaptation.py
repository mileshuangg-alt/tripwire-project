from __future__ import annotations

import numpy as np
import pytest
import torch

from d011_inner_adaptation import (
    D010_ANCHOR_O150,
    D011CandidateResult,
    d010_anchor_candidate,
    d011_anchor_compatibility_record,
    d011_candidate_grid,
    d011_new_candidate_grid,
    d011_workload_plan,
    results_with_d010_anchor,
    select_d011_candidate,
)
from p2_anil_model import (
    P2_EMBEDDING_DIM,
    P2MetaHead,
    adapt_selected_output_exact,
    query_loss_after_exact_adaptation,
    selected_task_prediction,
    standardized_mse,
)
from p2_anil_training import D011Candidate, full_support_task_adaptation_prediction


def _support_fixture():
    torch.manual_seed(11011)
    head = P2MetaHead().double()
    support_x = torch.randn(13, P2_EMBEDDING_DIM, dtype=torch.float64, requires_grad=True)
    query_x = torch.randn(3, P2_EMBEDDING_DIM, dtype=torch.float64, requires_grad=True)
    support_y = torch.linspace(-1.0, 1.0, 13, dtype=torch.float64)
    query_y = torch.tensor([-0.25, 0.5, 0.75], dtype=torch.float64)
    return head, support_x, support_y, query_x, query_y


def _manual_k_step(head, support_x, support_y, *, task_index, inner_lr, inner_steps):
    features = head.features(support_x)
    weight = head.output.weight[task_index]
    bias = head.output.bias[task_index]
    losses = []
    for _ in range(inner_steps):
        pred = selected_task_prediction(features, task_index=task_index, weight=weight, bias=bias)
        loss = standardized_mse(pred, support_y)
        grad_w, grad_b = torch.autograd.grad(loss, (weight, bias), create_graph=True, retain_graph=True)
        weight = weight - inner_lr * grad_w
        bias = bias - inner_lr * grad_b
        losses.append(loss)
    return weight, bias, losses


@pytest.mark.parametrize("inner_steps", [1, 2, 3])
def test_k_step_exact_anil_sequential_updates_and_gradients(inner_steps):
    head, support_x, support_y, query_x, query_y = _support_fixture()
    torch.manual_seed(999)
    adapted_w, adapted_b, support_loss = adapt_selected_output_exact(
        head,
        support_x,
        support_y,
        task_index=0,
        inner_lr=0.003,
        inner_steps=inner_steps,
    )
    torch.manual_seed(999)
    manual_w, manual_b, manual_losses = _manual_k_step(
        head,
        support_x,
        support_y,
        task_index=0,
        inner_lr=0.003,
        inner_steps=inner_steps,
    )
    assert len(manual_losses) == inner_steps
    torch.testing.assert_close(adapted_w, manual_w)
    torch.testing.assert_close(adapted_b, manual_b)
    torch.testing.assert_close(support_loss, manual_losses[-1])
    query_loss, _ = query_loss_after_exact_adaptation(
        head,
        support_x,
        support_y,
        query_x,
        query_y,
        task_index=0,
        inner_lr=0.003,
        inner_steps=inner_steps,
    )
    grad_hidden = torch.autograd.grad(query_loss, head.shared_hidden.weight, retain_graph=True)[0]
    grad_output = torch.autograd.grad(query_loss, head.output.weight, retain_graph=True)[0][0]
    assert support_loss.grad_fn is not None
    assert torch.isfinite(grad_hidden).all()
    assert torch.isfinite(grad_output).all()


def test_k_step_inner_adaptation_does_not_mutate_module_parameters():
    head, support_x, support_y, *_ = _support_fixture()
    before = {name: value.detach().clone() for name, value in head.state_dict().items()}
    adapt_selected_output_exact(head, support_x, support_y, task_index=2, inner_lr=0.03, inner_steps=3)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_evaluation_uses_candidate_specific_k_and_fresh_state_per_isoform():
    torch.manual_seed(42)
    head = P2MetaHead().double()
    train_x = torch.randn(24, P2_EMBEDDING_DIM, dtype=torch.float64, requires_grad=True)
    val_x = torch.randn(4, P2_EMBEDDING_DIM, dtype=torch.float64)
    labels = torch.randn(24, 4, dtype=torch.float64)
    mask = torch.ones(24, 4, dtype=torch.bool)
    means = labels.mean(dim=0)
    scales = labels.std(dim=0, unbiased=False).clamp_min(0.1)
    before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    pred_k1 = full_support_task_adaptation_prediction(
        meta_head=head,
        train_embeddings=train_x,
        train_labels=labels,
        train_mask=mask,
        validation_embeddings=val_x,
        scaler_means=means,
        scaler_scales=scales,
        task_index=1,
        inner_lr=0.01,
        inner_steps=1,
    )
    pred_k3 = full_support_task_adaptation_prediction(
        meta_head=head,
        train_embeddings=train_x,
        train_labels=labels,
        train_mask=mask,
        validation_embeddings=val_x,
        scaler_means=means,
        scaler_scales=scales,
        task_index=1,
        inner_lr=0.01,
        inner_steps=3,
    )
    assert not torch.allclose(pred_k1, pred_k3)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_d010_anchor_compatibility_and_workload_plan():
    record = d011_anchor_compatibility_record()
    assert record["compatible"] is True
    assert d010_anchor_candidate() == D011Candidate(0.01, 1)
    assert record["anchor_o150"] == pytest.approx(D010_ANCHOR_O150)
    plan = d011_workload_plan(anchor_reusable=True)
    assert plan["configuration_count"] == 12
    assert plan["anchor_reused"] is True
    assert plan["new_candidate_count"] == 11
    assert plan["new_cell_count"] == 1650
    assert len(d011_candidate_grid()) == 12
    assert len(d011_new_candidate_grid()) == 11
    assert d010_anchor_candidate() not in d011_new_candidate_grid()


def test_d011_selection_tie_breaking_and_advancement_arithmetic():
    rows = []
    for candidate in d011_candidate_grid():
        rows.append(D011CandidateResult(candidate=candidate, o150=0.8, source="synthetic"))
    rows[0] = D011CandidateResult(candidate=rows[0].candidate, o150=0.7, source="best")
    selected = select_d011_candidate(rows)
    assert selected["selected_o150"] == pytest.approx(0.7)
    assert selected["advances"] is True
    tied = [D011CandidateResult(candidate=c, o150=0.75, source="tie") for c in d011_candidate_grid()]
    selected_tie = select_d011_candidate(tied)
    assert selected_tie["selected_candidate"] == {"inner_lr": 0.001, "inner_steps": 1}
    no_win = [D011CandidateResult(candidate=c, o150=0.701212576796, source="tie_t121") for c in d011_candidate_grid()]
    assert select_d011_candidate(no_win)["advances"] is False


def test_results_with_d010_anchor_requires_11_new_results():
    new = [
        D011CandidateResult(candidate=c, o150=0.8 + i * 0.001, source="new")
        for i, c in enumerate(d011_new_candidate_grid())
    ]
    combined = results_with_d010_anchor(new)
    assert len(combined) == 12
    assert any(row.candidate == d010_anchor_candidate() and row.source == "frozen D010 anchor" for row in combined)
    with pytest.raises(ValueError):
        results_with_d010_anchor(new[:-1])
