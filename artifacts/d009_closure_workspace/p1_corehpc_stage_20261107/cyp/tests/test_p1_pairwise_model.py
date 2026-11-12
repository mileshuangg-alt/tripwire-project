from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import p1_pairwise

from p1_pairwise import (
    compute_pairwise_partition_contract,
    sample_supervised_pairs,
)
from p1_pairwise_model import (
    P1AntisymmetricPairHead,
    normalized_weighted_pair_mse,
)


ENDPOINTS = ("E1", "E2", "E3", "E4")


def test_p1_pair_head_has_frozen_architecture_and_parameter_count():
    model = P1AntisymmetricPairHead()

    layers = list(model.pair_function)

    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 1200
    assert layers[0].out_features == 2000

    assert isinstance(layers[1], nn.ReLU)

    assert isinstance(layers[2], nn.Dropout)
    assert layers[2].p == 0.4

    assert isinstance(layers[3], nn.Linear)
    assert layers[3].in_features == 2000
    assert layers[3].out_features == 4

    expected_parameters = (
        1200 * 2000
        + 2000
        + 2000 * 4
        + 4
    )

    observed_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert observed_parameters == expected_parameters
    assert trainable_parameters == expected_parameters


def test_p1_pair_head_eval_is_antisymmetric_and_self_zero():
    torch.manual_seed(20261107)

    model = P1AntisymmetricPairHead()
    model.eval()

    h_a = torch.randn(5, 600)
    h_b = torch.randn(5, 600)

    with torch.no_grad():
        ab = model(h_a, h_b)
        ba = model(h_b, h_a)
        aa = model(h_a, h_a)

    torch.testing.assert_close(
        ab,
        -ba,
        rtol=1e-6,
        atol=1e-7,
    )

    torch.testing.assert_close(
        aa,
        torch.zeros_like(aa),
        rtol=0.0,
        atol=0.0,
    )


def test_p1_pair_head_train_mode_has_finite_forward_and_backward():
    torch.manual_seed(20261107)

    model = P1AntisymmetricPairHead()
    model.train()

    h_a = torch.randn(
        4,
        600,
        requires_grad=True,
    )
    h_b = torch.randn(
        4,
        600,
        requires_grad=True,
    )

    output = model(h_a, h_b)

    assert output.shape == (4, 4)
    assert torch.isfinite(output).all()

    scalar = output.square().mean()
    scalar.backward()

    assert h_a.grad is not None
    assert h_b.grad is not None
    assert torch.isfinite(h_a.grad).all()
    assert torch.isfinite(h_b.grad).all()

    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in parameter_gradients
    )

    assert all(
        torch.isfinite(gradient).all()
        for gradient in parameter_gradients
    )


def test_normalized_weighted_pair_mse_matches_sum_mw_definition():
    preds = torch.tensor(
        [
            [2.0, 4.0, 8.0, 16.0],
            [3.0, 5.0, 9.0, 17.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    targets = torch.tensor(
        [
            [1.0, float("nan"), 6.0, float("nan")],
            [1.0, 4.0, float("nan"), 13.0],
        ],
        dtype=torch.float64,
    )

    mask = torch.tensor(
        [
            [True, False, True, False],
            [True, True, False, True],
        ],
        dtype=torch.bool,
    )

    weights = torch.tensor(
        [2.0, 3.0, 5.0, 7.0],
        dtype=torch.float64,
    )

    # Eligible weighted squared errors:
    #
    # row 0:
    # E1 = 2 * (2 - 1)^2 = 2
    # E3 = 5 * (8 - 6)^2 = 20
    #
    # row 1:
    # E1 = 2 * (3 - 1)^2 = 8
    # E2 = 3 * (5 - 4)^2 = 3
    # E4 = 7 * (17 - 13)^2 = 112
    #
    # numerator   = 145
    # denominator = 2 + 5 + 2 + 3 + 7 = 19

    expected = torch.tensor(
        145.0 / 19.0,
        dtype=torch.float64,
    )

    loss = normalized_weighted_pair_mse(
        preds,
        targets,
        mask,
        weights,
    )

    torch.testing.assert_close(
        loss,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    loss.backward()

    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


def test_normalized_weighted_pair_mse_masks_nan_before_arithmetic():
    preds = torch.zeros(
        (1, 4),
        dtype=torch.float64,
        requires_grad=True,
    )

    targets = torch.tensor(
        [[1.0, float("nan"), 2.0, float("nan")]],
        dtype=torch.float64,
    )

    mask = torch.tensor(
        [[True, False, True, False]],
        dtype=torch.bool,
    )

    weights = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
    )

    loss = normalized_weighted_pair_mse(
        preds,
        targets,
        mask,
        weights,
    )

    assert torch.isfinite(loss)

    bad_targets = targets.clone()
    bad_targets[0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="eligible targets must be finite",
    ):
        normalized_weighted_pair_mse(
            preds,
            bad_targets,
            mask,
            weights,
        )


def test_sampled_batch_to_pair_head_to_weighted_loss_has_finite_gradients(
    monkeypatch,
):
    ids = ["a", "b", "c"]

    labels = [
        [1.0, 10.0, None, None],
        [2.0, 20.0, 5.0, None],
        [3.0, None, 7.0, None],
    ]

    means = (
        2.0,
        15.0,
        6.0,
        None,
    )

    scales = (
        1.0,
        10.0,
        2.0,
        None,
    )

    contract = compute_pairwise_partition_contract(
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
    )

    # N=3 ordered-distinct mapping:
    # k=0 -> (a,b)
    # k=3 -> (b,c)
    # k=4 -> (c,a)
    draws = iter([0, 3, 4])

    def controlled_draw(rng, pair_count):
        value = next(draws)
        assert 0 <= value < pair_count
        return value

    monkeypatch.setattr(
        p1_pairwise,
        "_draw_ordered_pair_index",
        controlled_draw,
    )

    batch = sample_supervised_pairs(
        contract,
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
        accepted_pair_count=3,
        rng=np.random.default_rng(20261107),
    )

    assert batch.accepted_count == 3

    mask = torch.tensor(
        batch.eligibility_masks,
        dtype=torch.bool,
    )

    targets = torch.tensor(
        batch.delta_targets,
        dtype=torch.float32,
    )

    weights = torch.tensor(
        [
            endpoint.weight
            if endpoint.weight is not None
            else 1.0
            for endpoint in contract.endpoints
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(20261107)

    embedding_table = {
        molecule_id: torch.randn(600)
        for molecule_id in contract.molecule_ids
    }

    h_a = torch.stack(
        [
            embedding_table[molecule_id]
            for molecule_id in batch.molecule_a_ids
        ]
    )

    h_b = torch.stack(
        [
            embedding_table[molecule_id]
            for molecule_id in batch.molecule_b_ids
        ]
    )

    model = P1AntisymmetricPairHead()
    model.train()

    predictions = model(h_a, h_b)

    loss = normalized_weighted_pair_mse(
        predictions,
        targets,
        mask,
        weights,
    )

    assert torch.isfinite(loss)

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in gradients
    )

    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_reference_reconstruction_exact_absolute_mean_and_dispersion():
    from p1_pairwise_model import (
        reconstruct_absolute_from_references,
    )

    # Two queries, three training references, four endpoints.
    predicted_delta_z = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0],
                [-1.0, 0.0, 1.0, 2.0],
            ],
            [
                [0.5, 1.0, 1.5, 2.0],
                [-0.5, 0.0, 0.5, 1.0],
                [-1.5, -1.0, -0.5, 0.0],
            ],
        ],
        dtype=torch.float64,
    )

    reference_y = torch.tensor(
        [
            [5.0, 10.0, float("nan"), 20.0],
            [7.0, float("nan"), 30.0, 22.0],
            [9.0, 14.0, 34.0, float("nan")],
        ],
        dtype=torch.float64,
    )

    reference_mask = torch.tensor(
        [
            [True, True, False, True],
            [True, False, True, True],
            [True, True, True, False],
        ],
        dtype=torch.bool,
    )

    scales = torch.tensor(
        [2.0, 1.0, 0.5, 2.0],
        dtype=torch.float64,
    )

    means, dispersion = (
        reconstruct_absolute_from_references(
            predicted_delta_z,
            reference_y,
            reference_mask,
            scales,
        )
    )

    # Query 0:
    # E1: [5+2, 7+0, 9-2] = [7,7,7] -> mean 7
    # E2: [10+2, 14+0] = [12,14] -> mean 13
    # E3: [30+1, 34+0.5] = [31,34.5] -> mean 32.75
    # E4: [20+8, 22+6] = [28,28] -> mean 28
    expected_q0 = torch.tensor(
        [7.0, 13.0, 32.75, 28.0],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        means[0],
        expected_q0,
        rtol=0.0,
        atol=1e-12,
    )

    expected_q0_dispersion = torch.tensor(
        [
            0.0,
            1.0,
            1.75,
            0.0,
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        dispersion[0],
        expected_q0_dispersion,
        rtol=0.0,
        atol=1e-12,
    )

    assert means.shape == (2, 4)
    assert dispersion.shape == (2, 4)
    assert torch.isfinite(means).all()
    assert torch.isfinite(dispersion).all()


def test_reference_reconstruction_uses_only_eligible_training_references():
    from p1_pairwise_model import (
        reconstruct_absolute_from_references,
    )

    predicted_delta_z = torch.zeros(
        (1, 3, 4),
        dtype=torch.float64,
    )

    reference_y = torch.tensor(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [3.0, 20.0, 200.0, 2000.0],
            [999999.0, 30.0, 300.0, 3000.0],
        ],
        dtype=torch.float64,
    )

    reference_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )

    scales = torch.ones(
        4,
        dtype=torch.float64,
    )

    means, _ = reconstruct_absolute_from_references(
        predicted_delta_z,
        reference_y,
        reference_mask,
        scales,
    )

    # The huge E1 value is masked and must not affect the mean.
    assert means[0, 0].item() == 2.0


def test_reference_reconstruction_single_reference_dispersion_is_zero():
    from p1_pairwise_model import (
        reconstruct_absolute_from_references,
    )

    predicted_delta_z = torch.tensor(
        [[[1.0, -1.0, 2.0, -2.0]]],
        dtype=torch.float64,
    )

    reference_y = torch.tensor(
        [[5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float64,
    )

    reference_mask = torch.ones(
        (1, 4),
        dtype=torch.bool,
    )

    scales = torch.tensor(
        [2.0, 2.0, 0.5, 0.5],
        dtype=torch.float64,
    )

    means, dispersion = (
        reconstruct_absolute_from_references(
            predicted_delta_z,
            reference_y,
            reference_mask,
            scales,
        )
    )

    expected = torch.tensor(
        [[7.0, 4.0, 8.0, 7.0]],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        means,
        expected,
    )

    torch.testing.assert_close(
        dispersion,
        torch.zeros_like(dispersion),
        rtol=0.0,
        atol=0.0,
    )


def test_reference_reconstruction_fails_without_endpoint_reference():
    from p1_pairwise_model import (
        reconstruct_absolute_from_references,
    )

    predicted_delta_z = torch.zeros(
        (1, 2, 4),
        dtype=torch.float64,
    )

    reference_y = torch.tensor(
        [
            [1.0, 2.0, float("nan"), 4.0],
            [2.0, 3.0, float("nan"), 5.0],
        ],
        dtype=torch.float64,
    )

    reference_mask = torch.tensor(
        [
            [True, True, False, True],
            [True, True, False, True],
        ],
        dtype=torch.bool,
    )

    scales = torch.ones(
        4,
        dtype=torch.float64,
    )

    with pytest.raises(
        ValueError,
        match="at least one eligible reference",
    ):
        reconstruct_absolute_from_references(
            predicted_delta_z,
            reference_y,
            reference_mask,
            scales,
        )


def test_reference_reconstruction_rejects_nonfinite_eligible_values():
    from p1_pairwise_model import (
        reconstruct_absolute_from_references,
    )

    predicted_delta_z = torch.zeros(
        (1, 1, 4),
        dtype=torch.float64,
    )

    reference_y = torch.tensor(
        [[1.0, float("inf"), 3.0, 4.0]],
        dtype=torch.float64,
    )

    reference_mask = torch.ones(
        (1, 4),
        dtype=torch.bool,
    )

    scales = torch.ones(
        4,
        dtype=torch.float64,
    )

    with pytest.raises(
        ValueError,
        match="eligible reference labels must be finite",
    ):
        reconstruct_absolute_from_references(
            predicted_delta_z,
            reference_y,
            reference_mask,
            scales,
        )
