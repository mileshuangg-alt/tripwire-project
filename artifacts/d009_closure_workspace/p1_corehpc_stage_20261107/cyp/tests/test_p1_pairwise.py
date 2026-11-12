from __future__ import annotations

import math

from p1_pairwise import compute_pairwise_partition_contract


ENDPOINTS = ("E1", "E2", "E3", "E4")


def test_partition_contract_exact_statistics_and_canonical_identity_order():
    molecule_ids = [
        "mol_c",
        "mol_a",
        "mol_e",
        "mol_b",
        "mol_d",
    ]

    labels = [
        [3.0, None, None, None],  # mol_c
        [1.0, 10.0, 7.0, None],  # mol_a
        [5.0, None, None, None],  # mol_e
        [2.0, 20.0, None, None],  # mol_b
        [None, 30.0, None, None], # mol_d
    ]

    result = compute_pairwise_partition_contract(
        molecule_ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=(3.0, 20.0, None, None),
        scaler_scales=(2.0, 10.0, None, None),
    )

    assert result.n_molecules == 5

    assert result.molecule_ids == (
        "mol_a",
        "mol_b",
        "mol_c",
        "mol_d",
        "mol_e",
    )

    # Availability must follow molecule identity through canonical reordering.
    assert result.label_availability == (
        (True, True, True, False),    # mol_a
        (True, True, False, False),   # mol_b
        (True, False, False, False),  # mol_c
        (False, True, False, False),  # mol_d
        (True, False, False, False),  # mol_e
    )

    e1, e2, e3, e4 = result.endpoints

    assert e1.n_labeled == 4
    assert e1.pair_supervision_available is True
    assert math.isclose(e1.q_pair_eligible, 0.6)
    assert math.isclose(e1.weight, 4.0 / 3.0)

    assert e2.n_labeled == 3
    assert e2.pair_supervision_available is True
    assert math.isclose(e2.q_pair_eligible, 0.3)
    assert math.isclose(e2.weight, 2.0)

    assert e3.n_labeled == 1
    assert e3.pair_supervision_available is False
    assert e3.q_pair_eligible == 0.0
    assert e3.weight is None
    assert e3.scaler_mean is None
    assert e3.scaler_scale is None

    assert e4.n_labeled == 0
    assert e4.pair_supervision_available is False
    assert e4.q_pair_eligible == 0.0
    assert e4.weight is None
    assert e4.scaler_mean is None
    assert e4.scaler_scale is None

import numpy as np
import pytest


def _valid_contract_inputs():
    molecule_ids = ["mol_a", "mol_b", "mol_c"]

    labels = [
        [1.0, 10.0, None, None],
        [2.0, 20.0, None, None],
        [3.0, None, None, None],
    ]

    return molecule_ids, labels


def test_partition_contract_rejects_n_less_than_two():
    with pytest.raises(
        ValueError,
        match="at least two molecules",
    ):
        compute_pairwise_partition_contract(
            ["mol_a"],
            [[1.0, None, None, None]],
            endpoint_order=ENDPOINTS,
            scaler_means=(0.0, None, None, None),
            scaler_scales=(1.0, None, None, None),
        )


def test_partition_contract_rejects_duplicate_molecule_ids():
    with pytest.raises(
        ValueError,
        match="duplicate molecule identities",
    ):
        compute_pairwise_partition_contract(
            ["mol_a", "mol_a"],
            [
                [1.0, None, None, None],
                [2.0, None, None, None],
            ],
            endpoint_order=ENDPOINTS,
            scaler_means=(0.0, None, None, None),
            scaler_scales=(1.0, None, None, None),
        )


def test_partition_contract_rejects_label_row_count_mismatch():
    with pytest.raises(
        ValueError,
        match="one-to-one",
    ):
        compute_pairwise_partition_contract(
            ["mol_a", "mol_b"],
            [[1.0, None, None, None]],
            endpoint_order=ENDPOINTS,
            scaler_means=(0.0, None, None, None),
            scaler_scales=(1.0, None, None, None),
        )


def test_partition_contract_rejects_label_endpoint_width_mismatch():
    with pytest.raises(
        ValueError,
        match="align exactly with endpoint order",
    ):
        compute_pairwise_partition_contract(
            ["mol_a", "mol_b"],
            [
                [1.0, None, None],
                [2.0, None, None],
            ],
            endpoint_order=ENDPOINTS,
            scaler_means=(0.0, None, None, None),
            scaler_scales=(1.0, None, None, None),
        )


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_partition_contract_rejects_nonfinite_observed_label(bad_value):
    molecule_ids, labels = _valid_contract_inputs()
    labels[0][0] = bad_value

    with pytest.raises(
        ValueError,
        match="numeric and finite",
    ):
        compute_pairwise_partition_contract(
            molecule_ids,
            labels,
            endpoint_order=ENDPOINTS,
            scaler_means=(0.0, 0.0, None, None),
            scaler_scales=(1.0, 1.0, None, None),
        )


def test_partition_contract_preserves_nan_as_missing():
    molecule_ids, labels = _valid_contract_inputs()
    labels[2][0] = np.nan

    result = compute_pairwise_partition_contract(
        molecule_ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, 0.0, None, None),
        scaler_scales=(1.0, 1.0, None, None),
    )

    e1 = result.endpoints[0]

    assert e1.n_labeled == 2
    assert e1.pair_supervision_available is True
    assert result.label_availability[2][0] is False


@pytest.mark.parametrize(
    ("mean", "scale"),
    [
        (None, 1.0),
        (np.inf, 1.0),
        (0.0, None),
        (0.0, np.inf),
        (0.0, 0.0),
        (0.0, -1.0),
    ],
)
def test_partition_contract_rejects_invalid_supervised_scaler(
    mean,
    scale,
):
    molecule_ids, labels = _valid_contract_inputs()

    with pytest.raises(ValueError, match="scaler"):
        compute_pairwise_partition_contract(
            molecule_ids,
            labels,
            endpoint_order=ENDPOINTS,
            scaler_means=(mean, 0.0, None, None),
            scaler_scales=(scale, 1.0, None, None),
        )


def test_partition_contract_does_not_require_scaler_when_n_less_than_two():
    result = compute_pairwise_partition_contract(
        ["mol_a", "mol_b", "mol_c"],
        [
            [1.0, 10.0, None, None],
            [2.0, None, None, None],
            [3.0, None, None, None],
        ],
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, None, None, None),
        scaler_scales=(1.0, None, None, None),
    )

    e2 = result.endpoints[1]

    assert e2.n_labeled == 1
    assert e2.pair_supervision_available is False
    assert e2.q_pair_eligible == 0.0
    assert e2.weight is None
    assert e2.scaler_mean is None
    assert e2.scaler_scale is None


def test_partition_contract_rejects_partition_with_no_supervised_distinct_pair():
    with pytest.raises(
        ValueError,
        match="no possible supervised distinct pair",
    ):
        compute_pairwise_partition_contract(
            ["mol_a", "mol_b", "mol_c", "mol_d"],
            [
                [1.0, None, None, None],
                [None, 2.0, None, None],
                [None, None, None, None],
                [None, None, None, 4.0],
            ],
            endpoint_order=ENDPOINTS,
            scaler_means=(None, None, None, None),
            scaler_scales=(None, None, None, None),
        )


def test_partition_contract_is_invariant_to_input_row_order():
    ids_a = ["mol_c", "mol_a", "mol_d", "mol_b"]
    labels_a = [
        [3.0, None, 30.0, None],  # mol_c
        [1.0, 10.0, None, None],  # mol_a
        [4.0, 40.0, None, None],  # mol_d
        [2.0, 20.0, 20.0, None],  # mol_b
    ]

    ids_b = ["mol_b", "mol_d", "mol_a", "mol_c"]
    labels_b = [
        [2.0, 20.0, 20.0, None],  # mol_b
        [4.0, 40.0, None, None],  # mol_d
        [1.0, 10.0, None, None],  # mol_a
        [3.0, None, 30.0, None],  # mol_c
    ]

    kwargs = {
        "endpoint_order": ENDPOINTS,
        "scaler_means": (2.5, 25.0, 25.0, None),
        "scaler_scales": (1.0, 10.0, 5.0, None),
    }

    contract_a = compute_pairwise_partition_contract(
        ids_a,
        labels_a,
        **kwargs,
    )
    contract_b = compute_pairwise_partition_contract(
        ids_b,
        labels_b,
        **kwargs,
    )

    assert contract_a == contract_b


def test_partition_contract_binds_availability_to_canonical_ids_and_endpoints():
    result = compute_pairwise_partition_contract(
        ["mol_b", "mol_a", "mol_c"],
        [
            [2.0, None, 20.0, None],  # mol_b
            [1.0, 10.0, None, None],  # mol_a
            [3.0, 30.0, 30.0, None],  # mol_c
        ],
        endpoint_order=ENDPOINTS,
        scaler_means=(2.0, 20.0, 25.0, None),
        scaler_scales=(1.0, 10.0, 5.0, None),
    )

    assert result.endpoint_order == ENDPOINTS

    assert result.molecule_ids == (
        "mol_a",
        "mol_b",
        "mol_c",
    )

    assert result.label_availability == (
        (True, True, False, False),   # mol_a
        (True, False, True, False),   # mol_b
        (True, True, True, False),    # mol_c
    )


@pytest.mark.parametrize(
    ("N", "n_t"),
    [
        (2, 2),
        (3, 2),
        (3, 3),
        (5, 2),
        (5, 4),
        (10, 3),
        (10, 9),
        (25, 17),
    ],
)
def test_partition_contract_weight_identity_across_population_sizes(
    N,
    n_t,
):
    molecule_ids = [
        f"mol_{index:03d}"
        for index in range(N)
    ]

    labels = []

    for index in range(N):
        labels.append(
            [
                float(index) if index < n_t else None,
                None,
                None,
                None,
            ]
        )

    result = compute_pairwise_partition_contract(
        molecule_ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, None, None, None),
        scaler_scales=(1.0, None, None, None),
    )

    endpoint = result.endpoints[0]

    expected_q = (
        n_t * (n_t - 1)
        / (N * (N - 1))
    )
    expected_w = (N - 1) / (n_t - 1)

    assert math.isclose(
        endpoint.q_pair_eligible,
        expected_q,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )

    assert math.isclose(
        endpoint.weight,
        expected_w,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )

    assert math.isclose(
        endpoint.q_pair_eligible * endpoint.weight,
        n_t / N,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


# ---------------------------------------------------------------------------
# D009 accepted label-value binding qualification
# ---------------------------------------------------------------------------


def test_partition_contract_label_binding_changes_with_observed_value():
    molecule_ids, labels = _valid_contract_inputs()

    original = compute_pairwise_partition_contract(
        molecule_ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, 0.0, None, None),
        scaler_scales=(1.0, 1.0, None, None),
    )

    changed_labels = [row.copy() for row in labels]
    changed_labels[0][0] = 1.5

    changed = compute_pairwise_partition_contract(
        molecule_ids,
        changed_labels,
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, 0.0, None, None),
        scaler_scales=(1.0, 1.0, None, None),
    )

    assert original.label_values_sha256 != changed.label_values_sha256


def test_partition_contract_label_binding_is_row_order_invariant():
    ids_a = ["mol_c", "mol_a", "mol_b"]
    labels_a = [
        [3.0, None, None, None],
        [1.0, 10.0, None, None],
        [2.0, 20.0, None, None],
    ]

    ids_b = ["mol_b", "mol_c", "mol_a"]
    labels_b = [
        [2.0, 20.0, None, None],
        [3.0, None, None, None],
        [1.0, 10.0, None, None],
    ]

    kwargs = {
        "endpoint_order": ENDPOINTS,
        "scaler_means": (0.0, 0.0, None, None),
        "scaler_scales": (1.0, 1.0, None, None),
    }

    first = compute_pairwise_partition_contract(
        ids_a,
        labels_a,
        **kwargs,
    )
    second = compute_pairwise_partition_contract(
        ids_b,
        labels_b,
        **kwargs,
    )

    assert first.label_values_sha256 == second.label_values_sha256
    assert first == second


def test_partition_contract_label_binding_changes_with_missingness():
    molecule_ids, labels = _valid_contract_inputs()

    original = compute_pairwise_partition_contract(
        molecule_ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, 0.0, None, None),
        scaler_scales=(1.0, 1.0, None, None),
    )

    changed_labels = [row.copy() for row in labels]
    changed_labels[2][0] = None

    changed = compute_pairwise_partition_contract(
        molecule_ids,
        changed_labels,
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, 0.0, None, None),
        scaler_scales=(1.0, 1.0, None, None),
    )

    assert original.label_values_sha256 != changed.label_values_sha256
    assert original.label_availability != changed.label_availability


def test_partition_contract_label_binding_normalizes_signed_zero():
    positive = compute_pairwise_partition_contract(
        ["mol_a", "mol_b"],
        [
            [0.0, None, None, None],
            [1.0, None, None, None],
        ],
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, None, None, None),
        scaler_scales=(1.0, None, None, None),
    )

    negative = compute_pairwise_partition_contract(
        ["mol_a", "mol_b"],
        [
            [-0.0, None, None, None],
            [1.0, None, None, None],
        ],
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, None, None, None),
        scaler_scales=(1.0, None, None, None),
    )

    assert positive.label_values_sha256 == negative.label_values_sha256


def test_partition_contract_label_binding_uses_normalized_binary64_values():
    numeric_strings = compute_pairwise_partition_contract(
        ["mol_a", "mol_b"],
        [
            ["1.0", None, None, None],
            ["2.500", None, None, None],
        ],
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, None, None, None),
        scaler_scales=(1.0, None, None, None),
    )

    numeric_values = compute_pairwise_partition_contract(
        ["mol_a", "mol_b"],
        [
            [1, None, None, None],
            [2.5, None, None, None],
        ],
        endpoint_order=ENDPOINTS,
        scaler_means=(0.0, None, None, None),
        scaler_scales=(1.0, None, None, None),
    )

    assert (
        numeric_strings.label_values_sha256
        == numeric_values.label_values_sha256
    )


# ---------------------------------------------------------------------------
# D009 accepted sampler qualification
# ---------------------------------------------------------------------------

import p1_pairwise

from p1_pairwise import (
    _ordered_distinct_pair_from_index,
    sample_supervised_pairs,
)


def _sampler_contract_inputs():
    ids = ["mol_c", "mol_a", "mol_b"]

    labels = [
        [3.0, None, 30.0, None],
        [1.0, 10.0, None, None],
        [2.0, 20.0, 20.0, None],
    ]

    means = (2.0, 15.0, 25.0, None)
    scales = (1.0, 10.0, 5.0, None)

    contract = compute_pairwise_partition_contract(
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
    )

    return contract, ids, labels, means, scales


def _patch_pair_index_sequence(monkeypatch, values):
    iterator = iter(values)

    def draw(rng, pair_count):
        value = next(iterator)
        assert 0 <= value < pair_count
        return value

    monkeypatch.setattr(
        p1_pairwise,
        "_draw_ordered_pair_index",
        draw,
    )


def test_ordered_distinct_index_mapping_is_exact_bijection_for_n4():
    observed = [
        _ordered_distinct_pair_from_index(k, 4)
        for k in range(12)
    ]

    expected = [
        (i, j)
        for i in range(4)
        for j in range(4)
        if i != j
    ]

    assert len(observed) == 12
    assert len(set(observed)) == 12
    assert set(observed) == set(expected)
    assert all(i != j for i, j in observed)


def test_sampler_is_row_order_invariant_with_same_generator_state():
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    ids2 = ["mol_b", "mol_c", "mol_a"]
    labels2 = [labels[2], labels[0], labels[1]]

    rng1 = np.random.default_rng(20261107)
    rng2 = np.random.default_rng(20261107)

    first = sample_supervised_pairs(
        contract,
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
        accepted_pair_count=8,
        rng=rng1,
    )

    second = sample_supervised_pairs(
        contract,
        ids2,
        labels2,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
        accepted_pair_count=8,
        rng=rng2,
    )

    assert first == second


@pytest.mark.parametrize(
    "bad_count",
    [True, False, 0, -1, 1.5, "2"],
)
def test_sampler_rejects_invalid_accepted_pair_count_before_rng(
    bad_count,
):
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    rng = np.random.default_rng(7)
    before = rng.bit_generator.state

    with pytest.raises((ValueError, TypeError)):
        sample_supervised_pairs(
            contract,
            ids,
            labels,
            endpoint_order=ENDPOINTS,
            scaler_means=means,
            scaler_scales=scales,
            accepted_pair_count=bad_count,
            rng=rng,
        )

    assert rng.bit_generator.state == before


def test_sampler_rejects_id_mismatch_before_rng():
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    rng = np.random.default_rng(7)
    before = rng.bit_generator.state

    bad_ids = ids.copy()
    bad_ids[0] = "other"

    with pytest.raises(ValueError, match="do not match"):
        sample_supervised_pairs(
            contract,
            bad_ids,
            labels,
            endpoint_order=ENDPOINTS,
            scaler_means=means,
            scaler_scales=scales,
            accepted_pair_count=1,
            rng=rng,
        )

    assert rng.bit_generator.state == before


def test_sampler_rejects_availability_mismatch_before_rng():
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    rng = np.random.default_rng(7)
    before = rng.bit_generator.state

    changed = [row.copy() for row in labels]
    changed[0][1] = 15.0

    with pytest.raises(ValueError, match="do not match"):
        sample_supervised_pairs(
            contract,
            ids,
            changed,
            endpoint_order=ENDPOINTS,
            scaler_means=means,
            scaler_scales=scales,
            accepted_pair_count=1,
            rng=rng,
        )

    assert rng.bit_generator.state == before


def test_sampler_rejects_label_value_mismatch_before_rng():
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    rng = np.random.default_rng(7)
    before = rng.bit_generator.state

    changed = [row.copy() for row in labels]
    changed[0][0] = 3.5

    with pytest.raises(ValueError, match="do not match"):
        sample_supervised_pairs(
            contract,
            ids,
            changed,
            endpoint_order=ENDPOINTS,
            scaler_means=means,
            scaler_scales=scales,
            accepted_pair_count=1,
            rng=rng,
        )

    assert rng.bit_generator.state == before


def test_sampler_rejects_endpoint_order_mismatch_before_rng():
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    rng = np.random.default_rng(7)
    before = rng.bit_generator.state

    bad_order = ("E2", "E1", "E3", "E4")

    with pytest.raises(ValueError, match="do not match"):
        sample_supervised_pairs(
            contract,
            ids,
            labels,
            endpoint_order=bad_order,
            scaler_means=means,
            scaler_scales=scales,
            accepted_pair_count=1,
            rng=rng,
        )

    assert rng.bit_generator.state == before


def test_sampler_rejects_scaler_mismatch_before_rng():
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    rng = np.random.default_rng(7)
    before = rng.bit_generator.state

    bad_scales = (2.0, 10.0, 5.0, None)

    with pytest.raises(ValueError, match="do not match"):
        sample_supervised_pairs(
            contract,
            ids,
            labels,
            endpoint_order=ENDPOINTS,
            scaler_means=means,
            scaler_scales=bad_scales,
            accepted_pair_count=1,
            rng=rng,
        )

    assert rng.bit_generator.state == before


def test_sampler_exact_mask_delta_orientation_and_nan_targets(
    monkeypatch,
):
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    _patch_pair_index_sequence(
        monkeypatch,
        [1, 4],
    )

    batch = sample_supervised_pairs(
        contract,
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
        accepted_pair_count=2,
        rng=np.random.default_rng(1),
    )

    assert batch.molecule_a_ids == ("mol_a", "mol_c")
    assert batch.molecule_b_ids == ("mol_c", "mol_a")

    assert batch.eligibility_masks == (
        (True, False, False, False),
        (True, False, False, False),
    )

    assert batch.delta_targets[0][0] == -2.0
    assert batch.delta_targets[1][0] == 2.0

    assert np.isnan(batch.delta_targets[0][1])
    assert np.isnan(batch.delta_targets[0][2])
    assert np.isnan(batch.delta_targets[0][3])

    assert batch.eligible_target_count == (2, 0, 0, 0)


def test_sampler_no_supervision_rejection_and_accounting(
    monkeypatch,
):
    ids = ["a", "b", "c"]

    labels = [
        [1.0, None, None, None],
        [None, 2.0, None, None],
        [3.0, 4.0, None, None],
    ]

    means = (0.0, 0.0, None, None)
    scales = (1.0, 1.0, None, None)

    contract = compute_pairwise_partition_contract(
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
    )

    _patch_pair_index_sequence(
        monkeypatch,
        [0, 1],
    )

    batch = sample_supervised_pairs(
        contract,
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
        accepted_pair_count=1,
        rng=np.random.default_rng(1),
    )

    assert batch.attempts == 2
    assert batch.no_supervision_rejections == 1
    assert batch.accepted_count == 1

    assert (
        batch.attempts
        == batch.accepted_count
        + batch.no_supervision_rejections
    )

    assert batch.eligible_target_count == (1, 0, 0, 0)


def test_sampler_returns_exact_requested_accepted_batch_and_allows_repeats(
    monkeypatch,
):
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    _patch_pair_index_sequence(
        monkeypatch,
        [0, 0, 0],
    )

    batch = sample_supervised_pairs(
        contract,
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
        accepted_pair_count=3,
        rng=np.random.default_rng(1),
    )

    assert batch.accepted_count == 3
    assert batch.molecule_a_ids == ("mol_a", "mol_a", "mol_a")
    assert batch.molecule_b_ids == ("mol_b", "mol_b", "mol_b")
    assert batch.attempts == 3
    assert batch.no_supervision_rejections == 0


def test_sampler_is_deterministic_from_same_generator_state():
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)

    first = sample_supervised_pairs(
        contract,
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
        accepted_pair_count=12,
        rng=rng1,
    )

    second = sample_supervised_pairs(
        contract,
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
        accepted_pair_count=12,
        rng=rng2,
    )

    assert first == second


def test_sampler_rejects_non_generator_before_sampling():
    contract, ids, labels, means, scales = _sampler_contract_inputs()

    with pytest.raises(TypeError, match="numpy.random.Generator"):
        sample_supervised_pairs(
            contract,
            ids,
            labels,
            endpoint_order=ENDPOINTS,
            scaler_means=means,
            scaler_scales=scales,
            accepted_pair_count=1,
            rng=np.random.RandomState(1),
        )


def test_sampler_fails_on_nonfinite_eligible_delta_not_rejection(
    monkeypatch,
):
    ids = ["a", "b"]

    labels = [
        [1.0e308, None, None, None],
        [-1.0e308, None, None, None],
    ]

    means = (0.0, None, None, None)
    scales = (1.0, None, None, None)

    contract = compute_pairwise_partition_contract(
        ids,
        labels,
        endpoint_order=ENDPOINTS,
        scaler_means=means,
        scaler_scales=scales,
    )

    _patch_pair_index_sequence(
        monkeypatch,
        [0],
    )

    with np.errstate(over="ignore"):
        with pytest.raises(
            FloatingPointError,
            match="standardized delta target is non-finite",
        ):
            sample_supervised_pairs(
                contract,
                ids,
                labels,
                endpoint_order=ENDPOINTS,
                scaler_means=means,
                scaler_scales=scales,
                accepted_pair_count=1,
                rng=np.random.default_rng(1),
            )
