from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch import nn

from p1_pairwise_benchmark import (
    DEVELOPMENT_CELL_COUNT,
    compare_chunked_pair_head_outputs,
    run_p1_all_reference_benchmark,
)
from p1_pairwise_model import P1AntisymmetricPairHead


class TrackingEncoder:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.calls = []

    def __call__(self, key):
        self.calls.append(key)
        return self.embeddings[key].clone()


class ZeroPairFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.directional_rows = []
        self.batch_sizes = []

    def forward(self, pair_input):
        if pair_input.ndim != 2 or pair_input.shape[1] != 1200:
            raise ValueError("pair input must have shape (rows, 1200)")

        self.batch_sizes.append(int(pair_input.shape[0]))
        self.directional_rows.extend(
            (
                float(row[0].item()),
                float(row[600].item()),
            )
            for row in pair_input
        )
        return torch.zeros(
            (pair_input.shape[0], 4),
            dtype=pair_input.dtype,
            device=pair_input.device,
        )


class NonfinitePairFunction(nn.Module):
    def forward(self, pair_input):
        result = torch.zeros(
            (pair_input.shape[0], 4),
            dtype=pair_input.dtype,
            device=pair_input.device,
        )
        result[0, 0] = float("nan")
        return result


def _p1_head_with_pair_function(pair_function):
    pair_head = P1AntisymmetricPairHead()
    pair_head.pair_function = pair_function
    return pair_head


def _zero_pair_head():
    pair_function = ZeroPairFunction()
    return _p1_head_with_pair_function(pair_function), pair_function


def _synthetic_inputs():
    query_embeddings = torch.zeros((3, 600), dtype=torch.float64)
    reference_embeddings = torch.zeros((4, 600), dtype=torch.float64)

    query_embeddings[:, 0] = torch.tensor(
        [10.0, 20.0, 30.0],
        dtype=torch.float64,
    )
    reference_embeddings[:, 0] = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
    )

    reference_y = torch.tensor(
        [
            [1.0, 10.0, float("nan"), 1000.0],
            [3.0, float("nan"), 30.0, 3000.0],
            [5.0, 50.0, 50.0, float("nan")],
            [999.0, 70.0, 70.0, 7000.0],
        ],
        dtype=torch.float64,
    )

    reference_mask = torch.tensor(
        [
            [True, True, False, True],
            [True, False, True, True],
            [True, True, True, False],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )

    scaler_scales = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
    )

    encoder = TrackingEncoder(
        {
            "queries": query_embeddings,
            "references": reference_embeddings,
        }
    )
    return encoder, reference_y, reference_mask, scaler_scales


def test_benchmark_caches_embeddings_within_stage_b_and_reruns_stage_d():
    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    pair_head, _ = _zero_pair_head()

    result = run_p1_all_reference_benchmark(
        encoder=encoder,
        pair_head=pair_head,
        query_inputs="queries",
        reference_inputs="references",
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        chunk_size=5,
        validation_evaluations_per_cell=2,
        warmup_iterations=0,
    )

    assert encoder.calls == [
        "queries",
        "references",
        "queries",
        "references",
    ]
    assert result.timings.stage_d_end_to_end_seconds >= 0.0


def test_embedding_cache_is_local_to_one_benchmark_execution():
    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    pair_head, _ = _zero_pair_head()

    kwargs = {
        "encoder": encoder,
        "pair_head": pair_head,
        "query_inputs": "queries",
        "reference_inputs": "references",
        "reference_y": reference_y,
        "reference_mask": reference_mask,
        "scaler_scales": scaler_scales,
        "chunk_size": 4,
        "validation_evaluations_per_cell": 1,
        "warmup_iterations": 0,
    }

    run_p1_all_reference_benchmark(**kwargs)
    run_p1_all_reference_benchmark(**kwargs)

    assert encoder.calls == [
        "queries",
        "references",
        "queries",
        "references",
        "queries",
        "references",
        "queries",
        "references",
    ]


def test_chunk_coverage_pair_accounting_and_endpoint_reference_counts():
    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    pair_head, pair_function = _zero_pair_head()

    result = run_p1_all_reference_benchmark(
        encoder=encoder,
        pair_head=pair_head,
        query_inputs="queries",
        reference_inputs="references",
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        chunk_size=5,
        validation_evaluations_per_cell=1,
        warmup_iterations=0,
    )

    assert result.accounting.query_count == 3
    assert result.accounting.reference_count == 4
    assert result.accounting.unique_query_reference_pairs == 12
    assert result.accounting.directional_g_evaluations == 24
    assert result.accounting.endpoint_eligible_reference_counts == (3, 3, 3, 3)
    assert result.accounting.chunk_size == 5
    assert result.accounting.chunk_ranges == ((0, 5), (5, 10), (10, 12))

    covered = []
    for start, stop in result.accounting.chunk_ranges:
        covered.extend(range(start, stop))
    assert covered == list(range(12))

    expected_stage_b_and_d_pairs = [
        (10.0, 1.0),
        (10.0, 2.0),
        (10.0, 3.0),
        (10.0, 4.0),
        (20.0, 1.0),
        (20.0, 2.0),
        (20.0, 3.0),
        (20.0, 4.0),
        (30.0, 1.0),
        (30.0, 2.0),
        (30.0, 3.0),
        (30.0, 4.0),
    ]
    expected_directional_rows = []
    for start, stop in result.accounting.chunk_ranges:
        chunk = expected_stage_b_and_d_pairs[start:stop]
        expected_directional_rows.extend(chunk)
        expected_directional_rows.extend(
            (right, left)
            for left, right in chunk
        )

    assert pair_function.directional_rows == expected_directional_rows * 2


def test_reconstruction_uses_all_endpoint_eligible_references_and_is_finite():
    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    pair_head, _ = _zero_pair_head()

    result = run_p1_all_reference_benchmark(
        encoder=encoder,
        pair_head=pair_head,
        query_inputs="queries",
        reference_inputs="references",
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        chunk_size=6,
        validation_evaluations_per_cell=1,
        warmup_iterations=0,
    )

    expected = torch.tensor(
        [[3.0, 130.0 / 3.0, 50.0, 11000.0 / 3.0]],
        dtype=torch.float64,
    ).expand(3, 4)

    torch.testing.assert_close(
        result.absolute_mean.cpu(),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.stage_d_absolute_mean.cpu(),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(result.absolute_mean).all()
    assert torch.isfinite(result.reference_dispersion).all()
    assert torch.isfinite(result.stage_d_absolute_mean).all()
    assert torch.isfinite(result.stage_d_reference_dispersion).all()


def test_chunked_vs_unchunked_real_pair_head_reports_observed_difference():
    torch.manual_seed(20261107)
    pair_head = P1AntisymmetricPairHead()
    pair_head.eval()

    query_embeddings = torch.randn(3, 600)
    reference_embeddings = torch.randn(4, 600)

    comparison = compare_chunked_pair_head_outputs(
        pair_head,
        query_embeddings,
        reference_embeddings,
        chunk_size=5,
    )

    assert math.isfinite(comparison.max_abs_difference)
    assert comparison.max_abs_difference >= 0.0
    assert comparison.query_count == 3
    assert comparison.reference_count == 4
    assert isinstance(comparison.bitwise_equal, bool)
    print(
        "chunked_vs_unchunked_max_abs_difference=",
        comparison.max_abs_difference,
        "bitwise_equal=",
        comparison.bitwise_equal,
    )


def test_real_p1_head_processes_two_directional_g_rows_per_unique_pair():
    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    pair_head, pair_function = _zero_pair_head()

    result = run_p1_all_reference_benchmark(
        encoder=encoder,
        pair_head=pair_head,
        query_inputs="queries",
        reference_inputs="references",
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        chunk_size=None,
        validation_evaluations_per_cell=1,
        warmup_iterations=0,
    )

    unique_pairs = result.accounting.unique_query_reference_pairs
    directional_rows_per_traversal = 2 * unique_pairs

    assert unique_pairs == 12
    assert result.accounting.directional_g_evaluations == 24
    assert pair_function.batch_sizes == [12, 12, 12, 12]

    expected_unique_pairs = [
        (10.0, 1.0),
        (10.0, 2.0),
        (10.0, 3.0),
        (10.0, 4.0),
        (20.0, 1.0),
        (20.0, 2.0),
        (20.0, 3.0),
        (20.0, 4.0),
        (30.0, 1.0),
        (30.0, 2.0),
        (30.0, 3.0),
        (30.0, 4.0),
    ]
    expected_directional_rows = expected_unique_pairs + [
        (right, left)
        for left, right in expected_unique_pairs
    ]

    assert pair_function.directional_rows[:24] == expected_directional_rows
    assert pair_function.directional_rows[24:] == expected_directional_rows
    assert len(pair_function.directional_rows[:24]) == directional_rows_per_traversal
    assert len(pair_function.directional_rows[24:]) == directional_rows_per_traversal
    print(
        "unique_query_reference_pairs=",
        unique_pairs,
        "directional_g_rows_per_traversal=",
        directional_rows_per_traversal,
    )


def test_timing_fields_are_finite_nonnegative_and_document_exclusions():
    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    pair_head, _ = _zero_pair_head()

    result = run_p1_all_reference_benchmark(
        encoder=encoder,
        pair_head=pair_head,
        query_inputs="queries",
        reference_inputs="references",
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        chunk_size=5,
        validation_evaluations_per_cell=1,
        warmup_iterations=1,
    )

    timings = result.timings
    for value in (
        timings.stage_a_encode_cache_seconds,
        timings.stage_b_pair_head_seconds,
        timings.stage_c_reconstruction_seconds,
        timings.stage_d_end_to_end_seconds,
    ):
        assert math.isfinite(value)
        assert value >= 0.0

    assert "warmup executions" in result.timing_excludes
    assert "validation-label scoring or performance metrics" in result.timing_excludes
    assert result.peak_accelerator_memory_bytes is None


def test_projection_requires_explicit_validation_evaluations_per_cell_arithmetic():
    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    pair_head, _ = _zero_pair_head()

    result = run_p1_all_reference_benchmark(
        encoder=encoder,
        pair_head=pair_head,
        query_inputs="queries",
        reference_inputs="references",
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        chunk_size=5,
        validation_evaluations_per_cell=7,
        warmup_iterations=0,
    )

    assert result.projection.development_cell_count == DEVELOPMENT_CELL_COUNT
    assert result.projection.validation_evaluations_per_cell == 7
    assert result.projection.projected_stage_d_seconds == pytest.approx(
        result.timings.stage_d_end_to_end_seconds * 7 * DEVELOPMENT_CELL_COUNT
    )
    assert result.projection.unresolved_reason is None

    unresolved = run_p1_all_reference_benchmark(
        encoder=encoder,
        pair_head=pair_head,
        query_inputs="queries",
        reference_inputs="references",
        reference_y=reference_y,
        reference_mask=reference_mask,
        scaler_scales=scaler_scales,
        chunk_size=5,
        validation_evaluations_per_cell=None,
        warmup_iterations=0,
    )

    assert unresolved.projection.projected_stage_d_seconds is None
    assert unresolved.projection.unresolved_reason is not None


def test_public_api_has_no_forbidden_validation_label_or_metric_arguments():
    signature = inspect.signature(run_p1_all_reference_benchmark)
    parameter_names = set(signature.parameters)

    assert "validation_labels" not in parameter_names
    assert "protected_internal_test_labels" not in parameter_names
    assert "d008_labels" not in parameter_names
    assert "st_rae" not in parameter_names
    assert "performance_metric" not in parameter_names
    assert "metric" not in parameter_names


def test_fail_closed_on_nonfinite_embeddings_pair_outputs_and_accounting():
    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    encoder.embeddings["queries"][0, 0] = float("nan")
    pair_head, _ = _zero_pair_head()

    with pytest.raises(ValueError, match="embeddings must be finite"):
        run_p1_all_reference_benchmark(
            encoder=encoder,
            pair_head=pair_head,
            query_inputs="queries",
            reference_inputs="references",
            reference_y=reference_y,
            reference_mask=reference_mask,
            scaler_scales=scaler_scales,
            chunk_size=5,
            validation_evaluations_per_cell=1,
            warmup_iterations=0,
        )

    encoder, reference_y, reference_mask, scaler_scales = _synthetic_inputs()
    nonfinite_pair_head = _p1_head_with_pair_function(NonfinitePairFunction())
    with pytest.raises(FloatingPointError, match="pair-head output is non-finite"):
        run_p1_all_reference_benchmark(
            encoder=encoder,
            pair_head=nonfinite_pair_head,
            query_inputs="queries",
            reference_inputs="references",
            reference_y=reference_y,
            reference_mask=reference_mask,
            scaler_scales=scaler_scales,
            chunk_size=5,
            validation_evaluations_per_cell=1,
            warmup_iterations=0,
        )

    reference_mask[:, 0] = False
    pair_head, _ = _zero_pair_head()
    with pytest.raises(ValueError, match="at least one eligible reference"):
        run_p1_all_reference_benchmark(
            encoder=encoder,
            pair_head=pair_head,
            query_inputs="queries",
            reference_inputs="references",
            reference_y=reference_y,
            reference_mask=reference_mask,
            scaler_scales=scaler_scales,
            chunk_size=5,
            validation_evaluations_per_cell=1,
            warmup_iterations=0,
        )
