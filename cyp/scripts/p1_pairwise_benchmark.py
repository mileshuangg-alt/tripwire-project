from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import Tensor

from p1_pairwise_model import (
    P1_N_TASKS,
    P1AntisymmetricPairHead,
    reconstruct_absolute_from_references,
)


DEVELOPMENT_CELL_COUNT = 150


@dataclass(frozen=True)
class P1BenchmarkTimings:
    stage_a_encode_cache_seconds: float
    stage_b_pair_head_seconds: float
    stage_c_reconstruction_seconds: float
    stage_d_end_to_end_seconds: float


@dataclass(frozen=True)
class P1PairAccounting:
    query_count: int
    reference_count: int
    unique_query_reference_pairs: int
    directional_g_evaluations: int
    endpoint_eligible_reference_counts: tuple[int, int, int, int]
    chunk_size: int | None
    chunk_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class P1Projection:
    development_cell_count: int
    validation_evaluations_per_cell: int | None
    projected_stage_d_seconds: float | None
    unresolved_reason: str | None


@dataclass(frozen=True)
class P1ChunkComparison:
    max_abs_difference: float
    bitwise_equal: bool
    chunk_size: int | None
    query_count: int
    reference_count: int


@dataclass(frozen=True)
class P1AllReferenceBenchmarkResult:
    timings: P1BenchmarkTimings
    accounting: P1PairAccounting
    projection: P1Projection
    absolute_mean: Tensor
    reference_dispersion: Tensor
    stage_d_absolute_mean: Tensor
    stage_d_reference_dispersion: Tensor
    peak_accelerator_memory_bytes: int | None
    timing_excludes: tuple[str, ...]


def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(
    device: torch.device,
    fn: Callable[[], Any],
) -> tuple[Any, float]:
    _synchronize_if_cuda(device)
    start = time.perf_counter()
    result = fn()
    _synchronize_if_cuda(device)
    elapsed = time.perf_counter() - start

    if not torch.isfinite(torch.tensor(elapsed)) or elapsed < 0.0:
        raise FloatingPointError("Measured timing must be finite and nonnegative.")

    return result, float(elapsed)


def _validate_reference_inputs(
    reference_y: Tensor,
    reference_mask: Tensor,
    scaler_scales: Tensor,
) -> None:
    if reference_y.ndim != 2 or reference_y.shape[1] != P1_N_TASKS:
        raise ValueError("reference_y must have shape (references, 4).")

    if reference_mask.shape != reference_y.shape:
        raise ValueError("reference_mask must match reference_y shape.")

    if reference_mask.dtype is not torch.bool:
        raise TypeError("reference_mask must be boolean.")

    if scaler_scales.shape != (P1_N_TASKS,):
        raise ValueError("scaler_scales must have shape (4,).")

    if not reference_mask.any(dim=0).all():
        raise ValueError("every endpoint requires at least one eligible reference.")

    if not torch.isfinite(reference_y[reference_mask]).all():
        raise ValueError("eligible reference labels must be finite.")

    if not torch.isfinite(scaler_scales).all() or (scaler_scales <= 0).any():
        raise ValueError("scaler scales must be finite and positive.")


def _validate_embeddings(
    query_embeddings: Tensor,
    reference_embeddings: Tensor,
) -> None:
    if query_embeddings.ndim != 2 or reference_embeddings.ndim != 2:
        raise ValueError("query and reference embeddings must be rank-2 tensors.")

    if query_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("query and reference embedding widths must match.")

    if query_embeddings.shape[0] <= 0:
        raise ValueError("at least one query embedding is required.")

    if reference_embeddings.shape[0] <= 0:
        raise ValueError("at least one reference embedding is required.")

    if not torch.isfinite(query_embeddings).all():
        raise ValueError("query embeddings must be finite.")

    if not torch.isfinite(reference_embeddings).all():
        raise ValueError("reference embeddings must be finite.")


def _validate_chunk_size(chunk_size: int | None) -> int | None:
    if chunk_size is None:
        return None

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a positive integer or None.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer or None.")

    return int(chunk_size)


def _validate_pair_head(pair_head: P1AntisymmetricPairHead) -> None:
    if not isinstance(pair_head, P1AntisymmetricPairHead):
        raise TypeError(
            "pair_head must be a qualified P1AntisymmetricPairHead."
        )


def _encode_cache_embeddings(
    encoder: Callable[[Any], Tensor],
    query_inputs: Any,
    reference_inputs: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    query_embeddings = encoder(query_inputs).to(device)
    reference_embeddings = encoder(reference_inputs).to(device)
    _validate_embeddings(query_embeddings, reference_embeddings)
    return query_embeddings, reference_embeddings


def _chunk_ranges(
    total_pairs: int,
    chunk_size: int | None,
) -> tuple[tuple[int, int], ...]:
    if total_pairs <= 0:
        raise ValueError("total pair count must be positive.")

    if chunk_size is None:
        return ((0, total_pairs),)

    ranges = []
    for start in range(0, total_pairs, chunk_size):
        stop = min(start + chunk_size, total_pairs)
        ranges.append((start, stop))
    return tuple(ranges)


def _all_reference_pair_head(
    pair_head: P1AntisymmetricPairHead,
    query_embeddings: Tensor,
    reference_embeddings: Tensor,
    chunk_size: int | None,
) -> tuple[Tensor, tuple[tuple[int, int], ...]]:
    _validate_embeddings(query_embeddings, reference_embeddings)
    chunk_size = _validate_chunk_size(chunk_size)

    query_count = query_embeddings.shape[0]
    reference_count = reference_embeddings.shape[0]
    total_pairs = query_count * reference_count
    ranges = _chunk_ranges(total_pairs, chunk_size)

    chunks = []
    for start, stop in ranges:
        flat_indices = torch.arange(
            start,
            stop,
            device=query_embeddings.device,
        )
        query_indices = torch.div(
            flat_indices,
            reference_count,
            rounding_mode="floor",
        )
        reference_indices = flat_indices.remainder(reference_count)

        chunk = pair_head(
            query_embeddings.index_select(0, query_indices),
            reference_embeddings.index_select(0, reference_indices),
        )

        if chunk.shape != (stop - start, P1_N_TASKS):
            raise ValueError("pair head must return shape (pairs, 4).")

        if not torch.isfinite(chunk).all():
            raise FloatingPointError("pair-head output is non-finite.")

        chunks.append(chunk)

    predicted_delta_z = torch.cat(chunks, dim=0).reshape(
        query_count,
        reference_count,
        P1_N_TASKS,
    )

    if not torch.isfinite(predicted_delta_z).all():
        raise FloatingPointError("all-reference pair-head output is non-finite.")

    return predicted_delta_z, ranges


def _reconstruct(
    predicted_delta_z: Tensor,
    reference_y: Tensor,
    reference_mask: Tensor,
    scaler_scales: Tensor,
) -> tuple[Tensor, Tensor]:
    absolute_mean, reference_dispersion = reconstruct_absolute_from_references(
        predicted_delta_z,
        reference_y,
        reference_mask,
        scaler_scales,
    )

    if not torch.isfinite(absolute_mean).all():
        raise FloatingPointError("absolute reconstruction output is non-finite.")

    if not torch.isfinite(reference_dispersion).all():
        raise FloatingPointError("reference dispersion output is non-finite.")

    return absolute_mean, reference_dispersion


def _execute_end_to_end(
    encoder: Callable[[Any], Tensor],
    pair_head: P1AntisymmetricPairHead,
    query_inputs: Any,
    reference_inputs: Any,
    reference_y: Tensor,
    reference_mask: Tensor,
    scaler_scales: Tensor,
    chunk_size: int | None,
    device: torch.device,
) -> tuple[Tensor, Tensor, tuple[tuple[int, int], ...]]:
    query_embeddings, reference_embeddings = _encode_cache_embeddings(
        encoder,
        query_inputs,
        reference_inputs,
        device,
    )
    predicted_delta_z, ranges = _all_reference_pair_head(
        pair_head,
        query_embeddings,
        reference_embeddings,
        chunk_size,
    )
    absolute_mean, reference_dispersion = _reconstruct(
        predicted_delta_z,
        reference_y,
        reference_mask,
        scaler_scales,
    )
    return absolute_mean, reference_dispersion, ranges


def compare_chunked_pair_head_outputs(
    pair_head: P1AntisymmetricPairHead,
    query_embeddings: Tensor,
    reference_embeddings: Tensor,
    *,
    chunk_size: int,
) -> P1ChunkComparison:
    """
    Compare deterministic chunked traversal to unchunked traversal.

    The returned maximum absolute difference is descriptive. This helper does
    not manufacture or apply a pass threshold.
    """
    _validate_pair_head(pair_head)
    pair_head.eval()
    chunk_size = _validate_chunk_size(chunk_size)

    with torch.no_grad():
        unchunked, _ = _all_reference_pair_head(
            pair_head,
            query_embeddings,
            reference_embeddings,
            None,
        )
        chunked, _ = _all_reference_pair_head(
            pair_head,
            query_embeddings,
            reference_embeddings,
            chunk_size,
        )

    diff = (chunked - unchunked).abs()
    if not torch.isfinite(diff).all():
        raise FloatingPointError("chunk comparison difference is non-finite.")

    bitwise_equal = bool(torch.equal(chunked, unchunked))
    max_abs_difference = float(diff.max().item())

    return P1ChunkComparison(
        max_abs_difference=max_abs_difference,
        bitwise_equal=bitwise_equal,
        chunk_size=chunk_size,
        query_count=int(query_embeddings.shape[0]),
        reference_count=int(reference_embeddings.shape[0]),
    )


def run_p1_all_reference_benchmark(
    *,
    encoder: Callable[[Any], Tensor],
    pair_head: P1AntisymmetricPairHead,
    query_inputs: Any,
    reference_inputs: Any,
    reference_y: Tensor,
    reference_mask: Tensor,
    scaler_scales: Tensor,
    chunk_size: int | None,
    validation_evaluations_per_cell: int | None,
    device: str | torch.device | None = None,
    warmup_iterations: int = 1,
) -> P1AllReferenceBenchmarkResult:
    """
    Operational D009 P1 all-reference validation-cost benchmark harness.

    The scientific estimator is exact: every query is paired with every
    training reference, endpoint eligibility is supplied by the training
    reference mask, and reconstruction is delegated to the qualified
    `reconstruct_absolute_from_references` implementation.

    Timing excludes data loading, checkpoint loading, model construction,
    optimizer/training work, validation-label scoring, and any caller-side
    preprocessing performed before invoking this function. Warmup executions
    are also excluded from reported timing.
    """
    if not callable(encoder):
        raise TypeError("encoder must be callable.")

    _validate_pair_head(pair_head)

    chunk_size = _validate_chunk_size(chunk_size)

    if isinstance(warmup_iterations, bool) or not isinstance(warmup_iterations, int):
        raise TypeError("warmup_iterations must be a nonnegative integer.")

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be nonnegative.")

    if validation_evaluations_per_cell is not None:
        if (
            isinstance(validation_evaluations_per_cell, bool)
            or not isinstance(validation_evaluations_per_cell, int)
        ):
            raise TypeError(
                "validation_evaluations_per_cell must be a positive integer or None."
            )
        if validation_evaluations_per_cell <= 0:
            raise ValueError(
                "validation_evaluations_per_cell must be a positive integer or None."
            )

    benchmark_device = _as_device(device)
    pair_head = pair_head.to(benchmark_device)
    pair_head.eval()

    reference_y = reference_y.to(benchmark_device)
    reference_mask = reference_mask.to(benchmark_device)
    scaler_scales = scaler_scales.to(benchmark_device)
    _validate_reference_inputs(reference_y, reference_mask, scaler_scales)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _execute_end_to_end(
                encoder,
                pair_head,
                query_inputs,
                reference_inputs,
                reference_y,
                reference_mask,
                scaler_scales,
                chunk_size,
                benchmark_device,
            )

        if benchmark_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(benchmark_device)

        (query_embeddings, reference_embeddings), stage_a_seconds = _time_call(
            benchmark_device,
            lambda: _encode_cache_embeddings(
                encoder,
                query_inputs,
                reference_inputs,
                benchmark_device,
            ),
        )

        (predicted_delta_z, chunk_ranges), stage_b_seconds = _time_call(
            benchmark_device,
            lambda: _all_reference_pair_head(
                pair_head,
                query_embeddings,
                reference_embeddings,
                chunk_size,
            ),
        )

        (absolute_mean, reference_dispersion), stage_c_seconds = _time_call(
            benchmark_device,
            lambda: _reconstruct(
                predicted_delta_z,
                reference_y,
                reference_mask,
                scaler_scales,
            ),
        )

        (
            stage_d_absolute_mean,
            stage_d_reference_dispersion,
            stage_d_chunk_ranges,
        ), stage_d_seconds = _time_call(
            benchmark_device,
            lambda: _execute_end_to_end(
                encoder,
                pair_head,
                query_inputs,
                reference_inputs,
                reference_y,
                reference_mask,
                scaler_scales,
                chunk_size,
                benchmark_device,
            ),
        )

    if stage_d_chunk_ranges != chunk_ranges:
        raise RuntimeError("Stage D chunk accounting differs from Stage B.")

    query_count = int(query_embeddings.shape[0])
    reference_count = int(reference_embeddings.shape[0])
    unique_pairs = query_count * reference_count
    directional_evaluations = 2 * unique_pairs

    if unique_pairs <= 0 or directional_evaluations <= 0:
        raise ValueError("pair accounting must be positive.")

    eligible_counts_tensor = reference_mask.sum(dim=0)
    endpoint_eligible_counts = tuple(
        int(value.item())
        for value in eligible_counts_tensor
    )

    if len(endpoint_eligible_counts) != P1_N_TASKS or min(endpoint_eligible_counts) <= 0:
        raise ValueError("endpoint-specific eligible reference counts are invalid.")

    accounting = P1PairAccounting(
        query_count=query_count,
        reference_count=reference_count,
        unique_query_reference_pairs=unique_pairs,
        directional_g_evaluations=directional_evaluations,
        endpoint_eligible_reference_counts=endpoint_eligible_counts,
        chunk_size=chunk_size,
        chunk_ranges=chunk_ranges,
    )

    if validation_evaluations_per_cell is None:
        projection = P1Projection(
            development_cell_count=DEVELOPMENT_CELL_COUNT,
            validation_evaluations_per_cell=None,
            projected_stage_d_seconds=None,
            unresolved_reason=(
                "validation_evaluations_per_cell was not supplied; projection "
                "is parameterized rather than guessed."
            ),
        )
    else:
        projection = P1Projection(
            development_cell_count=DEVELOPMENT_CELL_COUNT,
            validation_evaluations_per_cell=validation_evaluations_per_cell,
            projected_stage_d_seconds=(
                stage_d_seconds
                * validation_evaluations_per_cell
                * DEVELOPMENT_CELL_COUNT
            ),
            unresolved_reason=None,
        )

    peak_memory = None
    if benchmark_device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(benchmark_device))

    result = P1AllReferenceBenchmarkResult(
        timings=P1BenchmarkTimings(
            stage_a_encode_cache_seconds=stage_a_seconds,
            stage_b_pair_head_seconds=stage_b_seconds,
            stage_c_reconstruction_seconds=stage_c_seconds,
            stage_d_end_to_end_seconds=stage_d_seconds,
        ),
        accounting=accounting,
        projection=projection,
        absolute_mean=absolute_mean,
        reference_dispersion=reference_dispersion,
        stage_d_absolute_mean=stage_d_absolute_mean,
        stage_d_reference_dispersion=stage_d_reference_dispersion,
        peak_accelerator_memory_bytes=peak_memory,
        timing_excludes=(
            "warmup executions",
            "data loading",
            "checkpoint loading",
            "model construction",
            "training or optimizer work",
            "validation-label scoring or performance metrics",
            "caller-side preprocessing before function entry",
        ),
    )

    return result
