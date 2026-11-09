from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import struct

import numpy as np


@dataclass(frozen=True)
class PairwiseEndpointContract:
    endpoint: str
    n_labeled: int
    pair_supervision_available: bool
    q_pair_eligible: float
    weight: float | None
    scaler_mean: float | None
    scaler_scale: float | None


@dataclass(frozen=True)
class PairwisePartitionContract:
    endpoint_order: tuple[str, ...]
    molecule_ids: tuple[str, ...]
    label_availability: tuple[tuple[bool, ...], ...]
    label_values_sha256: str
    n_molecules: int
    supervised_distinct_pair_available: bool
    endpoints: tuple[PairwiseEndpointContract, ...]



def _canonical_label_values_sha256(
    endpoint_order: tuple[str, ...],
    molecule_ids: tuple[str, ...],
    canonical_labels: tuple[tuple[float, ...], ...],
) -> str:
    """Bind canonical IDs, endpoint order, missingness, and binary64 labels."""

    def encode_text(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack(">Q", len(encoded)) + encoded

    payload = bytearray(b"D009-P1-label-values-v1\x00")

    payload.extend(struct.pack(">Q", len(endpoint_order)))
    for endpoint in endpoint_order:
        payload.extend(encode_text(endpoint))

    payload.extend(struct.pack(">Q", len(molecule_ids)))
    for molecule_id, row in zip(
        molecule_ids,
        canonical_labels,
        strict=True,
    ):
        payload.extend(encode_text(molecule_id))

        if len(row) != len(endpoint_order):
            raise RuntimeError(
                "Canonical label row does not align with endpoint order."
            )

        for value in row:
            if np.isnan(value):
                payload.extend(b"\x00")
                continue

            numeric = 0.0 if value == 0.0 else float(value)

            if not np.isfinite(numeric):
                raise RuntimeError(
                    "Canonical observed label must be finite before binding."
                )

            payload.extend(b"\x01")
            payload.extend(struct.pack(">d", numeric))

    return hashlib.sha256(payload).hexdigest()

def compute_pairwise_partition_contract(
    molecule_ids: Sequence[str],
    labels: Sequence[Sequence[object]],
    *,
    endpoint_order: Sequence[str],
    scaler_means: Sequence[float | None],
    scaler_scales: Sequence[float | None],
) -> PairwisePartitionContract:
    """
    Construct the deterministic D009 P1 pairwise partition contract.

    This function characterizes one already-declared training partition.
    It does not sample molecular pairs, construct delta targets, consume RNG
    state, calculate loss, or access validation/performance information.
    """
    endpoints = tuple(endpoint_order)

    if len(endpoints) != 4:
        raise ValueError(
            "P1 requires exactly four endpoints in declared endpoint order."
        )

    if any(not isinstance(endpoint, str) or endpoint == "" for endpoint in endpoints):
        raise ValueError(
            "Every endpoint name must be a nonempty string."
        )

    if len(set(endpoints)) != len(endpoints):
        raise ValueError(
            "Endpoint order contains duplicate endpoint names."
        )

    ids = tuple(molecule_ids)

    if len(ids) < 2:
        raise ValueError(
            "P1 pairwise partition requires at least two molecules."
        )

    if any(not isinstance(molecule_id, str) or molecule_id == "" for molecule_id in ids):
        raise ValueError(
            "Every molecule identity must be a nonempty string."
        )

    if len(set(ids)) != len(ids):
        raise ValueError(
            "Training partition contains duplicate molecule identities."
        )

    if len(labels) != len(ids):
        raise ValueError(
            "Label rows must align one-to-one with molecule identities."
        )

    if len(scaler_means) != len(endpoints):
        raise ValueError(
            "Scaler means must align exactly with endpoint order."
        )

    if len(scaler_scales) != len(endpoints):
        raise ValueError(
            "Scaler scales must align exactly with endpoint order."
        )

    canonical_order = sorted(
        range(len(ids)),
        key=lambda index: ids[index],
    )

    canonical_ids = tuple(ids[index] for index in canonical_order)

    canonical_labels = []
    availability_rows = []

    for index in canonical_order:
        row = labels[index]

        if len(row) != len(endpoints):
            raise ValueError(
                "Every label row must align exactly with endpoint order."
            )

        normalized_row = []
        availability_row = []

        for value in row:
            if value is None:
                normalized_row.append(np.nan)
                availability_row.append(False)
                continue

            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Observed labels must be numeric and finite."
                ) from exc

            if np.isnan(numeric):
                normalized_row.append(np.nan)
                availability_row.append(False)
                continue

            if not np.isfinite(numeric):
                raise ValueError(
                    "Observed labels must be numeric and finite."
                )

            normalized_row.append(numeric)
            availability_row.append(True)

        canonical_labels.append(tuple(normalized_row))
        availability_rows.append(tuple(availability_row))

    canonical_labels_tuple = tuple(canonical_labels)
    label_values_sha256 = _canonical_label_values_sha256(
        endpoints,
        canonical_ids,
        canonical_labels_tuple,
    )

    N = len(canonical_ids)

    endpoint_contracts = []

    for task_index, endpoint in enumerate(endpoints):
        n_t = sum(
            availability[task_index]
            for availability in availability_rows
        )

        if n_t < 2:
            endpoint_contracts.append(
                PairwiseEndpointContract(
                    endpoint=endpoint,
                    n_labeled=n_t,
                    pair_supervision_available=False,
                    q_pair_eligible=0.0,
                    weight=None,
                    scaler_mean=None,
                    scaler_scale=None,
                )
            )
            continue

        mean = scaler_means[task_index]
        scale = scaler_scales[task_index]

        try:
            mean_value = float(mean)
            scale_value = float(scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Endpoint {endpoint} requires valid finite scaler values."
            ) from exc

        if not np.isfinite(mean_value):
            raise ValueError(
                f"Endpoint {endpoint} requires a finite scaler mean."
            )

        if not np.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError(
                f"Endpoint {endpoint} requires a finite positive scaler scale."
            )

        q_t = (
            n_t * (n_t - 1)
            / (N * (N - 1))
        )
        w_t = (N - 1) / (n_t - 1)

        expected_weighted_eligibility = n_t / N
        observed_weighted_eligibility = q_t * w_t

        if not np.isclose(
            observed_weighted_eligibility,
            expected_weighted_eligibility,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise RuntimeError(
                f"Endpoint {endpoint} violates q_t * w_t = n_t / N."
            )

        endpoint_contracts.append(
            PairwiseEndpointContract(
                endpoint=endpoint,
                n_labeled=n_t,
                pair_supervision_available=True,
                q_pair_eligible=q_t,
                weight=w_t,
                scaler_mean=mean_value,
                scaler_scale=scale_value,
            )
        )

    supervised_distinct_pair_available = any(
        endpoint.pair_supervision_available
        for endpoint in endpoint_contracts
    )

    if not supervised_distinct_pair_available:
        raise ValueError(
            "Training partition contains no possible supervised distinct pair: "
            "every endpoint has fewer than two labeled molecules."
        )

    result = PairwisePartitionContract(
        endpoint_order=endpoints,
        molecule_ids=canonical_ids,
        label_availability=tuple(availability_rows),
        label_values_sha256=label_values_sha256,
        n_molecules=N,
        supervised_distinct_pair_available=True,
        endpoints=tuple(endpoint_contracts),
    )
    return result


@dataclass(frozen=True)
class PairwiseSampleBatch:
    endpoint_order: tuple[str, ...]
    molecule_a_ids: tuple[str, ...]
    molecule_b_ids: tuple[str, ...]
    eligibility_masks: tuple[tuple[bool, ...], ...]
    delta_targets: tuple[tuple[float, ...], ...]
    attempts: int
    no_supervision_rejections: int
    accepted_count: int
    eligible_target_count: tuple[int, ...]


def _ordered_distinct_pair_from_index(
    k: int,
    n_molecules: int,
) -> tuple[int, int]:
    """Map one uniform integer onto one ordered distinct molecule-index pair."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("Ordered-pair index must be an integer.")

    if isinstance(n_molecules, bool) or not isinstance(
        n_molecules,
        (int, np.integer),
    ):
        raise TypeError("Molecule count must be an integer.")

    n_molecules = int(n_molecules)
    k = int(k)

    if n_molecules < 2:
        raise ValueError(
            "Ordered-distinct pair mapping requires at least two molecules."
        )

    pair_count = n_molecules * (n_molecules - 1)

    if k < 0 or k >= pair_count:
        raise ValueError(
            "Ordered-pair index is outside the valid pair universe."
        )

    i = k // (n_molecules - 1)
    r = k % (n_molecules - 1)
    j = r if r < i else r + 1

    return i, j


def _draw_ordered_pair_index(
    rng: np.random.Generator,
    pair_count: int,
) -> int:
    value = rng.integers(0, pair_count)

    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError("Pair-index draw must return an integer.")

    value = int(value)

    if value < 0 or value >= pair_count:
        raise ValueError(
            "Pair-index draw returned a value outside the pair universe."
        )

    return value


def sample_supervised_pairs(
    contract: PairwisePartitionContract,
    molecule_ids: Sequence[str],
    labels: Sequence[Sequence[object]],
    *,
    endpoint_order: Sequence[str],
    scaler_means: Sequence[float | None],
    scaler_scales: Sequence[float | None],
    accepted_pair_count: int,
    rng: np.random.Generator,
) -> PairwiseSampleBatch:
    """Sample D009 supervised ordered-distinct pairs from a bound partition."""
    if isinstance(accepted_pair_count, bool) or not isinstance(
        accepted_pair_count,
        (int, np.integer),
    ):
        raise ValueError(
            "accepted_pair_count must be a positive integer."
        )

    accepted_pair_count = int(accepted_pair_count)

    if accepted_pair_count <= 0:
        raise ValueError(
            "accepted_pair_count must be a positive integer."
        )

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a numpy.random.Generator."
        )

    supplied_contract = compute_pairwise_partition_contract(
        molecule_ids,
        labels,
        endpoint_order=endpoint_order,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
    )

    if supplied_contract != contract:
        raise ValueError(
            "Sampler scientific inputs do not match the supplied "
            "partition contract."
        )

    ids = tuple(molecule_ids)

    canonical_order = sorted(
        range(len(ids)),
        key=lambda index: ids[index],
    )

    canonical_labels = []

    for index in canonical_order:
        normalized_row = []

        for value in labels[index]:
            if value is None:
                normalized_row.append(np.nan)
                continue

            numeric = float(value)

            if np.isnan(numeric):
                normalized_row.append(np.nan)
            else:
                normalized_row.append(numeric)

        canonical_labels.append(tuple(normalized_row))

    availability = contract.label_availability

    pair_count = (
        contract.n_molecules
        * (contract.n_molecules - 1)
    )

    accepted_a = []
    accepted_b = []
    accepted_masks = []
    accepted_deltas = []

    eligible_counts = [
        0
        for _ in contract.endpoint_order
    ]

    attempts = 0
    rejections = 0

    while len(accepted_a) < accepted_pair_count:
        k = _draw_ordered_pair_index(
            rng,
            pair_count,
        )

        i, j = _ordered_distinct_pair_from_index(
            k,
            contract.n_molecules,
        )

        attempts += 1

        mask = tuple(
            availability[i][task_index]
            and availability[j][task_index]
            for task_index in range(
                len(contract.endpoint_order)
            )
        )

        if not any(mask):
            rejections += 1
            continue

        deltas = []

        for task_index, eligible in enumerate(mask):
            if not eligible:
                deltas.append(np.nan)
                continue

            scale = contract.endpoints[
                task_index
            ].scaler_scale

            if scale is None:
                raise RuntimeError(
                    "Eligible endpoint is missing its bound "
                    "scaler scale."
                )

            delta = (
                canonical_labels[i][task_index]
                - canonical_labels[j][task_index]
            ) / scale

            if not np.isfinite(delta):
                raise FloatingPointError(
                    "Eligible standardized delta target "
                    "is non-finite."
                )

            deltas.append(float(delta))
            eligible_counts[task_index] += 1

        accepted_a.append(contract.molecule_ids[i])
        accepted_b.append(contract.molecule_ids[j])
        accepted_masks.append(mask)
        accepted_deltas.append(tuple(deltas))

    result = PairwiseSampleBatch(
        endpoint_order=contract.endpoint_order,
        molecule_a_ids=tuple(accepted_a),
        molecule_b_ids=tuple(accepted_b),
        eligibility_masks=tuple(accepted_masks),
        delta_targets=tuple(accepted_deltas),
        attempts=attempts,
        no_supervision_rejections=rejections,
        accepted_count=len(accepted_a),
        eligible_target_count=tuple(eligible_counts),
    )

    return result

