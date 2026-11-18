"""Tripwire PXR-002 task-weight calibration (D018).

Chemprop 2.3.1 reduces the multitask loss as

    sum(loss * sample_weight * task_weight * mask) / sum(mask)

so task weights scale only the numerator and the denominator counts every observed
label. Equal weights therefore give each head influence proportional to its label
count. D018 fixes the *nominal* primary : combined-auxiliary allocation at 65 : 35,
equal coefficients across the three auxiliary heads (hence count-proportional
allocation among them), primary coefficient 1. The auxiliary coefficient is

    w_aux = (aux_share / primary_share) * n_primary / n_aux_total

computed from the verified eligible training label counts. The allocation is a
statement about coefficient x label-count mass, not about realized loss values or
gradients.
"""
from __future__ import annotations

PRIMARY_SHARE = 0.65
AUX_SHARE = 0.35


def calibrate_aux_coefficient(
    n_primary: int,
    n_aux_by_head: dict[str, int] | list[int] | tuple[int, ...],
    primary_share: float = PRIMARY_SHARE,
    aux_share: float = AUX_SHARE,
) -> float:
    counts = list(n_aux_by_head.values()) if isinstance(n_aux_by_head, dict) else list(n_aux_by_head)
    if n_primary <= 0 or sum(counts) <= 0 or any(c <= 0 for c in counts):
        raise ValueError("primary and every auxiliary head need a positive label count")
    if abs(primary_share + aux_share - 1.0) > 1e-12:
        raise ValueError("shares must sum to 1")
    return (aux_share / primary_share) * n_primary / sum(counts)


def task_weight_vector(n_primary: int, n_aux_by_head: dict[str, int]) -> list[float]:
    """[primary, aux head 1, aux head 2, aux head 3] with primary coefficient 1."""
    w_aux = calibrate_aux_coefficient(n_primary, n_aux_by_head)
    return [1.0] + [w_aux] * len(n_aux_by_head)


def nominal_allocation(task_weights, counts) -> list[float]:
    """Coefficient x label-count mass per head, normalized to sum to 1."""
    mass = [w * c for w, c in zip(task_weights, counts)]
    total = sum(mass)
    return [m / total for m in mass]
