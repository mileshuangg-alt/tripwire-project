from __future__ import annotations

import torch
from torch import Tensor, nn


P1_EMBEDDING_DIM = 600
P1_PAIR_INPUT_DIM = 1200
P1_HIDDEN_DIM = 2000
P1_N_TASKS = 4
P1_DROPOUT = 0.4


class P1AntisymmetricPairHead(nn.Module):
    """
    D009 P1 antisymmetric pair head.

    Each molecule contributes one 600-dimensional embedding. The shared
    pair function G receives ordered concatenation [h_A, h_B]:

        1200 -> 2000 -> ReLU -> dropout(0.4) -> 4

    The reported standardized delta is:

        (G(A, B) - G(B, A)) / 2

    Antisymmetry and self-zero are required in deterministic evaluation
    mode. Ordinary dropout means they are not guaranteed in train mode.
    """

    def __init__(self) -> None:
        super().__init__()

        self.pair_function = nn.Sequential(
            nn.Linear(
                P1_PAIR_INPUT_DIM,
                P1_HIDDEN_DIM,
            ),
            nn.ReLU(),
            nn.Dropout(P1_DROPOUT),
            nn.Linear(
                P1_HIDDEN_DIM,
                P1_N_TASKS,
            ),
        )

    def forward(
        self,
        h_a: Tensor,
        h_b: Tensor,
    ) -> Tensor:
        if h_a.shape != h_b.shape:
            raise ValueError(
                "h_a and h_b must have identical shapes."
            )

        if (
            h_a.ndim != 2
            or h_a.shape[1] != P1_EMBEDDING_DIM
        ):
            raise ValueError(
                "P1 embeddings must have shape (batch, 600)."
            )

        if not torch.isfinite(h_a).all():
            raise ValueError(
                "h_a must contain only finite values."
            )

        if not torch.isfinite(h_b).all():
            raise ValueError(
                "h_b must contain only finite values."
            )

        ab = self.pair_function(
            torch.cat(
                (h_a, h_b),
                dim=1,
            )
        )

        ba = self.pair_function(
            torch.cat(
                (h_b, h_a),
                dim=1,
            )
        )

        result = (ab - ba) / 2.0

        if not torch.isfinite(result).all():
            raise FloatingPointError(
                "P1 pair-head output is non-finite."
            )

        return result


def normalized_weighted_pair_mse(
    preds: Tensor,
    targets: Tensor,
    mask: Tensor,
    weights: Tensor,
) -> Tensor:
    """
    Compute the D009 normalized weighted pairwise MSE.

    The approved objective is:

        sum(M * w * squared_error) / sum(M * w)

    Ineligible target entries may be NaN at the pair-target boundary.
    They are explicitly sanitized before any subtraction or squaring;
    zero multiplication is not used as NaN masking.
    """
    if (
        preds.ndim != 2
        or preds.shape[1] != P1_N_TASKS
    ):
        raise ValueError(
            "preds must have shape (batch, 4)."
        )

    if targets.shape != preds.shape:
        raise ValueError(
            "targets must have the same shape as preds."
        )

    if mask.shape != preds.shape:
        raise ValueError(
            "mask must have the same shape as preds."
        )

    if mask.dtype is not torch.bool:
        raise TypeError(
            "mask must be boolean."
        )

    if (
        weights.ndim != 1
        or weights.shape[0] != P1_N_TASKS
    ):
        raise ValueError(
            "weights must have shape (4,)."
        )

    if not torch.isfinite(preds).all():
        raise ValueError(
            "preds must contain only finite values."
        )

    if (
        not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError(
            "weights must be finite and positive."
        )

    if not mask.any():
        raise ValueError(
            "pair batch contains no eligible targets."
        )

    if not torch.isfinite(targets[mask]).all():
        raise ValueError(
            "eligible targets must be finite."
        )

    # D009 requires sanitization before arithmetic.
    safe_targets = torch.where(
        mask,
        targets,
        torch.zeros_like(targets),
    )

    task_weights = weights.unsqueeze(0).expand_as(
        preds
    )

    weighted_mask = (
        mask.to(dtype=preds.dtype)
        * task_weights
    )

    denominator = weighted_mask.sum()

    if (
        not torch.isfinite(denominator)
        or denominator <= 0
    ):
        raise ValueError(
            "weighted loss denominator must be "
            "finite and positive."
        )

    squared_error = (
        preds - safe_targets
    ).square()

    result = (
        squared_error * weighted_mask
    ).sum() / denominator

    if not torch.isfinite(result):
        raise FloatingPointError(
            "weighted pair loss is non-finite."
        )

    return result


def reconstruct_absolute_from_references(
    predicted_delta_z: Tensor,
    reference_y: Tensor,
    reference_mask: Tensor,
    scaler_scales: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Reconstruct D009 absolute pIC50 predictions from training references.

    Shapes
    ------
    predicted_delta_z:
        (queries, references, 4)

    reference_y:
        (references, 4), original-pIC50 training labels.
        Missing labels may be NaN.

    reference_mask:
        (references, 4), True exactly where the reference has an
        observed endpoint label.

    scaler_scales:
        (4,), training-partition endpoint scales.

    Returns
    -------
    absolute_mean:
        (queries, 4), arithmetic mean across eligible training-reference
        reconstructions in original pIC50 units.

    reference_dispersion:
        (queries, 4), population standard deviation across eligible
        reference-derived absolute predictions. This is diagnostic only
        and is not calibrated uncertainty.
    """
    if (
        predicted_delta_z.ndim != 3
        or predicted_delta_z.shape[2] != P1_N_TASKS
    ):
        raise ValueError(
            "predicted_delta_z must have shape "
            "(queries, references, 4)."
        )

    n_references = predicted_delta_z.shape[1]

    if reference_y.shape != (
        n_references,
        P1_N_TASKS,
    ):
        raise ValueError(
            "reference_y must have shape (references, 4)."
        )

    if reference_mask.shape != reference_y.shape:
        raise ValueError(
            "reference_mask must match reference_y shape."
        )

    if reference_mask.dtype is not torch.bool:
        raise TypeError(
            "reference_mask must be boolean."
        )

    if scaler_scales.shape != (P1_N_TASKS,):
        raise ValueError(
            "scaler_scales must have shape (4,)."
        )

    if not torch.isfinite(predicted_delta_z).all():
        raise ValueError(
            "predicted standardized deltas must be finite."
        )

    if (
        not torch.isfinite(scaler_scales).all()
        or (scaler_scales <= 0).any()
    ):
        raise ValueError(
            "scaler scales must be finite and positive."
        )

    if not reference_mask.any(dim=0).all():
        raise ValueError(
            "every endpoint requires at least one eligible reference."
        )

    if not torch.isfinite(reference_y[reference_mask]).all():
        raise ValueError(
            "eligible reference labels must be finite."
        )

    # Sanitize missing reference labels before arithmetic.
    safe_reference_y = torch.where(
        reference_mask,
        reference_y,
        torch.zeros_like(reference_y),
    )

    reconstructed = (
        safe_reference_y.unsqueeze(0)
        + predicted_delta_z
        * scaler_scales.view(1, 1, P1_N_TASKS)
    )

    expanded_mask = reference_mask.unsqueeze(0).expand(
        predicted_delta_z.shape[0],
        -1,
        -1,
    )

    safe_reconstructed = torch.where(
        expanded_mask,
        reconstructed,
        torch.zeros_like(reconstructed),
    )

    counts = reference_mask.sum(dim=0).to(
        dtype=predicted_delta_z.dtype
    )

    absolute_mean = (
        safe_reconstructed.sum(dim=1)
        / counts.unsqueeze(0)
    )

    centered = torch.where(
        expanded_mask,
        reconstructed - absolute_mean.unsqueeze(1),
        torch.zeros_like(reconstructed),
    )

    reference_dispersion = torch.sqrt(
        centered.square().sum(dim=1)
        / counts.unsqueeze(0)
    )

    if not torch.isfinite(absolute_mean).all():
        raise FloatingPointError(
            "absolute reference mean is non-finite."
        )

    if not torch.isfinite(reference_dispersion).all():
        raise FloatingPointError(
            "reference dispersion is non-finite."
        )

    result = (
        absolute_mean,
        reference_dispersion,
    )
    return result
