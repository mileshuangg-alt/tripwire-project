from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
D012_INNER_LR = 0.01
TARGET_TASKS = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)


CORRECTED_EXECUTION_ROOT = ROOT / "qualification/cyp_external_task_qualification_20261109/d012_corrected_execution"
CORRECTED_FIRST_CELL_ROOT = CORRECTED_EXECUTION_ROOT / "first_cells"
CORRECTED_STAGED_METRICS_ROOT = CORRECTED_EXECUTION_ROOT / "staged_metrics"
CORRECTED_AGGREGATION_ROOT = CORRECTED_EXECUTION_ROOT / "aggregation"
FROZEN_T121_CELLS = ROOT / "artifacts/d009_closure_workspace/d009_t121_comparator_20261109/cells"


def chemprop_noam_target_scheduler(
    optimizer,
    *,
    updates_per_epoch: int,
    warmup_epochs: int,
    max_epochs: int,
    init_lr: float,
    max_lr: float,
    final_lr: float,
):
    """Authoritative D012 corrected target scheduler factory.

    This intentionally delegates to Chemprop 2.3.1 rather than the original
    D012 LinearAnchorScheduler. Call order remains optimizer.step() followed by
    scheduler.step(), matching the frozen production utilities.
    """
    from chemprop import schedulers

    warmup_steps = int(warmup_epochs) * int(updates_per_epoch)
    total_steps = int(max_epochs) * int(updates_per_epoch)
    return schedulers.build_NoamLike_LRSched(
        optimizer,
        warmup_steps=warmup_steps,
        cooldown_steps=total_steps - warmup_steps,
        init_lr=float(init_lr),
        max_lr=float(max_lr),
        final_lr=float(final_lr),
    )


def scalar_head_features(head: nn.Module, embeddings: Tensor) -> Tensor:
    import torch

    return head.dropout(head.activation(head.hidden(embeddings)))


def adapt_selected_output_exact_d012(
    head: nn.Module,
    support_embeddings: Tensor,
    support_targets_z: Tensor,
    *,
    inner_lr: float = D012_INNER_LR,
) -> tuple[dict[str, Tensor], Tensor]:
    """D012-corrected exact ANIL adaptation.

    Only the selected scalar output layer weight and bias are adapted. Hidden
    head parameters are included unchanged for functional_call so gradients can
    still flow through the query loss, but they are not inner-updated.
    """
    import torch
    from p2_anil_model import selected_task_prediction, standardized_mse

    params = dict(head.named_parameters())
    features = scalar_head_features(head, support_embeddings)
    weight = params["output.weight"].squeeze(0)
    bias = params["output.bias"].squeeze(0)
    support_pred = selected_task_prediction(features, task_index=0, weight=weight, bias=bias)
    support_loss = standardized_mse(support_pred, support_targets_z)
    grad_w, grad_b = torch.autograd.grad(support_loss, (weight, bias), create_graph=True, retain_graph=True)
    adapted = dict(params)
    adapted["output.weight"] = (weight - float(inner_lr) * grad_w).unsqueeze(0)
    adapted["output.bias"] = (bias - float(inner_lr) * grad_b).reshape_as(params["output.bias"])
    return adapted, support_loss


def functional_scalar_head_prediction(head: nn.Module, params: dict[str, Tensor], embeddings: Tensor) -> Tensor:
    from torch.func import functional_call

    return functional_call(head, params, (embeddings,))


def assert_frozen_t121_cell_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_name") != "D007-T121" or int(payload.get("trial_index", -1)) != 121:
        raise ValueError(f"not a frozen D007-T121 cell result: {path}")
    if payload.get("status") != "completed":
        raise ValueError(f"T121 cell result is not completed: {path}")
    best = payload.get("best_validation", {})
    if set(best.get("per_task", {})) != set(TARGET_TASKS):
        raise ValueError(f"T121 cell result has wrong endpoint set: {path}")
    return payload


def assert_not_p1_o150_artifact(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    p1_markers = {"p1_advances", "strict_advancement_rule", "endpoint_veto"}
    if p1_markers & set(payload):
        raise ValueError(f"P1 aggregate artifact is not an admissible T121 cell source: {path}")


def load_frozen_t121_cells(cells_dir: Path = FROZEN_T121_CELLS) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(cells_dir.glob("cell_*/result.json")):
        row = assert_frozen_t121_cell_result(path)
        key = f"rep_{int(row['repetition']):02d}_fold_{int(row['fold']):02d}|seed_{int(row['seed'])}"
        if key in result:
            raise ValueError(f"duplicate frozen T121 cell key: {key}")
        result[key] = row
    if len(result) != 150:
        raise ValueError(f"frozen T121 cell source must contain 150 cells, found {len(result)}")
    return result
