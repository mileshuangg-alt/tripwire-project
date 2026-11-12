from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from d012_bounded_runtime_benchmark import (  # noqa: E402
    MEASURED_COMPLETE_UPDATES,
    WARMUP_COMPLETE_UPDATES,
    LinearAnchorScheduler,
    project_workload,
)
from d012_arms import D012_FINAL_LR, D012_MAX_LR  # noqa: E402


def test_complete_update_protocol_constants() -> None:
    assert WARMUP_COMPLETE_UPDATES == 10
    assert MEASURED_COMPLETE_UPDATES == 100


def test_projection_uses_arm_specific_update_counts() -> None:
    projected = project_workload(
        b_aux_seconds=0.5,
        c_aux_seconds=0.25,
        b_target_update_seconds=0.1,
        c_target_adapt_seconds=2.0,
        b_u=388,
        c_u=857,
        target_updates_per_cell=11750,
    )

    single = projected["single_cell_seconds"]
    assert single["b_50_epoch_auxiliary"] == 50 * 388 * 0.5
    assert single["c_50_epoch_auxiliary"] == 50 * 857 * 0.25
    assert single["b_target_transfer"] == 11750 * 0.1
    assert single["c_target_adaptation"] == 2.0
    assert projected["basis"]["b_updates_per_epoch"] != projected["basis"]["c_updates_per_epoch"]


def test_scheduler_is_continuous_across_warmup_and_measurement() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([param], lr=0.0)
    scheduler = LinearAnchorScheduler(optimizer, warmup_steps=10, total_steps=110)

    for _ in range(10):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(D012_MAX_LR)

    for _ in range(100):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(D012_FINAL_LR)
    assert scheduler.step_count == 110
