"""Tests for the PXR-004 timing script's arithmetic (feasibility only; D020).

Failure risks covered: an off-by-one in batches per epoch, an extrapolation that omits
validation or checkpoint time, a wall-time comparison against the wrong grouping, and a
summary that reports a mean as a median. The hardware timing itself is not unit-testable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr004_timing as tm  # noqa: E402


def test_batches_per_epoch_rounds_up_for_the_pilot_sizes():
    assert tm.batches_per_epoch(8862) == 139
    assert tm.batches_per_epoch(2897) == 46
    assert tm.batches_per_epoch(985) == 16
    assert tm.batches_per_epoch(621) == 10
    assert tm.batches_per_epoch(64) == 1 and tm.batches_per_epoch(65) == 2


def test_summarize_median_p90_and_empty():
    s = tm.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median"] == 3.0 and s["mean"] == pytest.approx(22.0) and s["p90"] == 100.0 and s["n"] == 5
    with pytest.raises(ValueError):
        tm.summarize([])


def test_worst_case_run_includes_fetch_validation_and_checkpoint_every_epoch():
    # per epoch: 10*(0.5+0.1) + 4*0.2 + 1.0 = 7.8 s; 100 epochs = 780 s
    assert tm.worst_case_run_seconds(10, 4, 0.5, 0.1, 0.2, 1.0) == pytest.approx(780.0)
    assert tm.worst_case_run_seconds(10, 4, 0.5, 0.1, 0.2, 1.0, epochs=2) == pytest.approx(15.6)


def test_pilot_totals_groupings_and_wall_comparison():
    s1, s2 = 7200.0, 3600.0  # 2 h and 1 h per run
    out = tm.pilot_totals(s1, s2, requested_wall_hours=4.0)
    assert out["serial_all_runs_hours"] == pytest.approx(3 * 2 + 6 * 1)
    assert out["sequential_arm_one_seed_hours"] == pytest.approx(3.0)
    assert out["direct_arm_one_seed_hours"] == pytest.approx(1.0)
    assert out["fits_serial_all_runs"] is False and out["fits_sequential_arm_one_seed"] is True
    assert "requested_wall_hours" not in tm.pilot_totals(s1, s2)
