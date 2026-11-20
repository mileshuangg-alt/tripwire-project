"""Tests for the PXR-004 reporting script (D020; descriptive reporting only).

Failure risks covered: a paired difference with the wrong sign or arm order, a metrics file
paired with the wrong predictions, arms scored on different validation rows, a metrics file
whose arm/seed does not match its path, a missing run being silently skipped, and a context
point that would be hardcoded instead of passed in.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pxr.scripts import pxr004_report as rp  # noqa: E402

TRUE = [5.0, 6.0, 7.0, 8.0]


def _write_run(root: Path, arm: str, seed: int, pred_shift: float, spearman: float, ids=None,
               metrics_override: dict | None = None) -> None:
    d = root / f"seed{seed}" / f"{arm}_seed{seed}"
    (d / "stage1").mkdir(parents=True)
    (d / "stage2").mkdir(parents=True)
    pred = [t + pred_shift * (1 if i % 2 else -1) for i, t in enumerate(TRUE)]
    ids = ids or ["a", "b", "c", "d"]
    with (d / "stage2" / "validation_predictions_best.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "pEC50_true", "pEC50_pred"])
        for r, t, p in zip(ids, TRUE, pred):
            w.writerow([r, t, p])
    mae, rmse = rp.mae_rmse(TRUE, pred)
    m = {"stage": 2, "arm": arm, "seed": seed, "best_primary_validation_mae": mae,
         "best_primary_validation_rmse": rmse, "best_primary_validation_spearman": spearman,
         "best_epoch": 3, "epochs_run": 24}
    m.update(metrics_override or {})
    (d / "stage2" / "metrics.json").write_text(json.dumps(m))
    if arm == "sequential":
        (d / "stage1" / "metrics.json").write_text(json.dumps({
            "stage": 1, "seed": seed, "best_monitor_pooled_mse": 0.5, "best_epoch": 2, "epochs_run": 23,
            "best_heads": {"log2fc_33uM": {"mae": 0.3}, "log2fc_8p25uM": {"mae": 0.2}}}))


def _pilot(tmp_path: Path, seeds=(1, 2)) -> Path:
    for s in seeds:
        _write_run(tmp_path, "direct", s, 0.5, 0.80)
        _write_run(tmp_path, "sequential", s, 0.6, 0.75)
    return tmp_path


def test_mae_rmse_values():
    mae, rmse = rp.mae_rmse([0.0, 0.0], [1.0, -3.0])
    assert mae == pytest.approx(2.0) and rmse == pytest.approx(math.sqrt(5.0))


def test_paired_difference_is_sequential_minus_direct(tmp_path):
    rep = rp.build_report(_pilot(tmp_path), [1, 2])
    for s in (1, 2):
        assert rep["paired_sequential_minus_direct"]["mae"][s] == pytest.approx(0.1)
        assert rep["paired_sequential_minus_direct"]["spearman"][s] == pytest.approx(-0.05)
    assert rep["mean_paired_sequential_minus_direct"]["mae"] == pytest.approx(0.1)
    assert rep["arm_means"]["direct"]["mae"] == pytest.approx(0.5)
    assert rep["n_validation"] == 4


def test_markdown_contains_tables_and_passed_context_only(tmp_path):
    rep = rp.build_report(_pilot(tmp_path), [1, 2])
    md = rp.render_markdown(rep, [("floor", 0.9218)])
    assert "Paired differences (sequential minus direct)" in md and "| floor | 0.9218 |" in md
    assert "Context points" not in rp.render_markdown(rep, [])


def test_missing_run_raises(tmp_path):
    root = _pilot(tmp_path, seeds=(1,))
    with pytest.raises(FileNotFoundError):
        rp.build_report(root, [1, 2])


def test_metrics_arm_mismatch_raises(tmp_path):
    _write_run(tmp_path, "direct", 1, 0.5, 0.8, metrics_override={"arm": "sequential"})
    _write_run(tmp_path, "sequential", 1, 0.6, 0.75)
    with pytest.raises(ValueError, match="expected"):
        rp.build_report(tmp_path, [1])


def test_predictions_not_matching_metrics_raise(tmp_path):
    _write_run(tmp_path, "direct", 1, 0.5, 0.8, metrics_override={"best_primary_validation_mae": 0.123})
    _write_run(tmp_path, "sequential", 1, 0.6, 0.75)
    with pytest.raises(ValueError, match="disagrees"):
        rp.build_report(tmp_path, [1])


def test_arms_on_different_rows_raise(tmp_path):
    _write_run(tmp_path, "direct", 1, 0.5, 0.8)
    _write_run(tmp_path, "sequential", 1, 0.6, 0.75, ids=["a", "b", "c", "x"])
    with pytest.raises(ValueError, match="different validation rows"):
        rp.build_report(tmp_path, [1])


def test_duplicate_seeds_and_bad_context_rejected(tmp_path):
    with pytest.raises(ValueError):
        rp.build_report(_pilot(tmp_path), [1, 1])
    with pytest.raises(ValueError):
        rp.parse_context(["no-equals-sign"])
    assert rp.parse_context(["ECFP4 + LightGBM=0.6049"]) == [("ECFP4 + LightGBM", 0.6049)]


def test_out_json_refuses_existing_file(tmp_path):
    root = _pilot(tmp_path / "run")
    out = tmp_path / "out.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        rp.main(["--pilot-root", str(root), "--seeds", "1", "2", "--out-json", str(out)])
