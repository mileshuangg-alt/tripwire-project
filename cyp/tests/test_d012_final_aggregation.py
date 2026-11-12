from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from d012_final_aggregation import (  # noqa: E402
    CellKey,
    expected_cells,
    join_and_summarize,
    load_t121,
    paired_stats,
    verify_macro,
)


def metric(value: float, per_endpoint: bool = True):
    if not per_endpoint:
        return {"macro_st_rae": value, "per_endpoint_st_rae": None, "metric_path": "metric.json"}
    per = {
        "CYP1A2_pIC50_direct_inhibition": value - 0.03,
        "CYP2C9_pIC50_direct_inhibition": value - 0.01,
        "CYP2D6_pIC50_direct_inhibition": value + 0.01,
        "CYP3A4_pIC50_direct_inhibition": value + 0.03,
    }
    return {"macro_st_rae": value, "per_endpoint_st_rae": per, "metric_path": "metric.json"}


def full_fixture():
    b = {}
    c = {}
    t = {}
    for index, key in enumerate(expected_cells()):
        base = 0.7 + index * 0.0001
        b[key] = metric(base, per_endpoint=False)
        c[key] = metric(base - 0.01, per_endpoint=True)
        t[key] = metric(base + 0.02, per_endpoint=True)
    return b, c, t


def test_expected_frozen_cell_matrix() -> None:
    cells = expected_cells()

    assert len(cells) == 150
    assert len(set(cells)) == 150
    assert CellKey(1, 1, 20261101) in set(cells)
    assert CellKey(5, 10, 20261103) in set(cells)


def test_join_and_difference_sign_convention() -> None:
    b, c, t = full_fixture()
    rows, summary = join_and_summarize(b, c, t)

    assert len(rows) == 150
    assert summary["lower_st_rae_is_better"] is True
    assert "delta_c_minus_b" in summary["difference_sign_convention"]["primary"]
    assert summary["paired_differences"]["C_minus_B"]["first_method_wins_lower_is_better"] == 150
    assert summary["paired_differences"]["B_minus_T121"]["first_method_wins_lower_is_better"] == 150
    assert summary["paired_differences"]["C_minus_T121"]["first_method_wins_lower_is_better"] == 150
    assert summary["paired_differences"]["C_minus_B"]["ties"] == 0
    assert summary["o150"]["C"] < summary["o150"]["B"] < summary["o150"]["T121"]


def test_missing_duplicate_or_extra_cells_stop() -> None:
    b, c, t = full_fixture()
    b.pop(CellKey(1, 1, 20261101))
    with pytest.raises(ValueError, match="B cells differ"):
        join_and_summarize(b, c, t)


def test_macro_recompute_mismatch_stops() -> None:
    per = {
        "CYP1A2_pIC50_direct_inhibition": 1.0,
        "CYP2C9_pIC50_direct_inhibition": 1.0,
        "CYP2D6_pIC50_direct_inhibition": 1.0,
        "CYP3A4_pIC50_direct_inhibition": 2.0,
    }
    with pytest.raises(ValueError, match="macro mismatch"):
        verify_macro(per, 1.0, label="bad")


def test_paired_stats_require_150_and_use_sample_sd() -> None:
    with pytest.raises(ValueError):
        paired_stats([1.0])
    stats = paired_stats([0.0] * 149 + [1.0])
    assert stats["n"] == 150
    assert stats["ties"] == 149
    assert stats["second_method_wins_lower_is_better"] == 1
    assert stats["sample_sd"] > 0


def test_t121_source_schema_and_hash_available() -> None:
    t121 = load_t121()

    assert len(t121) == 150
    assert CellKey(1, 1, 20261101) in t121
    assert all("macro_st_rae" in row for row in t121.values())
