import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t121_d022_aggregate as agg  # noqa: E402
import t121_d021_aggregate as base  # noqa: E402
from test_t121_d021_aggregate import make_run  # noqa: E402

OFFSETS = {"r1": 0.60, "r2": 0.62, "r3": 0.64, "c1": 0.59, "c2": 0.63, "c3": 0.60}


def load(tmp_path, offsets):
    return {n: base.load_run(make_run(tmp_path / n, f"screen_{n}", o)) for n, o in offsets.items()}


def test_paired_differences_mean_and_direction(tmp_path):
    report = agg.aggregate(load(tmp_path, OFFSETS))
    assert report["delta_s"] == {k: pytest.approx(v) for k, v in
                                 {"s1": -0.01, "s2": 0.01, "s3": -0.04}.items()}
    assert report["delta_bar"] == pytest.approx(-0.04 / 3)
    assert report["chemeleon_preferred"] is True
    assert report["O_bar_R"] == pytest.approx(0.62 + 0.015)
    assert set(report["per_isoform_delta_s"]["s1"]) == set(base.TASKS)
    assert report["per_isoform_delta_bar"][base.TASKS[0]] == pytest.approx(-0.04 / 3)


def test_zero_or_positive_mean_is_not_preferred(tmp_path):
    zero = agg.aggregate(load(tmp_path / "a", dict(OFFSETS, c1=0.60, c2=0.62, c3=0.64)))
    assert zero["chemeleon_preferred"] is False
    worse = agg.aggregate(load(tmp_path / "b", dict(OFFSETS, c1=0.70, c2=0.70, c3=0.70)))
    assert worse["chemeleon_preferred"] is False


def test_run_set_and_unique_screens_are_enforced(tmp_path):
    runs = load(tmp_path, OFFSETS)
    with pytest.raises(ValueError, match="exactly"):
        agg.aggregate({k: v for k, v in runs.items() if k != "c3"})
    runs["c3"] = dict(runs["c3"], screen_checkpoint_sha256=runs["r3"]["screen_checkpoint_sha256"])
    with pytest.raises(ValueError, match="share a screening checkpoint"):
        agg.aggregate(runs)
