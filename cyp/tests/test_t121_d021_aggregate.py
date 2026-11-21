import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import t121_d021_aggregate as agg  # noqa: E402


def write_cell(root, index, macro_by_task, screen_sha, *, tamper=False):
    cell = root / f"cell_{index:03d}"
    cell.mkdir(parents=True)
    macro = sum(macro_by_task.values()) / 4
    result = {
        "status": "completed",
        "cell_index": index,
        "repetition": index // 30 + 1,
        "fold": (index // 3) % 10 + 1,
        "seed": 20261101 + index % 3,
        "split_manifest_sha256": f"split{index // 3}",
        "screen_checkpoint_sha256": screen_sha,
        "macro_st_rae": macro,
        "best_validation": {"macro_st_rae": macro, "per_task": dict(macro_by_task)},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    (cell / "result.json").write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (cell / "COMPLETED.json").write_text(json.dumps(
        {"status": "completed", "result_sha256": "0" * 64 if tamper else digest}))


def make_run(root, screen_sha, base):
    for index in range(agg.N_CELLS):
        tasks = {t: base + 0.01 * k for k, t in enumerate(agg.TASKS)}
        write_cell(root, index, tasks, screen_sha)
    return root


def make_all(tmp_path, offsets):
    runs = {}
    for name in agg.RUN_NAMES:
        runs[name] = make_run(tmp_path / name, f"screen_{name}", offsets[name])
    return runs


OFFSETS = {"r1": 0.60, "r2": 0.62, "r3": 0.64, "d1": 0.59, "d2": 0.63, "d3": 0.60}


def test_delta_and_direction(tmp_path):
    dirs = make_all(tmp_path, OFFSETS)
    runs = {n: agg.load_run(p) for n, p in dirs.items()}
    report = agg.aggregate(runs)
    assert report["delta_s"]["s1"] == pytest.approx(-0.01)
    assert report["delta_s"]["s2"] == pytest.approx(0.01)
    assert report["delta_s"]["s3"] == pytest.approx(-0.04)
    assert report["delta_bar"] == pytest.approx(-0.04 / 3)
    assert report["d_favoured"] is True
    assert report["runs"]["r1"]["O150"] == pytest.approx(0.615)


def test_zero_delta_is_not_favoured(tmp_path):
    offsets = dict(OFFSETS, d1=0.60, d2=0.62, d3=0.64)
    runs = {n: agg.load_run(p) for n, p in make_all(tmp_path, offsets).items()}
    assert agg.aggregate(runs)["d_favoured"] is False


def test_missing_cell_stops(tmp_path):
    root = make_run(tmp_path / "x", "s", 0.6)
    (root / "cell_007" / "COMPLETED.json").unlink()
    with pytest.raises(ValueError, match="not completed"):
        agg.load_run(root)


def test_incomplete_cell_stops(tmp_path):
    root = make_run(tmp_path / "x", "s", 0.6)
    (root / "cell_007" / "INCOMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        agg.load_run(root)


def test_tampered_result_hash_stops(tmp_path):
    root = tmp_path / "x"
    for index in range(agg.N_CELLS):
        write_cell(root, index, {t: 0.6 for t in agg.TASKS}, "s", tamper=(index == 3))
    with pytest.raises(ValueError, match="hash differs"):
        agg.load_run(root)


def test_mixed_screening_checkpoints_stop(tmp_path):
    root = make_run(tmp_path / "x", "s", 0.6)
    for index in (0,):
        shutil_dir = root / f"cell_{index:03d}"
        for f in shutil_dir.iterdir():
            f.unlink()
        shutil_dir.rmdir()
        write_cell(root, index, {t: 0.6 for t in agg.TASKS}, "other")
    with pytest.raises(ValueError, match="one screening checkpoint"):
        agg.load_run(root)


def test_wrong_cell_count_stops(tmp_path):
    root = make_run(tmp_path / "x", "s", 0.6)
    for f in (root / "cell_149").iterdir():
        f.unlink()
    (root / "cell_149").rmdir()
    with pytest.raises(ValueError, match="expected 150"):
        agg.load_run(root)


def test_r1_reproduction_mismatch_refuses_to_write(tmp_path):
    dirs = make_all(tmp_path, OFFSETS)
    out = tmp_path / "report.json"
    argv = ["--output", str(out)]
    for name, path in dirs.items():
        argv += ["--run", f"{name}={path}"]
    with pytest.raises(ValueError, match="does not reproduce"):
        agg.main(argv)
    assert not out.exists()


def test_output_not_overwritten(tmp_path):
    out = tmp_path / "report.json"
    out.write_text("x")
    with pytest.raises(FileExistsError):
        agg.main(["--output", str(out), "--run", "r1=/nonexistent"])


def test_duplicate_repetition_fold_seed_stops(tmp_path):
    root = make_run(tmp_path / "x", "s", 0.6)
    result_path = root / "cell_001" / "result.json"
    result = json.loads(result_path.read_text())
    result["seed"] = 20261101  # same triple as cell 0
    text = json.dumps(result, indent=2, sort_keys=True)
    result_path.write_text(text)
    (root / "cell_001" / "COMPLETED.json").write_text(json.dumps(
        {"status": "completed", "result_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    with pytest.raises(ValueError, match="duplicate"):
        agg.load_run(root)


def test_three_seeds_per_repetition_fold_is_valid(tmp_path):
    root = make_run(tmp_path / "x", "s", 0.6)
    assert len(agg.load_run(root)["cells"]) == agg.N_CELLS
