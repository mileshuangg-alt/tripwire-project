"""Tests for the D013 target-adaptation qualification harness.

Stdlib plus numpy only; no CUDA, no RDKit, no upstream model. The parts that
need a GPU are exercised through injected stubs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import count_cells as cc  # noqa: E402
import qualify_adaptation as qa  # noqa: E402
import score_predictions as sp  # noqa: E402

CONFIG = HERE / "d013_target_qualification_config_v1.json"
ENDPOINTS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _label_table(tmp: Path, mols, labels) -> Path:
    """labels: {molecule: {endpoint: value or None}}"""
    cols = ["Molecule_Name", "SMILES"]
    for e in ENDPOINTS:
        cols += [f"{e}_pIC50_direct_inhibition", f"{e}_pIC50_direct_inhibition_conf_low",
                 f"{e}_pIC50_direct_inhibition_conf_high"]
    p = tmp / "labels.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for i, m in enumerate(mols):
            row = {"Molecule_Name": m, "SMILES": "C" * (i % 5 + 1)}
            for e in ENDPOINTS:
                v = labels.get(m, {}).get(e)
                row[f"{e}_pIC50_direct_inhibition"] = "" if v is None else f"{v}"
                row[f"{e}_pIC50_direct_inhibition_conf_low"] = "" if v is None else f"{v - 0.5}"
                row[f"{e}_pIC50_direct_inhibition_conf_high"] = "" if v is None else f"{v + 0.5}"
            w.writerow(row)
    return p


def _manifests(tmp: Path, cells) -> tuple[Path, Path]:
    """cells: {name: (training_names, validation_names)}"""
    d = tmp / "manifests"
    d.mkdir()
    files = {}
    for name, (train, val) in cells.items():
        p = d / f"{name}.json"
        p.write_text(json.dumps({
            "training_names": train, "validation_names": val,
            "training_count": len(train), "validation_count": len(val),
            "training_seeds": [20261101, 20261102, 20261103],
        }))
        files[p.name] = _sha(p)
    fr = d / "FREEZE_RECORD.json"
    fr.write_text(json.dumps({"files": files}))
    return d, fr


def _assignments(tmp: Path, mols, protected) -> Path:
    p = tmp / "assign.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["Molecule_Name", "cluster_id", "split"])
        for i, m in enumerate(mols):
            w.writerow([m, i, "test" if m in protected else "train"])
    return p


def _blinded(tmp: Path, mols) -> Path:
    p = tmp / "blinded.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["Molecule_Name", "SMILES"])
        for m in mols:
            w.writerow([m, "CC"])
    return p


def _world(tmp: Path, *, protected=(), blinded=(), drop_label=None, train_n=6, val_n=3):
    train = [f"M{i:03d}" for i in range(train_n)]
    val = [f"V{i:03d}" for i in range(val_n)]
    # P000 is always a protected molecule outside every cell, so the exclusion
    # check is never vacuous; `protected` adds protected molecules inside a cell.
    extra = list(protected) + list(blinded) + ["P000"]
    mols = train + val + [m for m in extra if m not in train + val]
    labels = {}
    for i, m in enumerate(mols):
        labels[m] = {e: 5.0 + 0.1 * i + 0.01 * j for j, e in enumerate(ENDPOINTS)}
    if drop_label:
        mol, ep = drop_label
        labels[mol][ep] = None
    lt = _label_table(tmp, mols, labels)
    md, fr = _manifests(tmp, {"rep_01_fold_01": (train, val)})
    return {
        "label_table": lt,
        "manifest_dir": md,
        "freeze_record": fr,
        "assignments": _assignments(tmp, mols, set(protected) | {"P000"}),
        "blinded": _blinded(tmp, blinded or ["ZZZ999"]),
        "train": train,
        "val": val,
    }


def _count_args(w, out: Path) -> list:
    return ["--config", str(CONFIG), "--config-sha256", _sha(CONFIG),
            "--manifest-dir", str(w["manifest_dir"]), "--freeze-record", str(w["freeze_record"]),
            "--label-table", str(w["label_table"]), "--label-table-sha256", _sha(w["label_table"]),
            "--split-assignments", str(w["assignments"]),
            "--split-assignments-sha256", _sha(w["assignments"]),
            "--blinded", str(w["blinded"]), "--blinded-sha256", _sha(w["blinded"]),
            "--out-dir", str(out)]


# ---------------------------------------------------------------- counting

def test_counts_are_exact_and_per_endpoint(tmp_path):
    w = _world(tmp_path, drop_label=("M000", "CYP2C9"))
    out = tmp_path / "counts"
    assert cc.main(_count_args(w, out)) == 0
    s = json.loads((out / "cell_counts.json").read_text())
    cell = s["cells"]["rep_01_fold_01"]["endpoints"]
    assert cell["CYP1A2"] == {"support": 6, "target": 3}
    assert cell["CYP2C9"] == {"support": 5, "target": 3}       # one label withheld
    assert s["partitions"] == 1 and s["protected_molecules_in_any_cell"] == 0
    assert s["largest_endpoint_by_support"] == "CYP1A2"


def test_counting_aborts_on_protected_or_blinded_molecule(tmp_path):
    w = _world(tmp_path, protected=["M002"])
    with pytest.raises(SystemExit, match="protected molecules present"):
        cc.main(_count_args(w, tmp_path / "o1"))
    (tmp_path / "b").mkdir()
    w2 = _world(tmp_path / "b", blinded=["V001"])
    with pytest.raises(SystemExit, match="protected molecules present"):
        cc.main(_count_args(w2, tmp_path / "o2"))


def test_counting_aborts_on_vacuous_protected_set(tmp_path):
    w = _world(tmp_path)
    w["assignments"] = _assignments(tmp_path, w["train"] + w["val"] + ["P000"], set())
    with pytest.raises(SystemExit, match="vacuous exclusion check"):
        cc.main(_count_args(w, tmp_path / "o"))


def _remanifest(tmp: Path, w: dict, sub: str, train, val) -> dict:
    d = tmp / sub
    d.mkdir()
    md, fr = _manifests(d, {"rep_01_fold_01": (train, val)})
    return {**w, "manifest_dir": md, "freeze_record": fr}


def test_counting_aborts_on_train_validation_overlap(tmp_path):
    w = _world(tmp_path)
    w = _remanifest(tmp_path, w, "overlap", w["train"], w["train"][:2])
    with pytest.raises(SystemExit, match="both training and validation"):
        cc.main(_count_args(w, tmp_path / "o"))


def test_counting_aborts_on_molecule_absent_from_label_table(tmp_path):
    w = _world(tmp_path)
    w = _remanifest(tmp_path, w, "ghost", w["train"] + ["GHOST"], w["val"])
    with pytest.raises(SystemExit, match="absent from the label table"):
        cc.main(_count_args(w, tmp_path / "o"))


def test_counting_refuses_bad_hashes_and_existing_out_dir(tmp_path):
    w = _world(tmp_path)
    args = _count_args(w, tmp_path / "o")
    bad = list(args)
    bad[bad.index("--label-table-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        cc.main(bad)
    (tmp_path / "exists").mkdir()
    args2 = _count_args(w, tmp_path / "exists")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cc.main(args2)
    tampered = w["manifest_dir"] / "rep_01_fold_01.json"
    tampered.write_text(tampered.read_text() + " ")
    with pytest.raises(SystemExit, match="mismatch against FREEZE_RECORD"):
        cc.main(_count_args(w, tmp_path / "o2"))


# ---------------------------------------------------------------- cell build

def _cfg() -> dict:
    return json.loads(CONFIG.read_text())


def test_build_cell_takes_only_labelled_molecules_in_manifest_order(tmp_path):
    w = _world(tmp_path, drop_label=("M003", "CYP3A4"))
    labels = cc.read_table(w["label_table"])
    m = json.loads((w["manifest_dir"] / "rep_01_fold_01.json").read_text())
    spec = qa.endpoint_spec(_cfg(), "CYP3A4")
    c = qa.build_cell(m, labels, spec)
    assert c["support_names"] == [n for n in w["train"] if n != "M003"]
    assert c["target_names"] == w["val"]
    assert c["support_count"] == 5 and c["target_count"] == 3
    assert len(c["target_conf_low"]) == len(c["target_conf_high"]) == 3
    assert all(lo < y < hi for lo, y, hi in
               zip(c["target_conf_low"], c["target_y"], c["target_conf_high"]))


def test_featurize_rejects_a_dropped_molecule():
    good = qa.featurize(["C", "CC"], lambda s: np.zeros((len(s), 4), dtype=np.float32))
    assert good.shape == (2, 4)
    with pytest.raises(SystemExit, match="failed to parse"):
        qa.featurize(["C", "bad"], lambda s: np.zeros((1, 4), dtype=np.float32))


def test_assemble_arrays_split_and_offset(tmp_path):
    w = _world(tmp_path)
    labels = cc.read_table(w["label_table"])
    m = json.loads((w["manifest_dir"] / "rep_01_fold_01.json").read_text())
    c = qa.build_cell(m, labels, qa.endpoint_spec(_cfg(), "CYP1A2"))
    xs = np.ones((c["support_count"], 4), dtype=np.float32)
    xt = np.zeros((c["target_count"], 4), dtype=np.float32)
    x, y, split = qa.assemble_arrays(c, xs, xt)
    assert x.shape == (9, 4) and split.tolist() == [1] * 6 + [0] * 3
    assert y[:6].tolist() == pytest.approx(c["support_y"])
    _, y9, _ = qa.assemble_arrays(c, xs, xt, offset=-9.0)
    assert (y - y9).tolist() == pytest.approx([9.0] * 9)


def test_scramble_touches_only_target_labels():
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=np.float32)
    s = qa.scramble_target_labels(y, 3)
    assert s[:3].tolist() == [1.0, 2.0, 3.0]
    assert s[3:].tolist() == [31.0, 21.0, 11.0]
    assert y.tolist() == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]   # input untouched


# ---------------------------------------------------------------- checks maths

def test_verdict_threshold_is_inclusive():
    assert qa.verdict("x", 1e-6, 1e-6)["passed"] is True
    assert qa.verdict("x", 1.1e-6, 1e-6)["passed"] is False
    assert qa.verdict("x", 0.0, 0.0)["passed"] is True


def test_strae_bound_is_a_real_bound_against_the_official_scorer():
    """The derived bound must hold for predictions straddling the band edges."""
    pytest.importorskip("pandas")
    sys.path.insert(0, str(HERE.parents[2]))
    from cyp.scripts.st_rae import st_rae_metric
    rng = np.random.RandomState(0)
    y = rng.uniform(4.0, 9.0, size=40)
    lo, hi = y - 0.4, y + 0.4
    a = y + rng.uniform(-1.0, 1.0, size=40)
    b = a + rng.uniform(-0.05, 0.05, size=40)
    bound = qa.strae_lipschitz_bound(a, b, y, lo, hi)
    observed = abs(st_rae_metric(y, a, lo, hi) - st_rae_metric(y, b, lo, hi))
    assert bound["strae_difference_upper_bound"] is not None
    assert observed <= bound["strae_difference_upper_bound"] + 1e-12


def test_strae_bound_scales_with_the_prediction_difference():
    y = np.array([5.0, 6.0, 7.0])
    lo, hi = y - 0.5, y + 0.5
    small = qa.strae_lipschitz_bound(y, y + 0.001, y, lo, hi)
    large = qa.strae_lipschitz_bound(y, y + 1.0, y, lo, hi)
    assert small["sum_abs_pred_difference"] == pytest.approx(0.003)
    assert large["strae_difference_upper_bound"] > small["strae_difference_upper_bound"]


def test_strae_bound_reports_none_when_the_baseline_is_degenerate():
    y = np.array([5.0, 5.0])
    b = qa.strae_lipschitz_bound(y, y + 1.0, y, y - 9.0, y + 9.0)
    assert b["strae_difference_upper_bound"] is None


# ---------------------------------------------------------------- env, phases

def test_pinned_env_is_enforced():
    assert qa.assert_pinned_env("2023.03.1", version_fn=lambda: "2023.03.1")["rdkit"] == "2023.03.1"
    with pytest.raises(SystemExit, match="refusing to featurise under RDKit 2026.03.6"):
        qa.assert_pinned_env("2023.03.1", version_fn=lambda: "2026.03.6")


def test_phases_record_each_phase_and_synchronise():
    ticks = iter([0.0, 1.5, 10.0, 12.5])
    syncs = []
    ph = qa.Phases(sync=lambda: syncs.append(1), clock=lambda: next(ticks))
    with ph("load"):
        pass
    with ph("adapt"):
        pass
    assert ph.times == {"load": 1.5, "adapt": 2.5}
    assert len(syncs) == 4            # before and after each phase


def test_config_is_the_frozen_one():
    cfg = _cfg()
    assert cfg["artifact_type"] == qa.CONFIG_ARTIFACT
    assert [e["name"] for e in cfg["endpoints"]] == ENDPOINTS
    assert cfg["pretraining_scale_offset"] == -9.0
    assert cfg["required_rdkit"] == "2023.03.1"
    assert cfg["tolerances"] == {"offset_max_abs_pred_pic50": 1e-3,
                                 "repeat_max_abs_pred_pic50": 1e-6,
                                 "scramble_max_abs_pred_pic50": 0.0}
    assert len(cfg["qualification_partitions"]) == 3 and len(cfg["seed_values"]) == 2
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        qa.read_config(CONFIG, "0" * 64)
    assert qa.read_config(CONFIG, _sha(CONFIG))["artifact_type"] == qa.CONFIG_ARTIFACT


def test_endpoint_spec_rejects_an_unknown_endpoint():
    with pytest.raises(SystemExit, match="not in the qualification config"):
        qa.endpoint_spec(_cfg(), "CYP9Z9")


def test_strae_bound_is_attained_when_every_prediction_is_outside_the_band():
    """With all predictions above the upper edge the clip is locally slope-1 for
    every molecule, so the ST-RAE difference equals the bound. This pins the
    bound's normalisation: a looser formula would not be attained."""
    pytest.importorskip("pandas")
    sys.path.insert(0, str(HERE.parents[2]))
    from cyp.scripts.st_rae import st_rae_metric
    y = np.array([5.0, 6.0, 7.0, 8.0])
    lo, hi = y - 0.3, y + 0.3
    a = y + 2.0                      # all well above the upper edge
    b = a + 0.05                     # same side, same direction
    bound = qa.strae_lipschitz_bound(a, b, y, lo, hi)
    observed = abs(st_rae_metric(y, a, lo, hi) - st_rae_metric(y, b, lo, hi))
    assert observed == pytest.approx(bound["strae_difference_upper_bound"], rel=1e-9)


# ---------------------------------------------------------------- smoke mode

def test_synthetic_cell_is_fixture_data_only():
    c = qa.synthetic_cell()
    assert c["support_count"] == qa.SMOKE_SUPPORT == 40
    assert c["target_count"] == qa.SMOKE_TARGET == 10
    assert len(c["support_smiles"]) == 40 and len(c["target_smiles"]) == 10
    assert len(c["support_y"]) == 40 and len(c["target_y"]) == 10
    assert all(n.startswith("SMOKE_") for n in c["support_names"] + c["target_names"])
    assert all(set(s) <= set("CON") and s for s in c["support_smiles"] + c["target_smiles"])
    assert all(lo < y < hi for lo, y, hi in
               zip(c["target_conf_low"], c["target_y"], c["target_conf_high"]))
    assert c["support_count"] > 5          # run_task skips the inner update at sup_num <= 5
    # Distinct molecules, so the similarity and top-k anchor paths are actually exercised
    # rather than degenerating to an all-identical matrix.
    assert len(set(c["support_smiles"] + c["target_smiles"])) >= 8


def test_synthetic_cell_assembles_like_a_real_one():
    c = qa.synthetic_cell(n_support=6, n_target=3)
    xs = np.ones((6, 4), dtype=np.float32)
    xt = np.zeros((3, 4), dtype=np.float32)
    x, y, split = qa.assemble_arrays(c, xs, xt)
    assert x.shape == (9, 4) and split.tolist() == [1] * 6 + [0] * 3
    assert y.tolist() == pytest.approx(list(c["support_y"]) + list(c["target_y"]))


@pytest.mark.parametrize("times,peak,match", [
    ({}, 1.0, "no phase times"),
    ({"adapt_total": 0.0}, 1.0, "non-positive time"),
    ({"adapt_total": -1.0}, 1.0, "non-positive time"),
    ({"adapt_total": float("nan")}, 1.0, "non-positive time"),
    ({"adapt_total": 1.0}, 0.0, "no GPU memory"),
    ({"adapt_total": 1.0}, float("nan"), "no GPU memory"),
])
def test_smoke_assertions_reject_a_run_that_did_not_happen(times, peak, match):
    with pytest.raises(SystemExit, match=match):
        qa.check_smoke(times, peak)


def test_smoke_assertions_accept_a_real_run():
    qa.check_smoke({"featurize": 0.4, "assemble": 0.01, "adapt_total": 12.0}, 1234.5)


# ---------------------------------------------------------------- split scoring

def _pred_artifact(tmp: Path, *, arms=None, n=4, bound=None) -> tuple[Path, dict]:
    y = [5.0, 6.0, 7.0, 8.0][:n]
    base = [v + 0.2 for v in y]
    arms = arms or {"pic50_seed_20261101": base,
                    "pic50_seed_20261102": base,
                    "pretraining_scale_corrected": base,
                    "target_labels_scrambled": base}
    art = {
        "artifact_type": "D013_target_adaptation_checks_v1",
        "endpoint": "CYP3A4",
        "cells": [{
            "partition": "rep_01_fold_01", "endpoint": "CYP3A4",
            "support_count": 1799, "target_count": n,
            "target_names": [f"V{i}" for i in range(n)],
            "target_y": y,
            "target_conf_low": [v - 0.3 for v in y],
            "target_conf_high": [v + 0.3 for v in y],
            "target_predictions": arms,
            "strae_derived_bounds": {
                "offset": {"strae_difference_upper_bound": 1.0 if bound is None else bound},
                "repeat": {"strae_difference_upper_bound": 1.0 if bound is None else bound},
            },
        }],
    }
    p = tmp / "checks.json"
    p.write_text(json.dumps(art))
    return p, art


def test_scoring_python_gate():
    assert sp.assert_scoring_python((3, 12))["python"] == "3.12"
    with pytest.raises(SystemExit, match="needs Python >= 3.10"):
        sp.assert_scoring_python((3, 9))


def test_split_scoring_uses_the_frozen_scorer_and_checks_the_bounds(tmp_path):
    pytest.importorskip("pandas")
    path, _ = _pred_artifact(tmp_path)
    out = tmp_path / "scores"
    assert sp.main(["--predictions", str(path), "--predictions-sha256", _sha(path),
                    "--repo-root", str(HERE.parents[2]), "--out-dir", str(out)]) == 0
    s = json.loads((out / "strae_scores.json").read_text())
    cell = s["cells"][0]
    assert s["all_bounds_hold"] is True
    assert cell["reference_arm"] == "pic50_seed_20261101"
    assert set(cell["strae_by_arm"]) == {"pic50_seed_20261101", "pic50_seed_20261102",
                                         "pretraining_scale_corrected", "target_labels_scrambled"}
    # identical predictions in every arm, so every ST-RAE difference is exactly zero
    assert {v["observed_strae_difference"] for v in cell["bound_checks"]} == {0.0}
    assert s["frozen_scorer"]["custom_scoring_functions_sha256"] == \
        _sha(HERE.parents[2] / "references" / "cyp_upstream" / "custom_scoring_functions.py")


def test_split_scoring_flags_a_violated_bound(tmp_path):
    pytest.importorskip("pandas")
    y = [5.0, 6.0, 7.0, 8.0]
    arms = {"pic50_seed_20261101": [v + 0.2 for v in y],
            "pic50_seed_20261102": [v + 3.0 for v in y],      # far apart
            "pretraining_scale_corrected": [v + 0.2 for v in y],
            "target_labels_scrambled": [v + 0.2 for v in y]}
    path, _ = _pred_artifact(tmp_path, arms=arms, bound=1e-9)   # an impossibly tight bound
    out = tmp_path / "scores"
    sp.main(["--predictions", str(path), "--predictions-sha256", _sha(path),
             "--repo-root", str(HERE.parents[2]), "--out-dir", str(out)])
    s = json.loads((out / "strae_scores.json").read_text())
    assert s["all_bounds_hold"] is False
    repeat = [v for v in s["cells"][0]["bound_checks"] if v["pair"] == "repeat"][0]
    assert repeat["bound_holds"] is False and repeat["observed_strae_difference"] > 0


def test_split_scoring_refuses_bad_hash_wrong_artifact_and_existing_out_dir(tmp_path):
    path, art = _pred_artifact(tmp_path)
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        sp.main(["--predictions", str(path), "--predictions-sha256", "0" * 64,
                 "--repo-root", str(HERE.parents[2]), "--out-dir", str(tmp_path / "o1")])
    other = tmp_path / "resource.json"
    other.write_text(json.dumps({**art, "artifact_type": "D013_target_resource_qualification_v1"}))
    with pytest.raises(SystemExit, match="not a scorable"):
        sp.main(["--predictions", str(other), "--predictions-sha256", _sha(other),
                 "--repo-root", str(HERE.parents[2]), "--out-dir", str(tmp_path / "o2")])
    (tmp_path / "exists").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        sp.main(["--predictions", str(path), "--predictions-sha256", _sha(path),
                 "--repo-root", str(HERE.parents[2]), "--out-dir", str(tmp_path / "exists")])


def test_split_scoring_rejects_a_prediction_count_mismatch(tmp_path):
    arms = {"pic50_seed_20261101": [5.2, 6.2, 7.2, 8.2], "pic50_seed_20261102": [5.2, 6.2]}
    path, _ = _pred_artifact(tmp_path, arms=arms)
    with pytest.raises(SystemExit, match="predictions for 4 targets"):
        sp.main(["--predictions", str(path), "--predictions-sha256", _sha(path),
                 "--repo-root", str(HERE.parents[2]), "--out-dir", str(tmp_path / "o")])


def test_adaptation_harness_no_longer_imports_the_scorer():
    """The pinned 3.9 environment cannot import it, so it must not appear here."""
    src = (HERE / "qualify_adaptation.py").read_text()
    assert "from st_rae import" not in src and "st_rae_metric" not in src


def test_scripts_run_under_python_isolated_mode():
    """`python -I` implies -P, so the script directory is not on sys.path. Each
    script must still resolve its sibling imports; otherwise the failure only
    appears part-way through a GPU session."""
    import subprocess
    for script in ("count_cells.py", "qualify_adaptation.py", "score_predictions.py"):
        r = subprocess.run([sys.executable, "-I", str(HERE / script), "--help"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"{script} failed under -I: {r.stderr[-400:]}"
        assert "--out-dir" in r.stdout
